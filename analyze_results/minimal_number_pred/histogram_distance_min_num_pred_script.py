# %%
# Cell 1: Imports
import os
import re
import glob
import joblib
from collections import defaultdict
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import gc
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import time
import sys
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Define min_y values following the pattern
def get_min_y_values(max_value):
    """Generate sequence 0, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000, etc. up to max_value"""
    sequence = [0]
    
    # Powers of 10
    power = 1
    while 10**power <= max_value:
        sequence.append(10**power)
        
        # Add 2× and 5× multipliers for larger powers
        if 10**power >= 1000:  # Starting from 1000
            if 5 * 10**power <= max_value:
                sequence.append(5 * 10**power)
        
        if 10**power >= 10000:  # Starting from 10000
            if 2 * 10**power <= max_value:
                sequence.append(2 * 10**power)
                
        power += 1
    
    return sorted(sequence)

class HistogramDistanceEstimator:
    def __init__(self, histograms_dir="../../large_files/traditional_methods/histogram"):
        """
        Initialize the histogram-based distance estimator
        """
        self.histograms_dir = histograms_dir
        self.histograms = {}
        self.metadata = {}
        self.cache = {}
        
        os.makedirs(f"{histograms_dir}/results/distance", exist_ok=True)
        os.makedirs(f"{histograms_dir}/visualizations/distance", exist_ok=True)
        
        self.load_histograms()
    
    def load_histograms(self):
        """Load all available histograms from the histograms directory."""
        files = os.listdir(self.histograms_dir)
        histogram_files = [f for f in files if f.endswith('_histogram.npy')]
        
        for hf in histogram_files:
            dataset_name = hf.replace('_histogram.npy', '')
            metadata_file = f"{dataset_name}_metadata.npy"
            
            if os.path.exists(os.path.join(self.histograms_dir, metadata_file)):
                print(f"Loading histogram for {dataset_name}...")
                sys.stdout.flush()
                self.histograms[dataset_name] = np.load(
                    os.path.join(self.histograms_dir, hf)
                )
                self.metadata[dataset_name] = np.load(
                    os.path.join(self.histograms_dir, metadata_file),
                    allow_pickle=True
                ).item()
                
        print(f"Loaded {len(self.histograms)} histograms")
        sys.stdout.flush()

    def parse_mbr(self, mbr_str):
        """Parse MBR string from '(x1, y1, x2, y2)'."""
        if isinstance(mbr_str, str):
            coords = mbr_str.strip('"()').split(', ')
            return [float(coord) for coord in coords]
        return mbr_str
    
    def estimate_distance_count(self, dataset_name, object_mbr, min_distance, max_distance):
        """
        Optimized method to estimate objects within a distance range using rectangle pre-filtering
        """
        # Handle caching and input parsing as before
        if isinstance(object_mbr, list):
            object_mbr = tuple(object_mbr)
        
        cache_key = f"{dataset_name}_{object_mbr}_{min_distance}_{max_distance}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if dataset_name not in self.histograms:
            raise ValueError(f"No histogram found for {dataset_name}")
        
        if isinstance(object_mbr, str):
            object_mbr = self.parse_mbr(object_mbr)
            object_mbr = tuple(object_mbr)
        
        grid = self.histograms[dataset_name]
        metadata = self.metadata[dataset_name]
        grid_dim_x, grid_dim_y = metadata['dimensions']
        univ_xmin, univ_ymin, univ_xmax, univ_ymax = metadata['universe']
        
        # Extract object MBR coordinates and calculate center
        o_xmin, o_ymin, o_xmax, o_ymax = object_mbr
        o_center_x = (o_xmin + o_xmax) / 2
        o_center_y = (o_ymin + o_ymax) / 2
        
        # Calculate cell dimensions
        cell_width = (univ_xmax - univ_xmin) / grid_dim_x
        cell_height = (univ_ymax - univ_ymin) / grid_dim_y
        
        # Calculate inner and outer rectangle bounds based on min/max distances
        inner_rect = (
            o_center_x - min_distance, 
            o_center_y - min_distance, 
            o_center_x + min_distance, 
            o_center_y + min_distance
        )
        
        outer_rect = (
            o_center_x - max_distance, 
            o_center_y - max_distance, 
            o_center_x + max_distance, 
            o_center_y + max_distance
        )
        
        # Convert rectangle bounds to grid cell indices
        inner_min_i = max(0, int((inner_rect[0] - univ_xmin) / cell_width))
        inner_min_j = max(0, int((inner_rect[1] - univ_ymin) / cell_height))
        inner_max_i = min(grid_dim_x - 1, int((inner_rect[2] - univ_xmin) / cell_width))
        inner_max_j = min(grid_dim_y - 1, int((inner_rect[3] - univ_ymin) / cell_height))
        
        outer_min_i = max(0, int((outer_rect[0] - univ_xmin) / cell_width))
        outer_min_j = max(0, int((outer_rect[1] - univ_ymin) / cell_height))
        outer_max_i = min(grid_dim_x - 1, int((outer_rect[2] - univ_xmin) / cell_width))
        outer_max_j = min(grid_dim_y - 1, int((outer_rect[3] - univ_ymin) / cell_height))
        
        # Create a mask for all cells in the outer rectangle
        outer_mask = np.zeros((grid_dim_x, grid_dim_y), dtype=bool)
        outer_mask[outer_min_i:outer_max_i+1, outer_min_j:outer_max_j+1] = True
        
        # Create a mask for all cells in the inner rectangle
        inner_mask = np.zeros((grid_dim_x, grid_dim_y), dtype=bool)
        inner_mask[inner_min_i:inner_max_i+1, inner_min_j:inner_max_j+1] = True
        
        # Get the cells between the inner and outer rectangles
        candidate_cells = outer_mask & ~inner_mask
        
        # Create cell center coordinates only for candidate cells
        candidate_indices = np.where(candidate_cells)
        cell_centers_x = univ_xmin + (candidate_indices[0] + 0.5) * cell_width
        cell_centers_y = univ_ymin + (candidate_indices[1] + 0.5) * cell_height
        
        # Calculate distances only for candidate cells
        distances = np.sqrt((cell_centers_x - o_center_x)**2 + (cell_centers_y - o_center_y)**2)
        
        # Create distance mask for accurate filtering
        in_range_mask = (distances >= min_distance) & (distances <= max_distance)
        
        # Sum objects only from cells that are truly in range
        valid_cells = candidate_indices[0][in_range_mask], candidate_indices[1][in_range_mask]
        objects_in_range = np.sum(grid[valid_cells])
        
        # Add any objects in the inner rectangle when min_distance is 0
        if min_distance == 0:
            objects_in_range += np.sum(grid[inner_min_i:inner_max_i+1, inner_min_j:inner_max_j+1])
        
        # Cache and return the result
        self.cache[cache_key] = objects_in_range
        return objects_in_range
    
    def evaluate_on_dataset(self, dataset_name, results_file=None, sample_ratio=0.2):
        if not results_file:
            results_file = f"../../large_files/resultsDistance/{dataset_name}_results.csv"
        
        if not os.path.exists(results_file):
            raise ValueError(f"Results file not found: {results_file}")
        
        print(f"Loading query results from {results_file}")
        sys.stdout.flush()
        
        # Load results file - assume CSV format with Object MBR, Distance Min, Distance Max, Count MBR
        results_df = pd.read_csv(results_file)
        
        sample_size = max(1, int(len(results_df) * sample_ratio))
        print(f"Using {sample_ratio*100}% sample: {sample_size} out of {len(results_df)} queries")
        sys.stdout.flush()
        
        sampled_results = results_df.sample(n=sample_size, random_state=42)
        
        actual_counts = []
        estimated_counts = []
        estimation_times = []
        
        # Use simple progress reporting instead of relying solely on tqdm
        print(f"Processing {dataset_name} queries: ", end="", flush=True)
        sys.stdout.flush()
        
        total_queries = len(sampled_results)
        progress_step = max(1, total_queries // 10)
        
        for i, (index, row) in enumerate(sampled_results.iterrows()):
            # Show simple progress every 10%
            if i % progress_step == 0 or i == total_queries - 1:
                progress = (i+1) / total_queries * 100
                print(f"{progress:.1f}%... ", end="", flush=True)
                sys.stdout.flush()
            
            # Parse query parameters - object MBR, min and max distance
            object_mbr = self.parse_mbr(row['Object MBR'])
            min_distance = row['Distance Min']
            max_distance = row['Distance Max']
            actual_count = row['Count MBR']
            
            start_time = time.time()
            estimated_count = self.estimate_distance_count(dataset_name, object_mbr, min_distance, max_distance)
            end_time = time.time()
            
            actual_counts.append(actual_count)
            estimated_counts.append(estimated_count)
            estimation_times.append((end_time - start_time) * 1000)
        
        print("Done!")
        sys.stdout.flush()
        
        # Convert to arrays for calculations
        actual_counts = np.array(actual_counts)
        estimated_counts = np.array(estimated_counts)
        estimation_times = np.array(estimation_times)
        
        # Ensure non-negative estimates
        estimated_counts = np.maximum(0, estimated_counts)
        
        # Calculate MAE
        mae = mean_absolute_error(actual_counts, estimated_counts)
        
        # Calculate MAPE with handling for zeros
        non_zero_mask = (actual_counts != 0)
        zero_mask = ~non_zero_mask
        mape_sum = 0
        count = len(actual_counts)
        
        if np.any(non_zero_mask):
            mape_sum += np.sum(
                np.abs((actual_counts[non_zero_mask] - estimated_counts[non_zero_mask]) / actual_counts[non_zero_mask])
            )
        
        if np.any(zero_mask):
            mape_sum += np.sum(np.abs(actual_counts[zero_mask] - estimated_counts[zero_mask]) / 100)
        
        mape = mape_sum / count if count > 0 else 0
        
        # Calculate q-score
        valid_indices = (actual_counts != 0) & (estimated_counts != 0)
        if np.any(valid_indices):
            ratios = np.maximum(
                estimated_counts[valid_indices] / actual_counts[valid_indices],
                actual_counts[valid_indices] / estimated_counts[valid_indices]
            )
            q_score = np.mean(ratios)
        else:
            q_score = float('inf')
        
        avg_time_ms = np.mean(estimation_times) if len(estimation_times) > 0 else 0
        
        results = {
            'Dataset': dataset_name,
            'MAE': mae,
            'MAPE': mape,
            'Q_Score': q_score,
            'Avg_Time_ms': avg_time_ms,
            'Num_Queries': len(sampled_results),
            'Sample_Ratio': sample_ratio
        }
        
        results_df_out = pd.DataFrame([results])
        results_df_out.to_csv(
            f"{self.histograms_dir}/results/distance/{dataset_name}_evaluation_sample{int(sample_ratio*100)}.csv",
            index=False
        )
        
        # Generate visualization
        self.visualize_results(dataset_name, actual_counts, estimated_counts, sample_ratio)
        
        print(f"Evaluation results for {dataset_name} ({sample_ratio*100}% sample):")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape:.2%}")
        print(f"  Q-Score: {q_score:.2f}")
        print(f"  Avg. Estimation Time: {avg_time_ms:.4f} ms")
        sys.stdout.flush()
        
        return results
    
    def visualize_results(self, dataset_name, actual_counts, estimated_counts, sample_ratio=0.2):
        plt.figure(figsize=(12, 10))
        plt.scatter(actual_counts, estimated_counts, alpha=0.5, s=8)
        
        max_val = max(np.max(actual_counts), np.max(estimated_counts))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        plt.xlabel('Actual Count')
        plt.ylabel('Estimated Count')
        plt.title(f'Histogram-based Distance Estimation for {dataset_name} ({int(sample_ratio*100)}% sample)')
        plt.grid(True, alpha=0.3)
        
        out_path = f"{self.histograms_dir}/visualizations/distance/{dataset_name}_estimation_sample{int(sample_ratio*100)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        sample_size = min(100, len(actual_counts))
        indices = np.random.choice(len(actual_counts), sample_size, replace=False)
        
        plt.figure(figsize=(20, 10))
        plt.scatter(
            range(sample_size),
            actual_counts[indices],
            label='Actual Count',
            s=100, alpha=0.7, marker='o', color='green'
        )
        plt.scatter(
            range(sample_size),
            estimated_counts[indices],
            label='Histogram Estimate',
            s=100, alpha=0.7, marker='x', color='blue'
        )
        
        plt.xlabel('Query Index')
        plt.ylabel('Object Count')
        plt.title(
            f'Histogram Estimation vs. Actual Count for {dataset_name} - '
            f'Sample of {sample_size} Queries ({int(sample_ratio*100)}% dataset)'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path_comp = f"{self.histograms_dir}/visualizations/distance/{dataset_name}_comparison_sample{int(sample_ratio*100)}.png"
        plt.savefig(out_path_comp, dpi=150)
        plt.close()


# %%
histograms_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/traditional_methods/histogram/"
datasets_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsDistance/"

# %%
from sklearn.metrics import mean_absolute_error as MAE

def MAPE(actual_values, predicted_values):
    """Calculate Mean Absolute Percentage Error with special handling for zeros"""
    # Vectorized implementation
    actual_flat = actual_values.flatten()
    pred_flat = predicted_values.flatten()
    
    # Create mask for non-zero actual values
    non_zero_mask = actual_flat != 0
    zero_mask = ~non_zero_mask
    
    # Calculate MAPE for non-zero elements
    mape_sum = 0
    count = len(actual_flat)
    
    if np.any(non_zero_mask):
        mape_sum += np.sum(np.abs((actual_flat[non_zero_mask] - pred_flat[non_zero_mask]) / actual_flat[non_zero_mask]))
    
    if np.any(zero_mask):
        mape_sum += np.sum(np.abs(actual_flat[zero_mask] - pred_flat[zero_mask]) / 100)
    
    return mape_sum / count

def test_model_distance(model, dataset_name, data, min_y=0):
    # Extract columns - use list comprehension for better performance
    print("Parsing MBR coordinates...")
    Objects_MBR = np.array([model.parse_mbr(mbr) for mbr in data['Object MBR']])
    Distance_Min = data['Distance Min'].values.reshape(-1, 1)
    Distance_Max = data['Distance Max'].values.reshape(-1, 1)

    Y = data[['Count MBR']].values  # Using Count MBR as target

    # Determine the number of coordinates in each MBR
    mbr_dimensions = Objects_MBR.shape[1] if len(Objects_MBR.shape) > 1 else 4
    
    # Stack the arrays properly
    X = np.hstack((
        Objects_MBR,
        Distance_Min,
        Distance_Max,
    ))

    X_train, X, y_train, Y = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Filter data points where Y > min_y
    if min_y > 0:
        mask = Y.flatten() > min_y
        X = X[mask]
        Y = Y[mask]
        print(f"Filtered to {len(Y)} samples with count > {min_y}")

    # Calculate basic statistics
    total_samples = len(Y)
    if total_samples > 0:
        max_count = float(np.max(Y))
        min_count = float(np.min(Y))
        mean_count = float(np.mean(Y))
        median_count = float(np.median(Y))
    else:
        max_count = min_count = mean_count = median_count = 0

    print(f"Max count: {max_count}")
    print(f"Min count: {min_count}")
    print(f"Mean count: {mean_count:.2f}")
    print(f"Median count: {median_count:.2f}")
    print(f"Total samples: {total_samples}\n")

    # Fixed version with proper MBR extraction
    estimated_counts = np.array([
        model.estimate_distance_count(
            dataset_name,
            row[:mbr_dimensions].reshape(-1).tolist(),  # Extract MBR coordinates as a list
            row[mbr_dimensions],  # Distance_Min
            row[mbr_dimensions + 1]   # Distance_Max
        )
        for row in tqdm(X, desc="Estimating distance counts", unit="query")
    ])
    
    estimated_counts = np.maximum(0, estimated_counts)
    actual_counts = Y.flatten()
    actual_counts = np.maximum(0, actual_counts)
    # Calculate MAE
    mae_value = MAE(actual_counts, estimated_counts)
    # Calculate MAPE
    mape_value = MAPE(actual_counts, estimated_counts)

    return mae_value, mape_value, total_samples

# %%
# Create a dataframe to store results
results_df = pd.DataFrame(columns=['model', 'dataset', 'min_y', 'MAE', 'MAPE', 'sample_count'])

estimator = HistogramDistanceEstimator(histograms_dir)
dataset_names = list(estimator.histograms.keys())

print(dataset_names)

# Process each dataset with clear separation
for idx, dataset_name in enumerate(dataset_names, start=1):
    print("\n" + "="*80)
    print(f"DATASET {idx}/{len(dataset_names)}: {dataset_name}")
    print("="*80)
    sys.stdout.flush()

    gc.collect()

    csv_file = f"{dataset_name}_results.csv"
    # Load dataset - only load required columns
    data_path = os.path.join(datasets_dir, csv_file)
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, usecols=['Object MBR', 'Distance Min', 'Distance Max', 'Count MBR'])
    sample_size = len(data)

    min_y_values = get_min_y_values(sample_size)

    for min_y in min_y_values:
        print(f"\nTesting with min_y = {min_y}")
        mae, mape, sample_count = test_model_distance(estimator, dataset_name, data, min_y=min_y)
        
        # Add result to dataframe
        results_df = pd.concat([results_df, pd.DataFrame({
            'model': 'Histogram',
            'dataset': dataset_name,
            'min_y': [min_y],
            'MAE': [mae],
            'MAPE': [mape],
            'sample_count': [sample_count]
        })], ignore_index=True)
        
        print(f"MAE for Histogram/{dataset_name} (min_y={min_y}): {mae}")
        print(f"MAPE for Histogram/{dataset_name} (min_y={min_y}): {mape}")
        
        # Save results to CSV
        results_df.to_csv("histogram_distance_min_y_experiment_results_20.csv", index=False)
        print(f"\nResults saved to histogram_distance_min_y_experiment_results_20.csv")
        sys.stdout.flush()
        results_df
