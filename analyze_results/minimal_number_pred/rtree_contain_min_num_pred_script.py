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
import pickle
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
class RTreeEstimator:
    """
    Spatial selectivity estimator using pre-built R-tree models for contain filter operations
    """
    def __init__(self, data_dir="../../large_files"):
        """
        Initialize the R-tree based estimator using pre-built models
        
        Parameters:
        -----------
        data_dir : str
            Directory containing R-tree models
        """
        self.data_dir = data_dir
        self.rtree_dir = f"{data_dir}/traditional_methods/rtree/models"
        self.results_dir = f"{data_dir}/traditional_methods/rtree/results/contain"
        self.viz_dir = f"{data_dir}/traditional_methods/rtree/visualizations/contain"
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.universe_boundaries = {}
        self.dataset_sizes = {}
        self.level_nodes = {}
        self.model_metadata = {}
        self.cache = {}  # Query result cache
        
        # Load dataset metadata
        self.load_spatial_statistics()
        
        # Load available models
        self.load_available_models()
    
    def load_spatial_statistics(self):
        """Load dataset information from spatial_statistics.csv"""
        try:
            stats_df = pd.read_csv("../../spatial_statistics.csv")
            for _, row in stats_df.iterrows():
                table_name = row['Table Name']
                total_objects = row['Total Spatial Objects']
                bbox_str = row['Universe Limits (Bounding Box)']
                
                # Parse bounding box
                bbox = self.parse_bbox(bbox_str)
                self.universe_boundaries[table_name] = bbox
                self.dataset_sizes[table_name] = int(total_objects)
                
            print(f"Loaded metadata for {len(self.universe_boundaries)} datasets")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error loading spatial statistics: {e}")
            sys.stdout.flush()
    
    def parse_bbox(self, bbox_str):
        """Parse bounding box string into coordinates"""
        pattern = r"BOX\(([-\d\.]+) ([-\d\.]+),([-\d\.]+) ([-\d\.]+)\)"
        match = re.search(pattern, bbox_str)
        if match:
            xmin = float(match.group(1))
            ymin = float(match.group(2))
            xmax = float(match.group(3))
            ymax = float(match.group(4))
            return (xmin, ymin, xmax, ymax)
        return (-180, -90, 180, 90)  # Default if parsing fails
    
    def load_available_models(self):
        """Load all available pre-built R-tree models"""
        if not os.path.exists(self.rtree_dir):
            print(f"Model directory not found: {self.rtree_dir}")
            return
        
        # Check for metadata file first
        metadata_file = f"{self.rtree_dir}/all_rtree_metadata.csv"
        if os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
            for _, row in metadata_df.iterrows():
                dataset_name = row['dataset']
                self.model_metadata[dataset_name] = row.to_dict()
        
        # Load level nodes for each dataset
        loaded_count = 0
        for dataset_name in self.universe_boundaries.keys():
            level_nodes_path = f"{self.rtree_dir}/{dataset_name}_level_nodes.pkl"
            metadata_path = f"{self.rtree_dir}/{dataset_name}_metadata.json"
            
            if os.path.exists(level_nodes_path):
                try:
                    with open(level_nodes_path, 'rb') as f:
                        self.level_nodes[dataset_name] = pickle.load(f)
                    
                    # Load metadata if not already loaded
                    if dataset_name not in self.model_metadata and os.path.exists(metadata_path):
                        self.model_metadata[dataset_name] = pd.read_json(metadata_path, typ='series').to_dict()
                    
                    loaded_count += 1
                except Exception as e:
                    print(f"Error loading model for {dataset_name}: {e}")
        
        print(f"Loaded {loaded_count} pre-built R-tree models")
        sys.stdout.flush()
    
    def parse_mbr(self, mbr_str):
        """Parse MBR string from format like '(x1, y1, x2, y2)'"""
        if isinstance(mbr_str, str):
            coords = mbr_str.strip('"()').split(', ')
            return [float(coord) for coord in coords]
        return mbr_str  # Already parsed
    
    def estimate_contain_count(self, dataset_name, query_mbr):
        """
        Estimate the number of objects that are contained within a query rectangle
        using the level-before-leaves approach
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to query against
        query_mbr : list or str
            Query rectangle as [xmin, ymin, xmax, ymax] or '(xmin, ymin, xmax, ymax)'
            
        Returns:
        --------
        float
            Estimated number of objects contained within the query
        """
        # Check if result is in cache
        if isinstance(query_mbr, list):
            query_mbr = tuple(query_mbr)
        cache_key = f"{dataset_name}_{query_mbr}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Parse query rectangle if needed
        if isinstance(query_mbr, str):
            query_mbr = self.parse_mbr(query_mbr)
        
        # Query coordinates
        q_xmin, q_ymin, q_xmax, q_ymax = query_mbr
        query_area = (q_xmax - q_xmin) * (q_ymax - q_ymin)
        
        # If query area is 0, return 0
        if query_area <= 0:
            return 0
        
        # Check if model is available
        if dataset_name not in self.level_nodes:
            print(f"No R-tree model found for {dataset_name}")
            return 0
        
        # Get level nodes and total objects
        nodes = self.level_nodes[dataset_name]
        total_objects = self.dataset_sizes[dataset_name]
        
        # Count how many nodes are contained within the query and their coverage ratio
        total_node_objects = sum(node['objects'] for node in nodes)
        if total_node_objects <= 0:
            return 0  # Avoid division by zero
            
        contained_objects = 0
        
        for node in nodes:
            node_mbr = node['mbr']
            node_objects = node['objects']
            
            # Check if node is fully contained within the query MBR
            if (node_mbr[0] >= q_xmin and node_mbr[2] <= q_xmax and 
                node_mbr[1] >= q_ymin and node_mbr[3] <= q_ymax):
                # Node is fully contained in query - add all objects
                contained_objects += node_objects
            elif (node_mbr[0] < q_xmax and node_mbr[2] > q_xmin and 
                  node_mbr[1] < q_ymax and node_mbr[3] > q_ymin):
                # Node partially overlaps query - estimate contained fraction
                intersection_xmin = max(q_xmin, node_mbr[0])
                intersection_ymin = max(q_ymin, node_mbr[1])
                intersection_xmax = min(q_xmax, node_mbr[2])
                intersection_ymax = min(q_ymax, node_mbr[3])
                
                intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
                node_area = node['area']
                
                if node_area > 0:
                    # Calculate overlap ratio and add proportional count
                    overlap_ratio = intersection_area / node_area
                    contained_objects += node_objects * overlap_ratio
        
        # Scale to match total objects if needed
        estimated_count = contained_objects * (total_objects / total_node_objects)
        
        # Cache and return result
        estimated_count = max(0, round(estimated_count))
        self.cache[cache_key] = estimated_count
        return estimated_count
    
    def evaluate_on_dataset(self, dataset_name, results_file=None, sample_ratio=0.2):
        """
        Evaluate the R-tree based contain estimation method on a dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to evaluate
        results_file : str
            Path to the file containing actual query results
        sample_ratio : float
            Fraction of dataset to use (0.2 = 20%)
            
        Returns:
        --------
        dict
            Evaluation results including MAE, MAPE, q-score, and model metadata
        """
        if not results_file:
            results_file = f"../../large_files/resultsContains/{dataset_name}_results.csv"
            
        if not os.path.exists(results_file):
            raise ValueError(f"Results file not found: {results_file}")
            
        # Check if model is available
        if dataset_name not in self.level_nodes:
            raise ValueError(f"No R-tree model found for {dataset_name}")
            
        # Load query results
        print(f"Loading query results from {results_file}")
        sys.stdout.flush()
        
        try:
            results_df = pd.read_csv(results_file)
            
            # Sample only a portion of the dataset
            sample_size = max(1, int(len(results_df) * sample_ratio))
            print(f"Using {sample_ratio*100}% sample: {sample_size} out of {len(results_df)} queries")
            sys.stdout.flush()
            sampled_results = results_df.sample(n=sample_size, random_state=42)
            
            # Prepare arrays for evaluation
            actual_counts = []
            estimated_counts = []
            estimation_times = []
            
            # Process each query with progress reporting
            print(f"Processing {dataset_name} queries: ", end="", flush=True)
            sys.stdout.flush()
            
            total_queries = len(sampled_results)
            progress_step = max(1, total_queries // 10)
            
            for i, (index, row) in enumerate(sampled_results.iterrows()):
                # Show progress every 10%
                if i % progress_step == 0 or i == total_queries - 1:
                    progress = (i+1) / total_queries * 100
                    print(f"{progress:.1f}%... ", end="", flush=True)
                    sys.stdout.flush()
                    
                query_mbr = self.parse_mbr(row['Query MBR'])
                actual_count = row['Count MBR']
                
                # Measure estimation time
                start_time = time.time()
                estimated_count = self.estimate_contain_count(dataset_name, query_mbr)
                end_time = time.time()
                estimation_time = (end_time - start_time) * 1000  # ms
                
                actual_counts.append(actual_count)
                estimated_counts.append(estimated_count)
                estimation_times.append(estimation_time)
            
            print("Done!")
            sys.stdout.flush()
            
            # Convert to arrays for calculations
            actual_counts = np.array(actual_counts)
            estimated_counts = np.array(estimated_counts)
            estimation_times = np.array(estimation_times)
            
            # Calculate MAE
            mae = mean_absolute_error(actual_counts, estimated_counts)
            
            # Calculate MAPE with handling for zeros
            non_zero_mask = actual_counts != 0
            zero_mask = ~non_zero_mask
            mape_sum = 0
            count = len(actual_counts)
            
            if np.any(non_zero_mask):
                mape_sum += np.sum(np.abs((actual_counts[non_zero_mask] - estimated_counts[non_zero_mask]) / 
                                        actual_counts[non_zero_mask]))
            
            if np.any(zero_mask):
                mape_sum += np.sum(np.abs(actual_counts[zero_mask] - estimated_counts[zero_mask]) / 100)
            
            mape = mape_sum / count
            
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
                
            avg_time_ms = np.mean(estimation_times)
            
            # Get model metadata
            model_size_bytes = 0
            level_nodes_size_bytes = 0
            total_size_bytes = 0
            rtree_params = {}
            num_level_nodes = len(self.level_nodes.get(dataset_name, []))
            
            if dataset_name in self.model_metadata:
                meta = self.model_metadata[dataset_name]
                model_size_bytes = meta.get('model_size_bytes', 0)
                level_nodes_size_bytes = meta.get('level_nodes_size_bytes', 0)
                total_size_bytes = meta.get('total_size_bytes', 0)
                rtree_params = meta.get('rtree_params', {})
                if isinstance(rtree_params, str):
                    # Handle JSON parsing if needed
                    try:
                        rtree_params = eval(rtree_params)
                    except:
                        rtree_params = {}
            
            # Combine results with model metadata
            results = {
                'Dataset': dataset_name,
                'Method': 'RTree-Level-Contain',
                'MAE': mae,
                'MAPE': mape,
                'Q_Score': q_score,
                'Avg_Time_ms': avg_time_ms,
                'Num_Queries': len(sampled_results),
                'Sample_Ratio': sample_ratio,
                'Model_Size_MB': total_size_bytes / (1024*1024),
                'Level_Nodes_Size_MB': level_nodes_size_bytes / (1024*1024),
                'Num_Level_Nodes': num_level_nodes
            }
            
            # Add R-tree parameters to results
            for key, value in rtree_params.items():
                results[f'rtree_{key}'] = value
            
            # Save results
            results_file_out = f"{self.results_dir}/{dataset_name}_evaluation_sample{int(sample_ratio*100)}.csv"
            pd.DataFrame([results]).to_csv(results_file_out, index=False)
            
            # Generate visualization
            self.visualize_results(dataset_name, actual_counts, estimated_counts, sample_ratio)
            
            print(f"Evaluation results for {dataset_name} ({sample_ratio*100}% sample):")
            print(f"  MAE: {mae:.2f}")
            print(f"  MAPE: {mape:.2%}")
            print(f"  Q-Score: {q_score:.2f}")
            print(f"  Avg. Estimation Time: {avg_time_ms:.4f} ms")
            print(f"  Model Size: {results['Model_Size_MB']:.2f} MB")
            print(f"  Num Level Nodes: {num_level_nodes}")
            sys.stdout.flush()
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            sys.stdout.flush()
            raise
    
    def visualize_results(self, dataset_name, actual_counts, estimated_counts, sample_ratio=0.2):
        """Create visualization of actual vs. predicted counts"""
        os.makedirs(self.viz_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 10))
        plt.scatter(actual_counts, estimated_counts, alpha=0.5, s=8)
        
        max_val = max(np.max(actual_counts), np.max(estimated_counts))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        plt.xlabel('Actual Count')
        plt.ylabel('Estimated Count')
        plt.title(f'R-tree Contain Estimation for {dataset_name} ({int(sample_ratio*100)}% sample)')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(
            f"{self.viz_dir}/{dataset_name}_estimation_sample{int(sample_ratio*100)}.png", 
            dpi=150
        )
        plt.close()
        
        # Create a comparison for a sample of queries
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
            label='R-tree Estimate', 
            s=100, alpha=0.7, marker='x', color='blue'
        )
        
        plt.xlabel('Query Index')
        plt.ylabel('Object Count')
        plt.title(
            f'R-tree Contain vs. Actual Count for {dataset_name} - '
            f'Sample of {sample_size} Queries ({int(sample_ratio*100)}% dataset)'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(
            f"{self.viz_dir}/{dataset_name}_comparison_sample{int(sample_ratio*100)}.png", 
            dpi=150
        )
        plt.close()

# %%
rtrees_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files"
datasets_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsContains/"

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

def test_model_contain_intersect(model, dataset_name, data, min_y=0):
    # Extract query MBR column (needs parsing as it's in string format)
    def parse_mbr(mbr_str):
        coords = mbr_str.strip('"()').split(', ')
        return [float(coord) for coord in coords]
    
    # Extract columns - use list comprehension for better performance
    print("Parsing MBR coordinates...")
    Rectangles = np.array([parse_mbr(mbr) for mbr in data['Query MBR']])
    Y = data[['Count MBR']].values  # Using Count MBR as target

    X_train, Rectangles, y_train, Y = train_test_split(Rectangles, Y, test_size=0.2, random_state=3)

    # Filter data points where Y > min_y
    if min_y > 0:
        mask = Y.flatten() > min_y
        Rectangles = Rectangles[mask]
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

    X = Rectangles

    # Estimate counts using the rtree model

    # Original code with tqdm progress bar
    estimated_counts = np.array([
        model.estimate_contain_count(dataset_name, mbr) 
        for mbr in tqdm(X, desc="Estimating counts")
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

estimator = RTreeEstimator(rtrees_dir)
dataset_names = list(estimator.level_nodes.keys())

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
    data = pd.read_csv(data_path, usecols=['Query MBR', 'Count MBR'])
    sample_size = len(data)

    min_y_values = get_min_y_values(sample_size)

    for min_y in min_y_values:
        print(f"\nTesting with min_y = {min_y}")
        mae, mape, sample_count = test_model_contain_intersect(estimator, dataset_name, data, min_y=min_y)
        
        # Add result to dataframe
        results_df = pd.concat([results_df, pd.DataFrame({
            'model': 'RTree',
            'dataset': dataset_name,
            'min_y': [min_y],
            'MAE': [mae],
            'MAPE': [mape],
            'sample_count': [sample_count]
        })], ignore_index=True)
        
        print(f"MAE for RTree/{dataset_name} (min_y={min_y}): {mae}")
        print(f"MAPE for RTree/{dataset_name} (min_y={min_y}): {mape}")
        
        # Save results to CSV
        results_df.to_csv("rtree_contain_min_y_experiment_results_20.csv", index=False)
        print(f"\nResults saved to rtree_contain_min_y_experiment_results_20.csv")
        sys.stdout.flush()
        results_df
