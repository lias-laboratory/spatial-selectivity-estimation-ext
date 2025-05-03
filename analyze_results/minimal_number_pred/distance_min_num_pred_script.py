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

# %%
models_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/LearnedModels/distance"
datasets_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsDistance"

# %%
def find_max_sample_models(models_dir):

    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
    
    # Dictionary to store results
    max_models = {}
    loaded_models = {}
    
    for model_type in model_dirs:
        model_path = os.path.join(models_dir, model_type)
        
        # Get all joblib files in this model directory
        model_files = glob.glob(os.path.join(model_path, '*.joblib'))
        
        # Group models by dataset
        dataset_models = defaultdict(list)
        
        # Parse each model filename
        for file_path in model_files:
            filename = os.path.basename(file_path)
            
            # Parse the filename using regex
            # Pattern assumes: datasetname_modeltype_samplesize_otherinfo.joblib
            match = re.match(r'^([a-zA-Z0-9_]+)_[a-zA-Z0-9_]+_([0-9]+)_.*\.joblib$', filename)
            
            if match:
                dataset = match.group(1)
                sample_size = int(match.group(2))
                
                # Add to list of models for this dataset
                dataset_models[dataset].append({
                    'path': file_path,
                    'sample_size': sample_size,
                    'filename': filename
                })
        
        # Find max sample size model for each dataset
        max_models[model_type] = {}
        loaded_models[model_type] = {}
        
        for dataset, models in dataset_models.items():
            # Sort by sample size (descending)
            sorted_models = sorted(models, key=lambda x: x['sample_size'], reverse=True)
            
            if sorted_models:
                # Get the model with max sample size
                max_model = sorted_models[0]
                max_models[model_type][dataset] = max_model
                
                print(f"Keeping {model_type}/{dataset} model with {max_model['sample_size']} samples")
                
    rows = []
    for model_type, datasets in max_models.items():
        for dataset, info in datasets.items():
            rows.append({
                'Model Type': model_type,
                'Dataset': dataset,
                'Sample Size': info['sample_size'],
                'Model Path': info['path']
            })

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values(['Model Type', 'Dataset'], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    summary_df

    return max_models, summary_df

def load_model(max_model):
    return joblib.load(max_model['path'])


# %%
max_models, df = find_max_sample_models(models_dir)

# %%
df

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

def test_model_contain_intersect(model_type, model, dataset_name, min_y=0):
    
    csv_file = f"{dataset_name}_results.csv"

    spatial_stats = pd.read_csv('/home/adminlias/nadir/Spatial-Selectivity-Ext/spatial_statistics.csv')


    # Load dataset - only load required columns
    data_path = os.path.join(datasets_dir, csv_file)
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, usecols=['Object MBR', 'Distance Min', 'Distance Max', 'Count MBR'])

    
    # Extract query MBR column (needs parsing as it's in string format)
    def parse_mbr(mbr_str):
        coords = mbr_str.strip('"()').split(', ')
        return [float(coord) for coord in coords]
    
    def parse_bbox(bbox_str):
        # Extract coordinates from BOX string using regex
        pattern = r"BOX\(([-\d\.]+) ([-\d\.]+),([-\d\.]+) ([-\d\.]+)\)"
        match = re.search(pattern, bbox_str)
        if match:
            xmin = float(match.group(1))
            ymin = float(match.group(2))
            xmax = float(match.group(3))
            ymax = float(match.group(4))
            return xmin, ymin, xmax, ymax
        return -180, -90, 180, 90  # Default if parsing fails
    
    # Extract universe boundaries for each dataset
    universe_boundaries = {}
    for _, row in spatial_stats.iterrows():
        table_name = row['Table Name']
        bbox = parse_bbox(row['Universe Limits (Bounding Box)'])
        universe_boundaries[table_name] = bbox

    # Get universe boundaries for this dataset
    if dataset_name in universe_boundaries:
        univ_xmin, univ_ymin, univ_xmax, univ_ymax = universe_boundaries[dataset_name]
    else:
        # Default values if dataset not found in spatial stats
        univ_xmin, univ_ymin, univ_xmax, univ_ymax = -180, -90, 180, 90
    
    print(f"Universe boundaries for {dataset_name}: ({univ_xmin}, {univ_ymin}, {univ_xmax}, {univ_ymax})")
    
    # Extract columns - use list comprehension for better performance
    print("Parsing MBR coordinates...")
    Objects_MBR = np.array([parse_mbr(mbr) for mbr in data['Object MBR']])
    Distance_Min = data['Distance Min'].values.reshape(-1, 1)
    Distance_Max = data['Distance Max'].values.reshape(-1, 1)

    Y = data[['Count MBR']].values  # Using Count MBR as target

    # Extract MBR coordinates
    x1 = Objects_MBR[:, 0].reshape(-1, 1)  # Left
    y1 = Objects_MBR[:, 1].reshape(-1, 1)  # Bottom
    x2 = Objects_MBR[:, 2].reshape(-1, 1)  # Right
    y2 = Objects_MBR[:, 3].reshape(-1, 1)  # Top

    # Calculate MBR center points
    obj_x = (x1 + x2) / 2  # Center X
    obj_y = (y1 + y2) / 2  # Center Y

    # Calculate MBR dimensions
    mbr_width = (x2 - x1)
    mbr_height = (y2 - y1)
    mbr_area = mbr_width * mbr_height

    # Normalized coordinates of MBR center (0-1 range within universe)
    norm_x = (obj_x - univ_xmin) / (univ_xmax - univ_xmin) if (univ_xmax - univ_xmin) != 0 else 0.5
    norm_y = (obj_y - univ_ymin) / (univ_ymax - univ_ymin) if (univ_ymax - univ_ymin) != 0 else 0.5

    # Distance range
    distance_range = Distance_Max - Distance_Min

    # Distance ratio (max/min)
    # Avoid division by zero
    min_non_zero = np.where(Distance_Min == 0, 0.0001, Distance_Min)
    distance_ratio = Distance_Max / min_non_zero

    if model_type == 'KNN':
        X = np.hstack((
            obj_x,           # X coordinate of MBR center
            obj_y,           # Y coordinate of MBR center
            # mbr_width,       # Width of MBR
            # mbr_height,      # Height of MBR
            # mbr_area,        # Area of MBR
            # norm_x,          # Normalized X position (0-1)
            # norm_y,          # Normalized Y position (0-1)
            Distance_Min,    # Minimum distance
            Distance_Max,    # Maximum distance
            # distance_range,  # Range of distance
            # distance_ratio.reshape(-1, 1)  # Ratio of max/min distance
        ))
    else:
        X = np.hstack((
            obj_x,           # X coordinate of MBR center
            obj_y,           # Y coordinate of MBR center
            mbr_width,       # Width of MBR
            mbr_height,      # Height of MBR
            mbr_area,        # Area of MBR
            norm_x,          # Normalized X position (0-1)
            norm_y,          # Normalized Y position (0-1)
            Distance_Min,    # Minimum distance
            Distance_Max,    # Maximum distance
            distance_range,  # Range of distance
            distance_ratio.reshape(-1, 1)  # Ratio of max/min distance
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



    # If no samples left, return NaN for metrics
    if total_samples == 0:
        return float('nan'), float('nan'), 0

    print("Making predictions...")
    y_pred = model.predict(X).reshape(-1, 1)  # Reshape to match y_test_all format
    # Ensure no negative predictions
    y_pred = np.maximum(y_pred, 0)

    mae_value = MAE(Y, y_pred)
    mape_value = MAPE(Y, y_pred)

    return mae_value, mape_value, total_samples

# %%
# Create a dataframe to store results
results_df = pd.DataFrame(columns=['model', 'dataset', 'min_y', 'MAE', 'MAPE', 'sample_count'])

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

for model_type, datasets in max_models.items():
    if model_type == 'DT' : continue
    for dataset, info in datasets.items():
        gc.collect()
        model = load_model(info)
        print(f"Loaded model {model_type} for {dataset} with sample size {info['sample_size']}")
        
        # Generate min_y values up to sample size
        min_y_values = get_min_y_values(info['sample_size'])
        
        for min_y in min_y_values:
            print(f"\nTesting with min_y = {min_y}")
            mae, mape, sample_count = test_model_contain_intersect(model_type, model, dataset, min_y=min_y)
            
            # Add result to dataframe
            results_df = pd.concat([results_df, pd.DataFrame({
                'model': [model_type],
                'dataset': [dataset],
                'min_y': [min_y],
                'MAE': [mae],
                'MAPE': [mape],
                'sample_count': [sample_count]
            })], ignore_index=True)
            
            print(f"MAE for {model_type}/{dataset} (min_y={min_y}): {mae}")
            print(f"MAPE for {model_type}/{dataset} (min_y={min_y}): {mape}")

        # Save results to CSV
        results_df.to_csv("distance_min_y_experiment_results_20.csv", index=False)
        print(f"\nResults saved to min_y_experiment_results.csv")
        results_df


