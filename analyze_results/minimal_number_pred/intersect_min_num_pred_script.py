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
models_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/LearnedModels/intersect"
datasets_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsIntersects"

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

def test_model_contain_intersect(model, dataset_name, min_y=0):
    
    csv_file = f"{dataset_name}_results.csv"

    # Load dataset - only load required columns
    data_path = os.path.join(datasets_dir, csv_file)
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path, usecols=['Query MBR', 'Count MBR'])
    
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
            mae, mape, sample_count = test_model_contain_intersect(model, dataset, min_y=min_y)
            
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
        results_df.to_csv("intersect_min_y_experiment_results_20.csv", index=False)
        print(f"\nResults saved to min_y_experiment_results.csv")
        results_df


