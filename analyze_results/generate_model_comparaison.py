#!/usr/bin/env python3

import os
import json
import glob
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Update base directories to match your actual file structure
BASE_DIRS = {
    'Histogram': {
        'intersect': '../large_files/traditional_methods/histogram/results/intersect/',
        'contain': '../large_files/traditional_methods/histogram/results/contain/',
        'distance': '../large_files/traditional_methods/histogram/results/distance/'
    },
    'RTree': {
        'intersect': '../large_files/traditional_methods/rtree/results/intersect/',
        'contain': '../large_files/traditional_methods/rtree/results/contain/',
        'distance': '../large_files/traditional_methods/rtree/results/distance/'
    },
    'KNN': {
        'intersect': '../large_files/LearnedModels/intersect/KNN/',
        'contain': '../large_files/LearnedModels/contain/KNN/',
        'distance': '../large_files/LearnedModels/distance/KNN/'
    },
    'NN': {
        'intersect': '../large_files/LearnedModels/intersect/NN/',
        'contain': '../large_files/LearnedModels/contain/NN/',
        'distance': '../large_files/LearnedModels/distance/NN/'
    },
    'RF': {
        'intersect': '../large_files/LearnedModels/intersect/RF/',
        'contain': '../large_files/LearnedModels/contain/RF/',
        'distance': '../large_files/LearnedModels/distance/RF/'
    },
    'XGB': {
        'intersect': '../large_files/LearnedModels/intersect/XGB/',
        'contain': '../large_files/LearnedModels/contain/XGB/',
        'distance': '../large_files/LearnedModels/distance/XGB/'
    },
    'DT': {
        'intersect': '../large_files/LearnedModels/intersect/DT/',
        'contain': '../large_files/LearnedModels/contain/DT/',
        'distance': '../large_files/LearnedModels/distance/DT/'  
    }
}

# Metadata paths for datasets
METADATA_PATH = '../spatial_statistics.csv'
HISTOGRAM_METADATA_PATH = '../large_files/traditional_methods/histogram/all_histogram_metadata.csv'

# Default build time values for different model types (in seconds)
DEFAULT_BUILD_TIMES = {
    'Histogram': {
        'tiny': 0.0001,  # For datasets < 50,000 objects
        'small': 3,      # For datasets < 500,000 objects
        'medium': 13,    # For datasets < 2,000,000 objects
        'large': 67,     # For datasets < 10,000,000 objects
        'huge': 89,      # For datasets >= 10,000,000 objects
        'enormous': 4970 # For very large datasets like yago2
    },
    'RTree': {
        'factor': 0.06   # Approx. seconds per 1000 objects
    }
}

def load_metadata():
    """Load dataset metadata from spatial statistics file and histogram metadata"""
    metadata = {}
    
    # Load dataset statistics
    try:
        if os.path.exists(METADATA_PATH):
            stats_df = pd.read_csv(METADATA_PATH)
            for _, row in stats_df.iterrows():
                dataset_name = row['Table Name']
                metadata[dataset_name] = {
                    'total_spatial_objects': int(row['Num Objects']) if 'Num Objects' in row else None,
                    'spatial_object_types': row['Object Types'] if 'Object Types' in row else None,
                    'universe_limits': row['Universe Limits (Bounding Box)'] if 'Universe Limits (Bounding Box)' in row else None
                }
    except Exception as e:
        print(f"Error loading metadata from {METADATA_PATH}: {e}")
    
    # Load histogram metadata
    try:
        if os.path.exists(HISTOGRAM_METADATA_PATH):
            hist_df = pd.read_csv(HISTOGRAM_METADATA_PATH)
            for _, row in hist_df.iterrows():
                dataset_name = row['dataset']
                if dataset_name not in metadata:
                    metadata[dataset_name] = {}
                
                # Add or update information
                metadata[dataset_name].update({
                    'total_spatial_objects': row['objects'] if 'objects' in row else metadata.get(dataset_name, {}).get('total_spatial_objects'),
                    'universe_limits': f"BOX({row['min_x']} {row['min_y']},{row['max_x']} {row['max_y']})" if all(col in row for col in ['min_x', 'min_y', 'max_x', 'max_y']) else metadata.get(dataset_name, {}).get('universe_limits')
                })
    except Exception as e:
        print(f"Error loading histogram metadata from {HISTOGRAM_METADATA_PATH}: {e}")
    
    return metadata

def parse_model_size(model_dir, model_name, filter_type, dataset=None):
    """Parse model size from model files in the given directory"""
    model_size_mb = None
    
    # For Histogram, get model size from metadata
    if model_name == 'Histogram':
        try:
            if os.path.exists(HISTOGRAM_METADATA_PATH):
                hist_df = pd.read_csv(HISTOGRAM_METADATA_PATH)
                if dataset:
                    dataset_rows = hist_df[hist_df['dataset'] == dataset]
                    if not dataset_rows.empty:
                        if 'model_size_mb' in dataset_rows.columns:
                            model_size_mb = dataset_rows['model_size_mb'].values[0]
                        else:
                            # Calculate from bytes if available
                            if 'model_size_bytes' in dataset_rows.columns:
                                model_size_mb = dataset_rows['model_size_bytes'].values[0] / (1024 * 1024)
                            # Use num_cells/total_cells as an approximation
                            elif all(col in dataset_rows.columns for col in ['num_cells', 'total_cells']):
                                model_size_mb = 2.0  # Default size for complex histograms
                                if dataset_rows['num_cells'].values[0] < 10000:
                                    model_size_mb = 0.1  # Smaller size for simpler histograms
                        return model_size_mb
        except Exception as e:
            print(f"Error loading histogram model size for {dataset}: {e}")
    
    # For RTree, look for the model size in the results files
    elif model_name == 'RTree':
        try:
            # Try to find the dataset-specific size in multi-dataset evaluation files
            eval_files = glob.glob(os.path.join(model_dir, "*_evaluation_*.csv"))
            eval_files.extend(glob.glob(os.path.join(model_dir, "*_datasets_*.csv")))
            
            for file_path in eval_files:
                df = pd.read_csv(file_path)
                if 'Dataset' in df.columns and 'Model_Size_MB' in df.columns:
                    if dataset:
                        dataset_rows = df[df['Dataset'] == dataset]
                        if not dataset_rows.empty:
                            model_size_mb = dataset_rows['Model_Size_MB'].values[0]
                            return model_size_mb
                    else:
                        # If no specific dataset, use the max size as an approximation
                        model_size_mb = df['Model_Size_MB'].max()
                        return model_size_mb
                        
        except Exception as e:
            print(f"Error parsing RTree model size: {e}")
    
    # For ML models, find model files directly
    model_file_patterns = []
    
    if model_name == 'Histogram':
        model_file_patterns = [
            os.path.join(os.path.dirname(model_dir), f"*histogram*.joblib"),
            os.path.join(os.path.dirname(model_dir), f"*histogram*.pkl")
        ]
    elif model_name == 'RTree':
        model_file_patterns = [
            os.path.join(os.path.dirname(model_dir), f"*rtree*.dat"),
            os.path.join(os.path.dirname(model_dir), f"*rtree*.idx")
        ]
    else:  # ML models
        model_file_patterns = [
            os.path.join(model_dir, f"*{dataset}*.joblib") if dataset else os.path.join(model_dir, "*.joblib"),
            os.path.join(model_dir, f"*{dataset}*.pkl") if dataset else os.path.join(model_dir, "*.pkl")
        ]
    
    # Try to find model files
    for pattern in model_file_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            # Get the size of the latest model file
            latest_file = max(model_files, key=os.path.getmtime)
            model_size_mb = os.path.getsize(latest_file) / (1024 * 1024)  # Convert bytes to MB
            return model_size_mb
    
    return model_size_mb

def fix_mape_value(value):
    """Fix MAPE values to ensure consistency (as percentage)"""
    if isinstance(value, (int, float)):
        if value < 1:  # Likely a decimal representation (0.3 = 30%)
            return value * 100
    return value

def get_default_build_time(model_name, dataset_size):
    """Get default build time based on model type and dataset size"""
    if model_name == 'Histogram':
        if dataset_size < 50000:
            return DEFAULT_BUILD_TIMES['Histogram']['tiny']
        elif dataset_size < 500000:
            return DEFAULT_BUILD_TIMES['Histogram']['small']
        elif dataset_size < 2000000:
            return DEFAULT_BUILD_TIMES['Histogram']['medium']
        elif dataset_size < 10000000:
            return DEFAULT_BUILD_TIMES['Histogram']['large']
        elif dataset_size < 20000000:
            return DEFAULT_BUILD_TIMES['Histogram']['huge']
        else:
            return DEFAULT_BUILD_TIMES['Histogram']['enormous']
    elif model_name == 'RTree':
        # Scale build time based on dataset size
        return (dataset_size / 1000) * DEFAULT_BUILD_TIMES['RTree']['factor']
    else:
        return None  # No default for ML models

def load_results_from_traditional_method_file(file_path, model_name, dataset_sizes):
    """Load results from traditional method CSV files"""
    results = {}
    
    try:
        df = pd.read_csv(file_path)
        
        if 'Dataset' in df.columns:
            for _, row in df.iterrows():
                dataset_name = row['Dataset']
                metrics = {}
                
                # Add metrics
                if 'MAE' in row:
                    metrics['mae'] = row['MAE']
                if 'MAPE' in row:
                    metrics['mape'] = fix_mape_value(row['MAPE'])
                if 'Avg_Time_ms' in row:
                    metrics['avg_time_ms'] = row['Avg_Time_ms']
                if 'Model_Size_MB' in row:
                    metrics['model_size_mb'] = row['Model_Size_MB']
                
                # Add build time if it's not in the results
                if 'build_time_s' not in metrics:
                    dataset_size = dataset_sizes.get(dataset_name, 0)
                    default_build_time = get_default_build_time(model_name, dataset_size)
                    if default_build_time is not None:
                        metrics['build_time_s'] = default_build_time
                    
                if metrics:  # Only add if we have metrics
                    results[dataset_name] = metrics
    except Exception as e:
        print(f"Error loading traditional method results from {file_path}: {e}")
    
    return results

def load_results_from_ml_model_file(file_path, model_name):
    """Load results from ML model CSV files"""
    results = {}
    
    try:
        df = pd.read_csv(file_path)
        
        # Get dataset name from file
        dataset_name = os.path.basename(file_path).replace('_results.csv', '')
        
        # For files with multiple rows (different sample sizes), get the largest sample size
        if len(df) > 0:
            if 'Sample_Size' in df.columns:
                row = df.loc[df['Sample_Size'].idxmax()]
            else:
                # If no sample size column, use the last row
                row = df.iloc[-1]
                
            metrics = {}
            
            # Add metrics - looking for columns with various possible names
            metric_mappings = {
                'mae': ['MAE'],
                'mape': ['MAPE'],
                'avg_time_ms': ['Pred_Time_Microseconds', 'Inference_Time_ms', 'Avg_Time_ms'],
                'build_time_s': ['Training_Time', 'build_time_s', 'Cumulative_Training_Time', 
                                 'Training_Time_This_Scale', 'Total_Training_Time'],
                'model_size_mb': ['Model_Size_MB']
            }
            
            # Process all metrics using mappings
            for metric_name, possible_columns in metric_mappings.items():
                for col in possible_columns:
                    if col in row:
                        if metric_name == 'avg_time_ms' and col == 'Pred_Time_Microseconds':
                            # Convert microseconds to milliseconds
                            metrics[metric_name] = row[col] / 1000
                        elif metric_name == 'mape':
                            # Fix MAPE values
                            metrics[metric_name] = fix_mape_value(row[col])
                        else:
                            metrics[metric_name] = row[col]
                        break
            
            # Special handling for model size in KB
            if 'Model_Size_KB' in row and 'model_size_mb' not in metrics:
                metrics['model_size_mb'] = row['Model_Size_KB'] / 1024
            
            # Add max/min/mean counts if available (for better context)
            for col_prefix in ['Max_', 'Min_', 'Mean_', 'Median_']:
                col_suffix = 'Count'
                col_name = f"{col_prefix}{col_suffix}"
                if col_name in row:
                    metrics[col_name.lower()] = row[col_name]
                
            if metrics:  # Only add if we have metrics
                results[dataset_name] = metrics
    except Exception as e:
        print(f"Error loading ML model results from {file_path}: {e}")
    
    return results

def process_directory(directory, model_name, filter_type, dataset_sizes):
    """Process all result files in a directory"""
    results = {}
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return results
    
    # For traditional methods, first check for evaluation files
    if model_name in ['Histogram', 'RTree']:
        # Check for evaluation files that have multiple datasets
        for pattern in ["*_evaluation_sample*.csv", "*_datasets_*.csv", "all_datasets_*.csv"]:
            eval_files = glob.glob(os.path.join(directory, pattern))
            for file_path in eval_files:
                file_results = load_results_from_traditional_method_file(file_path, model_name, dataset_sizes)
                for dataset, metrics in file_results.items():
                    if dataset not in results:
                        results[dataset] = metrics
                    else:
                        # Update existing metrics
                        results[dataset].update(metrics)
    
    # For ML models, check for all_results.csv
    else:
        all_results_file = os.path.join(directory, "all_results.csv")
        individual_results = []
        
        if os.path.exists(all_results_file):
            # Load the all_results file which contains data for multiple datasets
            df = pd.read_csv(all_results_file)
            if 'Dataset' in df.columns:
                # Group by dataset and get the most recent/largest sample for each
                for dataset, group in df.groupby('Dataset'):
                    if 'Sample_Size' in group.columns:
                        row = group.loc[group['Sample_Size'].idxmax()]
                    else:
                        # If no sample size column, use the last row
                        row = group.iloc[-1]
                    
                    metrics = {}
                    # Extract metrics (similar to the ML model file function)
                    if 'MAE' in row:
                        metrics['mae'] = row['MAE']
                    if 'MAPE' in row:
                        metrics['mape'] = fix_mape_value(row['MAPE'])
                    if 'Pred_Time_Microseconds' in row:
                        metrics['avg_time_ms'] = row['Pred_Time_Microseconds'] / 1000
                    if any(col in row for col in ['Training_Time', 'build_time_s', 'Cumulative_Training_Time', 'Training_Time_This_Scale']):
                        time_col = [col for col in ['Training_Time', 'build_time_s', 'Cumulative_Training_Time', 'Training_Time_This_Scale'] if col in row][0]
                        metrics['build_time_s'] = row[time_col]
                    if 'Model_Size_KB' in row:
                        metrics['model_size_mb'] = row['Model_Size_KB'] / 1024
                    elif 'Model_Size_MB' in row:
                        metrics['model_size_mb'] = row['Model_Size_MB']
                    
                    if metrics:
                        results[dataset] = metrics
            else:
                # Process as regular ML model file if no Dataset column
                file_results = load_results_from_ml_model_file(all_results_file, model_name)
                results.update(file_results)
                
        else:
            # Check for individual dataset result files
            individual_results = glob.glob(os.path.join(directory, "*_results.csv"))
            for file_path in individual_results:
                file_results = load_results_from_ml_model_file(file_path, model_name)
                results.update(file_results)
    
    # Add model size for each dataset if not already present
    for dataset in results:
        if 'model_size_mb' not in results[dataset]:
            model_size = parse_model_size(directory, model_name, filter_type, dataset)
            if model_size:
                results[dataset]['model_size_mb'] = model_size
    
    return results

def generate_model_comparison():
    """Generate comparison data across all models, filters, and datasets"""
    # Load metadata
    metadata = load_metadata()
    
    # Extract dataset sizes
    dataset_sizes = {dataset: data.get('total_spatial_objects', 0) 
                     for dataset, data in metadata.items()}
    
    # Dictionary to hold all comparison data
    comparison_data = {}
    
    # Dictionary to hold aggregated data across datasets
    aggregated_data = defaultdict(lambda: defaultdict(dict))
    
    # Process each model type
    for model_name, filter_dirs in BASE_DIRS.items():
        print(f"Processing {model_name} results...")
        
        # Process each filter type
        for filter_type, results_dir in filter_dirs.items():
            print(f"  Processing {filter_type} filter...")
            
            # Get results for this model/filter combination
            results = process_directory(results_dir, model_name, filter_type, dataset_sizes)
            
            # Update comparison data
            for dataset, metrics in results.items():
                # Initialize dataset entry if not exists
                if dataset not in comparison_data:
                    comparison_data[dataset] = {}
                
                # Initialize model entry if not exists
                if model_name not in comparison_data[dataset]:
                    comparison_data[dataset][model_name] = {}
                
                # Add filter metrics
                comparison_data[dataset][model_name][filter_type] = metrics
                
                # Update aggregated data
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in aggregated_data[model_name][filter_type]:
                            aggregated_data[model_name][filter_type][metric_name] = []
                        aggregated_data[model_name][filter_type][metric_name].append(metric_value)
    
    # Add dataset metadata
    for dataset in comparison_data:
        if dataset != 'all' and dataset in metadata:
            comparison_data[dataset]['metadata'] = metadata[dataset]
    
    # Add aggregated "all" dataset with averages across all datasets
    comparison_data['all'] = {}
    for model_name, filter_data in aggregated_data.items():
        comparison_data['all'][model_name] = {}
        for filter_type, metrics in filter_data.items():
            avg_metrics = {}
            for metric_name, values in metrics.items():
                if values:
                    avg_metrics[metric_name] = np.mean(values)
            if avg_metrics:
                comparison_data['all'][model_name][filter_type] = avg_metrics
    
    # Add summary
    dataset_counts = {}
    for dataset in comparison_data:
        if dataset != 'all' and dataset in metadata and metadata[dataset].get('total_spatial_objects'):
            dataset_counts[dataset] = metadata[dataset]['total_spatial_objects']
        else:
            dataset_counts[dataset] = 0
    
    comparison_data['_summary'] = {
        'description': "Datasets ordered by total spatial objects (ascending)",
        'dataset_counts': dict(sorted(dataset_counts.items(), key=lambda x: x[1])),
        'total_datasets': len(comparison_data) - 2,  # Subtract 'all' and '_summary'
        'models_evaluated': list(BASE_DIRS.keys()),
        'filters_evaluated': list(BASE_DIRS[list(BASE_DIRS.keys())[0]].keys()),
        'metrics': ['model_size_mb', 'avg_time_ms', 'mae', 'mape', 'build_time_s']
    }
    
    return comparison_data

def save_comparison(comparison_data, output_file='model_comparison_new.json'):
    """Save comparison data to a JSON file"""
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"Comparison data saved to {output_path}")

def create_ordered_version(input_file='model_comparison_new.json', output_file='model_comparison_ordered_new.json'):
    """Create an ordered version of the comparison data where datasets are sorted by size"""
    input_path = os.path.join(os.path.dirname(__file__), input_file)
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create a new ordered dictionary
    ordered_data = {}
    
    # Add summary first
    if '_summary' in data:
        ordered_data['_summary'] = data['_summary']
    
    # Add datasets in order of size (based on summary)
    if '_summary' in data and 'dataset_counts' in data['_summary']:
        for dataset in data['_summary']['dataset_counts'].keys():
            if dataset in data:
                ordered_data[dataset] = data[dataset]
    
    # Save ordered data
    with open(output_path, 'w') as f:
        json.dump(ordered_data, f, indent=2)
    
    print(f"Ordered comparison data saved to {output_path}")

def main():
    """Main function to generate model comparison"""
    print("Generating model comparison data...")
    
    # Generate comparison data
    comparison_data = generate_model_comparison()
    
    # Save comparison data
    save_comparison(comparison_data)
    
    # Create ordered version
    create_ordered_version()
    
    print("Model comparison generation complete.")

if __name__ == "__main__":
    main()