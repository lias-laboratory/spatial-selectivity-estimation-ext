import json
import pandas as pd
import os

def generate_metrics_csvs():
    """
    Generate CSV files for each metric and filter combination, excluding DT model.
    Also generates combined CSV files for each metric across all filters.
    
    Creates files named like:
      - intersect_build_time_s.csv
      - contain_model_size_mb.csv
      - distance_mae.csv
      - combined_mape.csv
      - etc.
    """
    # Load the ordered JSON data
    json_path = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/model_comparison_ordered.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Define metrics and filters
    metrics = {
        'build_time_s': 'Build (s)',
        'model_size_mb': 'Size (MB)', 
        'avg_time_ms': 'Time (ms)',
        'mae': 'MAE',
        'mape': 'MAPE'
    }
    
    filters = ['intersect', 'contain', 'distance']
    
    # Define models to include (excluding DT)
    models = ['Histogram', 'RTree', 'KNN', 'NN', 'RF', 'XGB']
    
    # Skip the summary entry
    datasets = [key for key in data.keys() if not key.startswith("_")]
    
    # Output directory
    output_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/csv_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store DataFrames for combined metrics
    combined_dfs = {metric: [] for metric in metrics}
    
    # For each filter-metric combination
    for filter_type in filters:
        for metric_key, metric_name in metrics.items():
            # Create empty dataframe with datasets as rows and models as columns
            df = pd.DataFrame(index=datasets, columns=models)
            
            # Fill in the dataframe
            for dataset in datasets:
                for model in models:
                    try:
                        if (model in data[dataset] and 
                            filter_type in data[dataset][model] and 
                            metric_key in data[dataset][model][filter_type]):
                            
                            value = data[dataset][model][filter_type][metric_key]
                            df.at[dataset, model] = value
                    except:
                        pass  # Skip if any error
            
            # Save to CSV (without the metadata columns)
            file_name = f"{filter_type}_{metric_key}.csv"
            file_path = os.path.join(output_dir, file_name)
            df.to_csv(file_path)
            print(f"Created {file_path}")
            
            # Store for combined files
            # Add filter type as a column and reset index to make dataset a column
            df_copy = df.reset_index()
            df_copy.rename(columns={'index': 'Dataset'}, inplace=True)
            df_copy['Filter'] = filter_type
            combined_dfs[metric_key].append(df_copy)
    
    # Create combined files for each metric
    for metric_key in metrics:
        # Concatenate all filters for this metric
        if combined_dfs[metric_key]:
            combined_df = pd.concat(combined_dfs[metric_key], ignore_index=True)
            
            # Save combined CSV
            file_name = f"combined_{metric_key}.csv"
            file_path = os.path.join(output_dir, file_name)
            combined_df.to_csv(file_path, index=False)
            print(f"Created {file_path}")
    
    print("\nAll CSV files generated successfully!")
    print(f"Files saved to: {output_dir}")

def generate_wide_format_combined_csvs():
    """
    Generate wide-format combined CSV files where each filter is a separate column
    for each model and metric.
    """
    # Output directory
    csv_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/csv_files"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Define metrics
    metrics = ['build_time_s', 'model_size_mb', 'avg_time_ms', 'mae', 'mape']
    filters = ['intersect', 'contain', 'distance']
    
    for metric in metrics:
        # Load data from individual filter CSVs
        dfs = {}
        for filter_type in filters:
            file_path = os.path.join(csv_dir, f"{filter_type}_{metric}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Set first column (unnamed) as index
                df.set_index(df.columns[0], inplace=True)
                dfs[filter_type] = df
        
        # Create wide format DataFrame
        if dfs:
            wide_df = pd.DataFrame()
            
            # Get common index from first DataFrame
            if list(dfs.values()):
                wide_df.index = list(dfs.values())[0].index
                
                # Add columns for each filter and model
                for filter_type, df in dfs.items():
                    for column in df.columns:
                        wide_df[f"{column}_{filter_type}"] = df[column]
            
            # Save wide format combined CSV
            file_path = os.path.join(csv_dir, f"wide_combined_{metric}.csv")
            wide_df.to_csv(file_path)
            print(f"Created wide-format combined file: {file_path}")

if __name__ == "__main__":
    generate_metrics_csvs()
    generate_wide_format_combined_csvs()