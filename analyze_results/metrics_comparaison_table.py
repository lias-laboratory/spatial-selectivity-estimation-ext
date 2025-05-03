import json
import pandas as pd
from collections import defaultdict

def find_best_models():
    """Find the best model for each dataset and metric across all filters"""
    
    # Load the ordered JSON data
    with open("/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/model_comparison_ordered_last.json", "r") as f:
        data = json.load(f)
    
    # Skip the summary entry
    datasets = [key for key in data.keys() if not key.startswith("_")]
    
    # Define the metrics we want to find the best model for
    metrics = {
        'model_size_mb': 'min',  # smaller is better
        'avg_time_ms': 'min',    # smaller is better
        'mae': 'min',            # smaller is better
        'mape': 'min'            # smaller is better
    }
    
    # Filter types
    filters = ['intersect', 'contain', 'distance']
    
    # Store results
    results = []
    
    # Process each dataset
    for dataset in datasets:
        dataset_info = {'Dataset': dataset}
        
        # Get metadata
        if "metadata" in data[dataset]:
            dataset_info['Objects'] = data[dataset]["metadata"].get("total_spatial_objects", "N/A")
            dataset_info['Types'] = data[dataset]["metadata"].get("spatial_object_types", "N/A")
        else:
            dataset_info['Objects'] = "N/A"
            dataset_info['Types'] = "N/A"
        
        # For each filter type
        for filter_type in filters:
            filter_results = {}
            
            # For each metric, find the best model
            for metric, operation in metrics.items():
                best_model = None
                best_value = float('inf')  # For min operations
                
                # Check each model
                for model in ['Histogram', 'RTree', 'KNN', 'NN', 'RF', 'XGB']:
                    if (model in data[dataset] and 
                        filter_type in data[dataset][model] and 
                        metric in data[dataset][model][filter_type] and 
                        data[dataset][model][filter_type][metric] is not None):
                        
                        value = data[dataset][model][filter_type][metric]
                        
                        if operation == 'min' and value < best_value:
                            best_value = value
                            best_model = model
                
                # Format the result as "model (value)"
                if best_model:
                    if metric == 'model_size_mb':
                        if best_value < 0.01:
                            formatted_value = f"{best_model} (<0.01MB)"
                        elif best_value < 1:
                            formatted_value = f"{best_model} ({best_value:.2f}MB)"
                        else:
                            formatted_value = f"{best_model} ({best_value:.1f}MB)"
                    elif metric == 'avg_time_ms':
                        if best_value < 0.001:
                            formatted_value = f"{best_model} (<1μs)"
                        elif best_value < 0.01:
                            formatted_value = f"{best_model} ({best_value*1000:.1f}μs)"
                        elif best_value < 1:
                            formatted_value = f"{best_model} ({best_value*1000:.0f}μs)"
                        else:
                            formatted_value = f"{best_model} ({best_value:.1f}ms)"
                    elif metric == 'mae':
                        if best_value < 10:
                            formatted_value = f"{best_model} ({best_value:.2f})"
                        elif best_value < 100:
                            formatted_value = f"{best_model} ({best_value:.1f})"
                        elif best_value < 10000:
                            formatted_value = f"{best_model} ({best_value:.0f})"
                        else:
                            formatted_value = f"{best_model} ({best_value/1000:.1f}K)"
                    else:  # mape
                        if best_value < 10:
                            formatted_value = f"{best_model} ({best_value:.1f}\%)"
                        elif best_value < 100:
                            formatted_value = f"{best_model} ({best_value:.0f}\%)"
                        elif best_value < 1000:
                            formatted_value = f"{best_model} ({best_value:.0f}\%)"
                        else:
                            formatted_value = f"{best_model} ({best_value/1000:.1f}K\%)"
                else:
                    formatted_value = "N/A"
                
                filter_results[f"{filter_type}_{metric}"] = formatted_value
            
            # Add filter results to dataset info
            dataset_info.update(filter_results)
        
        results.append(dataset_info)
    
    return results

def generate_latex_table():
    """Generate a LaTeX table showing the best models for each dataset and metric"""
    results = find_best_models()
    
    # Create latex table
    latex = []
    latex.append("\\begin{table}")
    latex.append("\\centering")
    latex.append("\\caption{Best Models by Metric for Each Dataset and Filter Type}")
    latex.append("\\label{tab:best-models-by-dataset}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    
    # Start tabular environment - one column for dataset, one for objects count, one for each metric/filter combo
    latex.append("\\begin{tabular}{l|r|l|l|l|l|l|l|l|l|l|l|l|l}")
    latex.append("\\toprule")
    
    # Headers
    latex.append("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Objects} & \\multicolumn{4}{c|}{Intersect Filter} & \\multicolumn{4}{c|}{Contain Filter} & \\multicolumn{4}{c}{Distance Filter} \\\\")
    latex.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}")
    latex.append(" & & Size & Time & MAE & MAPE & Size & Time & MAE & MAPE & Size & Time & MAE & MAPE \\\\")
    latex.append("\\midrule")
    
    # Add rows for each dataset
    for result in results:
        row = [
            result['Dataset'],
            f"{result['Objects']:,}",
            result.get('intersect_model_size_mb', 'N/A'),
            result.get('intersect_avg_time_ms', 'N/A'),
            result.get('intersect_mae', 'N/A'),
            result.get('intersect_mape', 'N/A'),
            result.get('contain_model_size_mb', 'N/A'),
            result.get('contain_avg_time_ms', 'N/A'),
            result.get('contain_mae', 'N/A'),
            result.get('contain_mape', 'N/A'),
            result.get('distance_model_size_mb', 'N/A'),
            result.get('distance_avg_time_ms', 'N/A'),
            result.get('distance_mae', 'N/A'),
            result.get('distance_mape', 'N/A')
        ]
        
        latex.append(" & ".join(row) + " \\\\")
    
    # End the table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table}")
    
    # Additional table for spatial object types
    latex.append("\n\\begin{table}")
    latex.append("\\centering")
    latex.append("\\caption{Spatial Object Types by Dataset}")
    latex.append("\\label{tab:spatial-object-types}")
    latex.append("\\begin{tabular}{l|l}")
    latex.append("\\toprule")
    latex.append("Dataset & Spatial Object Types \\\\")
    latex.append("\\midrule")
    
    # Add rows for spatial object types
    for result in results:
        types = result['Types'].replace("_", "\\_")
        latex.append(f"{result['Dataset']} & {types} \\\\")
    
    # End the second table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

# Save the LaTeX table to a file
def main():
    latex_table = generate_latex_table()
    
    output_path = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/best_models_by_dataset.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_path}")

if __name__ == "__main__":
    main()