import json
import numpy as np
import pandas as pd

def generate_comparison_table():
    """Generate LaTeX table from model comparison JSON data with build/train time"""
    
    # Load the JSON data
    with open("/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/model_comparison_ordered_new_cleaned.json", "r") as f:
        data = json.load(f)
    
    # Models to include in the comparison
    models = ['Histogram', 'RTree', 'KNN', 'NN', 'RF', 'XGB']
    
    # Make sure metrics order matches the column order in the LaTeX template
    metrics = ['build_time_s', 'model_size_mb', 'avg_time_ms', 'mae', 'mape']
    metric_labels = ['Build (s)', 'Size (MB)', 'Time (ms)', 'MAE', 'MAPE (%)']
    
    # Filter types to compare
    filter_types = ['intersect', 'contain', 'distance']
    filter_labels = ['Intersect', 'Contain', 'Distance']
    
    # Get all datasets
    datasets = list(data.keys())
    
    # Generate separate tables for each filter type
    all_latex = []
    
    for filter_idx, filter_type in enumerate(filter_types):
        # Start LaTeX table
        latex = []
        latex.append("\\begin{table}")
        latex.append("\\centering")
        latex.append(f"\\caption{{Model Performance Metrics for {filter_labels[filter_idx]} Filter}}")
        latex.append(f"\\label{{tab:model-comparison-{filter_type}}}")
        latex.append("\\resizebox{\\textwidth}{!}{")
        latex.append("\\begin{tabular}{l|" + "".join(["c" for _ in models]) + "|" + "".join(["c" for _ in models]) + "|" + "".join(["c" for _ in models]) + "|" + "".join(["c" for _ in models]) + "|" + "".join(["c" for _ in models]) + "}")
        latex.append("\\toprule")
        
        # Column headers for models
        latex.append("Dataset & " + " & ".join(models) + " & " + " & ".join(models) + " & " + " & ".join(models) + " & " + " & ".join(models) + " & " + " & ".join(models) + " \\\\")
        latex.append("\\midrule")
        
        # Subheaders for metrics
        latex.append("& \\multicolumn{" + str(len(models)) + "}{c|}{Build (s)} & " +
                        "\\multicolumn{" + str(len(models)) + "}{c|}{Size (MB)} & " +
                        "\\multicolumn{" + str(len(models)) + "}{c|}{Time (ms)} & " +
                        "\\multicolumn{" + str(len(models)) + "}{c|}{MAE} & " +
                        "\\multicolumn{" + str(len(models)) + "}{c}{MAPE (\\%)} \\\\")
        latex.append("\\midrule")
        
        # Store metric values for computing averages
        all_values = {metric: {model: [] for model in models} for metric in metrics}
        
        # Add rows for each dataset
        for dataset in datasets:
            # Skip datasets that don't have data for this filter type
            has_data = False
            for model in models:
                if model in data[dataset] and filter_type in data[dataset][model] and any(data[dataset][model][filter_type].values()):
                    has_data = True
                    break
            
            if not has_data:
                continue
                
            row = [dataset]
            
            # For each metric, find the best (lowest) value to make bold
            best_values = {}
            for metric in metrics:
                metric_values = {}
                for model in models:
                    if (model in data[dataset] and 
                        filter_type in data[dataset][model] and 
                        data[dataset][model][filter_type][metric] is not None):
                        metric_values[model] = data[dataset][model][filter_type][metric]
                
                if metric_values:
                    best_model = min(metric_values, key=metric_values.get)
                    best_values[(metric, best_model)] = True
            
            # Add values for each metric
            for metric in metrics:
                for model in models:
                    if (model in data[dataset] and 
                        filter_type in data[dataset][model] and 
                        data[dataset][model][filter_type][metric] is not None):
                        
                        value = data[dataset][model][filter_type][metric]
                        all_values[metric][model].append(value)
                        
                        # Format the value for display
                        if value < 0.01:
                            formatted_value = "$<$0.01"
                        elif value > 10000:
                            formatted_value = f"{value/1000:.2f}K"
                        else:
                            formatted_value = f"{value:.2f}"
                        
                        # Make the best value bold
                        if (metric, model) in best_values:
                            formatted_value = f"\\textbf{{{formatted_value}}}"
                        
                        row.append(formatted_value)
                    else:
                        row.append("-")
            
            # Add the row to the table
            latex.append(" & ".join(row) + " \\\\")
        
        # Add average row
        latex.append("\\midrule")
        avg_row = ["\\textbf{Average}"]
        
        ranks = {metric: {} for metric in metrics}
        best_avg_values = {}
        
        # Calculate averages and rankings for each metric
        for metric in metrics:
            avg_values = {}
            for model in models:
                values = all_values[metric][model]
                if values:
                    avg = np.mean(values)
                    avg_values[model] = avg
                else:
                    avg_values[model] = float('inf')
            
            # Find best average value to make bold
            if avg_values:
                best_model = min(avg_values, key=lambda m: avg_values[m] if avg_values[m] != float('inf') else float('inf'))
                if avg_values[best_model] != float('inf'):
                    best_avg_values[(metric, best_model)] = True
            
            # Calculate ranks for this metric
            sorted_models = sorted(models, key=lambda m: avg_values[m])
            for idx, model in enumerate(sorted_models):
                if avg_values[model] != float('inf'):
                    ranks[metric][model] = idx + 1
        
        # Add formatted average values
        for metric in metrics:
            for model in models:
                values = all_values[metric][model]
                if values:
                    avg = np.mean(values)
                    
                    # Format the average value
                    if avg < 0.01:
                        formatted_avg = "$<$0.01"
                    elif metric == 'avg_time_ms' and avg > 1000000:
                        formatted_avg = f"{avg/1000000:.2f}M"
                    elif avg > 10000:
                        formatted_avg = f"{avg/1000:.2f}K"
                    else:
                        formatted_avg = f"{avg:.2f}"
                    
                    # Make the best value bold
                    if (metric, model) in best_avg_values:
                        formatted_avg = f"\\textbf{{{formatted_avg}}}"
                    
                    avg_row.append(formatted_avg)
                else:
                    avg_row.append("-")
        
        # Add the average row to the table
        latex.append(" & ".join(avg_row) + " \\\\")
        
        # Add ranks row
        latex.append("\\midrule")
        rank_row = ["\\textbf{Rank}"]
        
        for metric in metrics:
            for model in models:
                if model in ranks[metric]:
                    rank_value = ranks[metric][model]
                    # Bold the best rank (rank 1)
                    if rank_value == 1:
                        rank_row.append(f"\\textbf{{{rank_value}}}")
                    else:
                        rank_row.append(f"{rank_value}")
                else:
                    rank_row.append("-")
        
        latex.append(" & ".join(rank_row) + " \\\\")
        
        # Calculate average ranks across metrics
        avg_ranks = {}
        best_avg_rank = float('inf')
        best_avg_rank_model = None
        
        for model in models:
            model_ranks = [ranks[metric][model] for metric in metrics if model in ranks[metric]]
            if model_ranks:
                avg_rank = np.mean(model_ranks)
                avg_ranks[model] = avg_rank
                if avg_rank < best_avg_rank:
                    best_avg_rank = avg_rank
                    best_avg_rank_model = model
        
        # Add average rank row
        latex.append("\\midrule")
        avg_rank_row = ["\\textbf{Avg Rank}"]
        
        for _ in range(len(metrics)):  # For each metric group
            for model in models:
                if model in avg_ranks:
                    avg_rank = avg_ranks[model]
                    # Bold the best average rank
                    if model == best_avg_rank_model:
                        avg_rank_row.append(f"\\textbf{{{avg_rank:.2f}}}")
                    else:
                        avg_rank_row.append(f"{avg_rank:.2f}")
                else:
                    avg_rank_row.append("-")
        
        latex.append(" & ".join(avg_rank_row) + " \\\\")
        
        # Close the table
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}}")
        latex.append("\\end{table}")
        
        all_latex.append("\n".join(latex))
    
    # Also generate a summary table with just the averages across all filter types
    latex = []
    latex.append("\\begin{table}")
    latex.append("\\centering")
    latex.append("\\caption{Summary of Model Performance Across All Filter Types}")
    latex.append("\\label{tab:model-comparison-summary}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l|" + "".join(["ccc|" for i in range(len(metrics)-1)]) + "ccc|c}")
    latex.append("\\toprule")
    
    # Column headers with filter types
    latex.append("& " + " & ".join([f"\\multicolumn{{3}}{{c|}}{{{metric_labels[i]}}}" for i in range(len(metrics)-1)]) + 
                 f"\\multicolumn{{3}}{{c|}}{{{metric_labels[-1]}}} & \\multirow{{2}}{{*}}{{Overall}} \\\\")
    latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16}")
    
    filter_header = "Model"
    for _ in range(len(metrics)):
        for filter_label in filter_labels:
            filter_header += f" & {filter_label}"
    filter_header += " & Avg Rank \\\\"
    latex.append(filter_header)
    latex.append("\\midrule")
    
    # Collect average values and ranks across all filter types
    all_filter_averages = {metric: {model: {} for model in models} for metric in metrics}
    all_filter_ranks = {metric: {model: [] for model in models} for metric in metrics}
    best_values_by_filter = {}  # To track the best value for each metric and filter type
    
    for filter_type in filter_types:
        for metric in metrics:
            values = {}
            for model in models:
                model_values = []
                for dataset in datasets:
                    try:
                        if (model in data[dataset] and 
                            filter_type in data[dataset][model] and 
                            metric in data[dataset][model][filter_type] and
                            data[dataset][model][filter_type][metric] is not None):
                            model_values.append(data[dataset][model][filter_type][metric])
                    except Exception as e:
                        print(f"Error processing {dataset}, {model}, {filter_type}, {metric}: {e}")
                
                if model_values:
                    avg = np.mean(model_values)
                    values[model] = avg
                    all_filter_averages[metric][model][filter_type] = avg
                else:
                    values[model] = float('inf')
                    all_filter_averages[metric][model][filter_type] = None
            
            # Find the best model for this metric and filter type
            if values:
                best_model = min(values, key=lambda m: values[m] if values[m] != float('inf') else float('inf'))
                if values[best_model] != float('inf'):
                    best_values_by_filter[(metric, filter_type, best_model)] = True
            
            # Calculate ranks
            sorted_models = sorted(models, key=lambda m: values[m])
            for rank, model in enumerate(sorted_models, 1):
                if values[model] != float('inf'):
                    all_filter_ranks[metric][model].append(rank)
    
    # Calculate overall average ranks
    overall_ranks = {}
    best_overall_rank = float('inf')
    best_overall_model = None
    
    for model in models:
        ranks = []
        for metric in metrics:
            ranks.extend(all_filter_ranks[metric][model])
        
        if ranks:
            avg_rank = np.mean(ranks)
            overall_ranks[model] = avg_rank
            if avg_rank < best_overall_rank:
                best_overall_rank = avg_rank
                best_overall_model = model
        else:
            overall_ranks[model] = float('inf')
    
    # Sort models by overall rank
    sorted_models = sorted(models, key=lambda m: overall_ranks[m])
    
    # Add rows for each model
    for model in sorted_models:
        if overall_ranks[model] == float('inf'):
            continue
            
        row = [model]
        
        for metric in metrics:
            for filter_type in filter_types:
                value = all_filter_averages[metric][model].get(filter_type)
                rank_idx = filter_types.index(filter_type)
                ranks_for_model = all_filter_ranks[metric][model]
                rank = ranks_for_model[rank_idx] if rank_idx < len(ranks_for_model) else "-"
                
                if value is not None:
                    # Format the value
                    if value < 0.01:
                        formatted_value = "$<$0.01"
                    elif metric == 'avg_time_ms' and value > 1000000:
                        formatted_value = f"{value/1000000:.2f}M"
                    elif value > 10000:
                        formatted_value = f"{value/1000:.2f}K"
                    else:
                        formatted_value = f"{value:.2f}"
                    
                    # Make best value bold and add rank as superscript
                    if (metric, filter_type, model) in best_values_by_filter:
                        formatted_value = f"\\textbf{{{formatted_value}}}$^{{{rank}}}$"
                    else:
                        formatted_value = f"{formatted_value}$^{{{rank}}}$"
                    
                    row.append(formatted_value)
                else:
                    row.append(f"-$^{{{rank}}}$")
        
        # Add overall average rank, bold for the best model
        if model == best_overall_model:
            row.append(f"\\textbf{{{overall_ranks[model]:.2f}}}")
        else:
            row.append(f"{overall_ranks[model]:.2f}")
        
        # Add the row to the table
        latex.append(" & ".join(row) + " \\\\")
    
    # Close the summary table
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table}")
    all_latex.append("\n".join(latex))
    
    return all_latex

# Save the LaTeX tables to files
def main():
    latex_tables = generate_comparison_table()
    
    # Save each table to a separate file
    for i, table in enumerate(latex_tables):
        if i < 3:
            filter_type = ['intersect', 'contain', 'distance'][i]
            output_path = f"/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/model_comparison_{filter_type}.tex"
        else:
            output_path = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/model_comparison_summary.tex"
        
        with open(output_path, 'w') as f:
            f.write(table)
        
        print(f"LaTeX table saved to {output_path}")
    
    print("\nLaTeX Table Sample (summary table, first few lines):")
    print("\n".join(latex_tables[-1].split("\n")[:10]) + "\n...")

if __name__ == "__main__":
    main()