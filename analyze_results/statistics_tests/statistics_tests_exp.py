import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

def run_statistics_tests():
    """
    Run statistical tests on the generated CSV files to compare model performance.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Directory containing the CSV files
    csv_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/csv_files"
    
    # Define metrics and filters
    metrics = ['build_time_s', 'model_size_mb', 'avg_time_ms', 'mae', 'mape']
    filters = ['intersect', 'contain', 'distance']
    models = ['Histogram', 'RTree', 'KNN', 'NN', 'RF', 'XGB']
    
    # Create output directory for results
    results_dir = "/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Function to load a specific CSV
    def load_csv(filter_type, metric):
        file_path = os.path.join(csv_dir, f"{filter_type}_{metric}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            return df
        else:
            print(f"Warning: File not found - {file_path}")
            return None
    
    # Open a file to write the summary results
    with open(os.path.join(results_dir, "statistical_summary.md"), "w") as summary_file:
        summary_file.write("# Statistical Analysis of Model Performance\n\n")
        
        # For each filter and metric
        for filter_type in filters:
            summary_file.write(f"## {filter_type.capitalize()} Filter\n\n")
            
            for metric in metrics:
                df = load_csv(filter_type, metric)
                if df is None:
                    continue
                
                # Drop metadata columns for analysis
                analysis_df = df.drop(columns=['objects', 'geometry_types'], errors='ignore')
                
                summary_file.write(f"### {metric}\n\n")
                
                # Basic statistics
                desc_stats = analysis_df.describe().transpose()
                summary_file.write("#### Descriptive Statistics\n\n")
                summary_file.write(tabulate(desc_stats, headers='keys', tablefmt='pipe') + "\n\n")
                
                # Mann-Whitney U Test (non-parametric) - comparing each pair of models
                summary_file.write("#### Mann-Whitney U Test Results\n\n")
                summary_file.write("*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*\n\n")
                
                pvalue_matrix = pd.DataFrame(index=models, columns=models)
                for m1 in models:
                    for m2 in models:
                        if m1 != m2:
                            data1 = analysis_df[m1].dropna()
                            data2 = analysis_df[m2].dropna()
                            if len(data1) > 0 and len(data2) > 0:
                                _, p_value = stats.mannwhitneyu(
                                    data1, 
                                    data2, 
                                    alternative='two-sided',
                                    nan_policy='omit'
                                )
                                pvalue_matrix.at[m1, m2] = p_value
                
                summary_file.write(tabulate(pvalue_matrix, headers='keys', tablefmt='pipe') + "\n\n")
                
                # Create boxplot visualization
                plt.figure(figsize=(12, 8))
                plt.boxplot([analysis_df[model].dropna() for model in models], labels=models)
                plt.title(f"Boxplot for {filter_type} - {metric}")
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save visualization
                vis_path = os.path.join(results_dir, f"{filter_type}_{metric}_boxplot.png")
                plt.savefig(vis_path)
                plt.close()
                
                summary_file.write(f"![Boxplot]({vis_path})\n\n")
                summary_file.write("---\n\n")
    
    print(f"Statistical analysis complete. Results saved to {results_dir}")

if __name__ == "__main__":
    run_statistics_tests()