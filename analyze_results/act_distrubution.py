import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set the Matplotlib style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# Create output directory for figures
os.makedirs('figs', exist_ok=True)

def analyze_act_distribution(file_path, filter_type):
    """Analyze the distribution of act values in a CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Try to find the appropriate column for MBR counts
        count_column = None
        for col in df.columns:
            if 'count' in col.lower() and 'mbr' in col.lower():
                count_column = col
                break
        
        if count_column is None and 'Count MBR' in df.columns:
            count_column = 'Count MBR'
            
        if count_column is None and len(df.columns) >= 3:
            # Assume third column is the count column if we can't find by name
            count_column = df.columns[2]
        
        if count_column is None:
            print(f"Could not determine count column in {file_path}")
            return None
            
        dataset_name = os.path.basename(file_path).replace('_results.csv', '')
        counts = df[count_column].values
        
        # Create bins for the counts
        bins = [0, 10, 100, 1000, 10000, 100000, 1000000, float('inf')]
        bin_labels = ['0-10', '10-100', '100-1K', '1K-10K', '10K-100K', '100K-1M', '>1M']
        
        # Get histogram data
        hist, _ = np.histogram(counts, bins=bins)
        
        return {
            'dataset': dataset_name,
            'filter_type': filter_type,
            'bins': bin_labels,
            'counts': hist,
            'total_mbrs': len(counts),
            'avg_act': np.mean(counts) if len(counts) > 0 else 0,
            'max_act': np.max(counts) if len(counts) > 0 else 0
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_distribution_data(base_dirs):
    """Collect distribution data from all CSV files in the given directories"""
    all_data = []
    
    for filter_type, directory in base_dirs.items():
        csv_files = glob.glob(os.path.join(directory, '*.csv'))
        print(f"Found {len(csv_files)} files in {directory}")
        
        for file_path in csv_files:
            dist_data = analyze_act_distribution(file_path, filter_type)
            if dist_data:
                all_data.append(dist_data)
    
    return all_data

def plot_act_distribution(all_data):
    """Create a stacked bar chart showing the distribution of act values by bins"""
    # Organize data by filter type
    data_by_filter = {}
    for item in all_data:
        filter_type = item['filter_type']
        if filter_type not in data_by_filter:
            data_by_filter[filter_type] = []
        data_by_filter[filter_type].append(item)
    
    # Get all unique datasets
    all_datasets = set()
    for filter_data in data_by_filter.values():
        for item in filter_data:
            all_datasets.add(item['dataset'])
    
    # Create a consistent colormap - assign a fixed color to each dataset
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_datasets)))
    dataset_colors = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Set up the plot with increased width for legend space
    fig, axes = plt.subplots(len(data_by_filter), 1, figsize=(16, 4*len(data_by_filter)), sharex=True)
    if len(data_by_filter) == 1:
        axes = [axes]
    
    for i, (filter_type, filter_data) in enumerate(data_by_filter.items()):
        ax = axes[i]
        
        # Create a DataFrame for easier plotting
        df_list = []
        for item in filter_data:
            bin_percentages = (item['counts'] / item['total_mbrs']) * 100 if item['total_mbrs'] > 0 else np.zeros_like(item['counts'])
            df_list.append(pd.DataFrame({
                'Dataset': item['dataset'],
                'Bin': item['bins'],
                'Percentage': bin_percentages
            }))
        
        if not df_list:
            continue
            
        df = pd.concat(df_list)
        df_pivot = df.pivot(index='Bin', columns='Dataset', values='Percentage')
        
        # Sort datasets by their total percentage
        sorted_datasets = df_pivot.sum().sort_values(ascending=False).index.tolist()
        df_pivot = df_pivot[sorted_datasets]
        
        # Plot stacked bar chart
        df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', figsize=(12, 6))
        
        ax.set_title(f"{filter_type} Filter - Distribution of Objects per MBR", fontsize=16)
        ax.set_ylabel('Percentage of MBRs (%)', fontsize=14)
        if i == len(data_by_filter) - 1:  # Only add x-label to bottom subplot
            ax.set_xlabel('Number of Objects in MBR (Log Scale)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Place legend outside the main plot
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figs/act_distribution_comparison.pdf', bbox_inches='tight')
    print("Saved distribution comparison to figs/act_distribution_comparison.pdf")
    
    # Also create individual filter plots
    for filter_type, filter_data in data_by_filter.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a DataFrame for easier plotting
        df_list = []
        for item in filter_data:
            df_list.append(pd.DataFrame({
                'Dataset': item['dataset'],
                'Bin': item['bins'],
                'Count': item['counts']
            }))
        
        if not df_list:
            continue
            
        df = pd.concat(df_list)
        df_pivot = df.pivot(index='Bin', columns='Dataset', values='Count')
        
        # Sort datasets consistently
        sorted_datasets = df_pivot.sum().sort_values(ascending=False).index.tolist()
        df_pivot = df_pivot[sorted_datasets]
        
        # Plot stacked bar chart
        df_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        ax.set_title(f"{filter_type} Filter - Absolute Distribution of Objects per MBR", fontsize=16)
        ax.set_ylabel('Number of MBRs', fontsize=14)
        ax.set_xlabel('Number of Objects in MBR (Log Scale)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Apply log scale to y-axis
        ax.set_yscale('log')
        
        # Legend outside the plot
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'figs/act_distribution_{filter_type.lower()}.pdf', bbox_inches='tight')
        print(f"Saved {filter_type} distribution to figs/act_distribution_{filter_type.lower()}.pdf")
    
    # Create a summary plot with statistics
    create_summary_stats(all_data)

def create_summary_stats(all_data):
    """Create a summary plot with statistics"""
    # Create a DataFrame with summary statistics
    stats_df = pd.DataFrame([{
        'Dataset': item['dataset'],
        'Filter': item['filter_type'],
        'Total MBRs': item['total_mbrs'],
        'Average Act': item['avg_act'],
        'Max Act': item['max_act'],
        '% Small (0-100)': (item['counts'][0] + item['counts'][1]) / item['total_mbrs'] * 100 if item['total_mbrs'] > 0 else 0,
        '% Medium (100-10K)': (item['counts'][2] + item['counts'][3]) / item['total_mbrs'] * 100 if item['total_mbrs'] > 0 else 0,
        '% Large (>10K)': sum(item['counts'][4:]) / item['total_mbrs'] * 100 if item['total_mbrs'] > 0 else 0
    } for item in all_data])
    
    # Sort datasets by their average act value to ensure consistent ordering
    dataset_order = stats_df.groupby('Dataset')['Average Act'].mean().sort_values(ascending=False).index.tolist()
    
    # Plot summaries
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average act by dataset and filter
    pivot1 = stats_df.pivot(index='Dataset', columns='Filter', values='Average Act')
    pivot1 = pivot1.reindex(dataset_order)  # Reorder rows consistently
    sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Average Objects per MBR', fontsize=14)
    
    # Plot 2: Maximum act by dataset and filter
    pivot2 = stats_df.pivot(index='Dataset', columns='Filter', values='Max Act')
    pivot2 = pivot2.reindex(dataset_order)  # Reorder rows consistently
    sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='viridis', ax=axes[0, 1])
    axes[0, 1].set_title('Maximum Objects per MBR', fontsize=14)
    
    # Plot 3: Percentage of small MBRs (0-100 objects)
    pivot3 = stats_df.pivot(index='Dataset', columns='Filter', values='% Small (0-100)')
    pivot3 = pivot3.reindex(dataset_order)  # Reorder rows consistently
    sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 0])
    axes[1, 0].set_title('% of MBRs with 0-100 Objects', fontsize=14)
    
    # Plot 4: Percentage of large MBRs (>10K objects)
    pivot4 = stats_df.pivot(index='Dataset', columns='Filter', values='% Large (>10K)')
    pivot4 = pivot4.reindex(dataset_order)  # Reorder rows consistently
    sns.heatmap(pivot4, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('% of MBRs with >10K Objects', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figs/act_distribution_summary.pdf', bbox_inches='tight')
    print("Saved summary statistics to figs/act_distribution_summary.pdf")
    
    # Also save the summary stats to a CSV
    stats_df.to_csv('figs/act_distribution_summary.csv', index=False)
    print("Saved summary statistics to figs/act_distribution_summary.csv")

def main():
    # Define base directories for different filter types
    base_dirs = {
        'Intersect': '/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsIntersects',
        'Contain': '/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsContains',
        'Distance': '/home/adminlias/nadir/Spatial-Selectivity-Ext/large_files/resultsDistance'
    }
    
    # Collect distribution data
    all_data = collect_distribution_data(base_dirs)
    
    # Plot the distribution
    if all_data:
        plot_act_distribution(all_data)
    else:
        print("No data collected. Check file paths and formats.")

if __name__ == "__main__":
    main()