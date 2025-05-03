import os
import papermill as pm

# Define the list of notebooks to run
notebooks_to_run = [
    "~/nadir/Spatial-Selectivity-Ext/traditional_methods/histogram/histogram-intersect-estimator.ipynb",
    "~/nadir/Spatial-Selectivity-Ext/traditional_methods/histogram/histogram-contain-estimator.ipynb",
    "~/nadir/Spatial-Selectivity-Ext/traditional_methods/histogram/histogram-distance-estimator.ipynb",
    # "~/nadir/Spatial-Selectivity-Ext/traditional_methods/rtree/RTree-intersect-estimator.ipynb",
    # "~/nadir/Spatial-Selectivity-Ext/traditional_methods/rtree/RTree-contain-estimator.ipynb",
    # "~/nadir/Spatial-Selectivity-Ext/traditional_methods/rtree/RTree-distance-estimator.ipynb",
    # Add more notebooks as needed
]

# Specify the kernel to use
kernel_name = "python3"

# Process each notebook in the list
for notebook_path in notebooks_to_run:
    # Expand the tilde to get the full path
    full_path = os.path.expanduser(notebook_path)
    
    # Check if the file exists
    if not os.path.exists(full_path):
        print(f"Error: Notebook '{full_path}' does not exist!")
        continue
    
    # Create output path in the same directory
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    output_path = os.path.join(directory, f"output_{filename}")
    
    print(f"Running {full_path}...")
    
    try:
        # Execute with kernel name specified
        pm.execute_notebook(
            full_path, 
            output_path,
            kernel_name=kernel_name
        )
        print(f"Successfully executed: {filename}")
    except Exception as e:
        print(f"Error executing {filename}: {e}")