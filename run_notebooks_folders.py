import os
import papermill as pm

folders = [
            os.path.expanduser("~/nadir/Spatial-Selectivity-Ext/distance_filter")
            # os.path.expanduser("~/nadir/Spatial-Selectivity-Ext/contain_filter"),
        ]  # Expand '~' to the full path

for folder in folders:
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist!")
        continue  # Skip if folder is missing

    for file in sorted(os.listdir(folder)):  # Ensure sorted execution
        if file.endswith(".ipynb"):
            notebook_path = os.path.join(folder, file)
            output_path = os.path.join(folder, f"output_{file}")
            print(f"Running {notebook_path}...")
            
            try:
                # Specify the kernel that has xgboost installed
                pm.execute_notebook(
                    notebook_path, 
                    output_path,
                    kernel_name='python3'  # or the name of your specific kernel
                )
                print(f"Successfully executed: {file}")
            except Exception as e:
                print(f"Error executing {file}: {e}")