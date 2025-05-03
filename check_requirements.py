import importlib.util

required_packages = [
    "numpy", "pandas", "matplotlib", "sklearn", "psutil", "os",
    "re", "multiprocessing", "joblib", "gc", "tqdm", "papermill"
]

missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]

if missing_packages:
    print(f"Missing packages: {', '.join(missing_packages)}")
    print("You can install them using:")
    print(f"pip install {' '.join(missing_packages)}")
else:
    print("All required packages are installed.")
