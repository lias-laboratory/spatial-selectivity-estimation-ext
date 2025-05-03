import os
import csv
import random
import time
from shapely.geometry import Polygon
from shapely.ops import unary_union, triangulate
from tqdm import tqdm
import configparser
from concurrent.futures import ProcessPoolExecutor

# Load configuration
def load_config(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

# Generate a random polygon within an MBR
# def generate_random_polygon(num_points, bounds):
#     min_x, min_y, max_x, max_y = bounds
#     points = [
#         (random.uniform(min_x, max_x), min_y),  # bottom side
#         (random.uniform(min_x, max_x), max_y),  # top side
#         (min_x, random.uniform(min_y, max_y)),  # left side
#         (max_x, random.uniform(min_y, max_y)),  # right side
#     ]
#     for _ in range(num_points - 4):
#         points.append((random.uniform(min_x, max_x), random.uniform(min_y, max_y)))

#     polygon = Polygon(points)
#     triangles = list(triangulate(polygon))
#     polygon = unary_union(triangles)
#     return polygon

def generate_random_polygon(bounds, num_points):
    """
    Generates a random polygon within specified bounds.

    Parameters:
        bounds (tuple): A tuple (min_x, min_y, max_x, max_y) defining the bounding box.
        num_points (int): Number of points for the polygon.

    Returns:
        Polygon: A Shapely Polygon object.
    """
    min_x, min_y, max_x, max_y = bounds

    # Ensure the polygon is not concave
    while True:
        points = [(random.uniform(min_x, max_x), random.uniform(min_y, max_y)) for _ in range(num_points)]
        # Modify points to align with MBR edges
        i, j = random.sample(range(num_points), 2)
        points[i] = (min_x, points[i][1])
        points[j] = (max_x, points[j][1])
        i, j = random.sample(range(num_points), 2)
        points[i] = (points[i][0], min_y)
        points[j] = (points[j][0], max_y)

        polygon = Polygon(points)
        # Check if the polygon is valid
        if polygon.is_valid:
            break

    return polygon


# Process a single spatial table
def process_table(row, output_dir):
    table_name = row['Table Name']
    total_objects = int(row['Total Spatial Objects'])
    if total_objects == 0:
        return None

    bbox = row['Universe Limits (Bounding Box)']
    if not bbox or bbox.lower() == 'none':
        return None

    try:
        coords = bbox.replace("BOX(", "").replace(")", "").split(",")
        min_x, min_y = map(float, coords[0].strip().split())
        max_x, max_y = map(float, coords[1].strip().split())
    except Exception as e:
        return None

    num_queries = max(1, total_objects // 5)
    num_queries = min(5000000, num_queries)

    size_proportions = {
        0.1: int(num_queries * 0.8),  # 80% polygons <= 10% of universe
        0.2: int(num_queries * 0.1),  # 10% polygons <= 20% of universe
        0.5: int(num_queries * 0.08),  # 8% polygons <= 50% of universe
        1.0: int(num_queries * 0.02),  # 2% polygons <= 100% of universe
    }

    output_file = os.path.join(output_dir, f"{table_name}_dataset.csv")
    start_time = time.time()

    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['MBR', 'Polygon'])

        for size, count in size_proportions.items():
            for _ in range(count):
                mbr_width = random.uniform(0, (max_x - min_x) * size)
                mbr_height = random.uniform(0, (max_y - min_y) * size)

                mbr_xmin = random.uniform(min_x, max_x - mbr_width)
                mbr_ymin = random.uniform(min_y, max_y - mbr_height)
                mbr_xmax = mbr_xmin + mbr_width
                mbr_ymax = mbr_ymin + mbr_height

                mbr = (mbr_xmin, mbr_ymin, mbr_xmax, mbr_ymax)
                polygon = generate_random_polygon(num_points=random.randint(3, 9), bounds=mbr)

                writer.writerow([mbr, polygon.wkt])

    time_taken = time.time() - start_time
    return {
        'Table Name': table_name,
        'Total Queries Generated': num_queries,
        'Time Taken (seconds)': time_taken,
    }

# Generate datasets with parallel processing
def generate_dataset_parallel(spatial_stats_file, output_dir, summary_csv, num_workers=12):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(spatial_stats_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        spatial_stats = list(reader)

    summary_data = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_table, row, output_dir)
            for row in spatial_stats
        ]

        for future in tqdm(futures, desc="Processing databases", unit="database"):
            result = future.result()
            if result:
                summary_data.append(result)

    with open(summary_csv, mode='w', newline='', encoding='utf-8') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=['Table Name', 'Total Queries Generated', 'Time Taken (seconds)'])
        writer.writeheader()
        writer.writerows(summary_data)

# Main function
if __name__ == "__main__":
    config = load_config("config.ini")
    spatial_stats_file = config['settings']['statistics_output_file']
    output_dir = "RealRandomPolygonsDataset"
    summary_csv = "generation_summary.csv"

    generate_dataset_parallel(spatial_stats_file, output_dir, summary_csv, num_workers=12)
