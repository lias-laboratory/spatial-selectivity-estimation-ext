import os
import csv
import random
import time
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
from tqdm import tqdm
import configparser
import psycopg2
from psycopg2 import sql
from concurrent.futures import ThreadPoolExecutor
from math import sqrt

# Load configuration
def load_config(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

# Fetch random spatial objects from the database table in batches
def fetch_random_spatial_objects(cursor, table_name, batch_size):
    try:
        cursor.execute(sql.SQL("""
            SELECT ST_AsText(geometry) FROM {table_name} 
            WHERE id IN (
                SELECT id FROM {table_name} 
                ORDER BY random() LIMIT {batch_size}
            )
        """).format(
            table_name=sql.Identifier(table_name),
            batch_size=sql.Literal(batch_size)
        ))
        rows = cursor.fetchall()
        spatial_objects = [load_wkt(row[0]) for row in rows]
        return spatial_objects
    except Exception as e:
        print(f"Error executing SQL query: {e}", flush=True)
        return []

# Generate a random point within the region circle
def generate_random_point_within_circle(center_x, center_y, radius):
    while True:
        random_x = random.uniform(center_x - radius, center_x + radius)
        random_y = random.uniform(center_y - radius, center_y + radius)
        if sqrt((random_x - center_x)**2 + (random_y - center_y)**2) <= radius:
            return Point(random_x, random_y)

# Process a single spatial table
def process_table(row, output_dir, conn_params):
    table_name = row['Table Name']
    output_file = os.path.join(output_dir, f"{table_name}_dataset.csv")
    
    # Check if the dataset file already exists
    if os.path.exists(output_file):
        print(f"Skipping {table_name}, dataset file already exists.")
        return None
    
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

    proportion_count = int(total_objects * 0.2)
    size_proportions = {
        0.1: int(proportion_count * 0.8),
        0.2: int(proportion_count * 0.1),
        0.5: int(proportion_count * 0.08),
        1.0: int(proportion_count * 0.02),
    }

    start_time = time.time()
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['MBR', 'Spatial Object', 'Distance Min', 'Distance Max'])

        universe_diagonal_half = sqrt((max_x - min_x)**2 + (max_y - min_y)**2) / 2

        for size, count in size_proportions.items():
            written_count = 0
            batch_size = 100  # Fetch 100 random objects at a time
            while written_count < count:
                spatial_objects = fetch_random_spatial_objects(cursor, table_name, batch_size)
                if not spatial_objects:
                    break

                for spatial_object in spatial_objects:
                    if written_count >= count:
                        break

                    mbr_xmin, mbr_ymin, mbr_xmax, mbr_ymax = spatial_object.bounds
                    mbr = (mbr_xmin, mbr_ymin, mbr_xmax, mbr_ymax)

                    center_x, center_y = spatial_object.centroid.x, spatial_object.centroid.y
                    radius = size * universe_diagonal_half

                    point1 = generate_random_point_within_circle(center_x, center_y, radius)
                    point2 = generate_random_point_within_circle(center_x, center_y, radius)

                    distance_1 = spatial_object.distance(point1)
                    distance_2 = spatial_object.distance(point2)

                    distance_min = min(distance_1, distance_2)
                    distance_max = max(distance_1, distance_2)

                    writer.writerow([mbr, spatial_object.wkt, distance_min, distance_max])
                    written_count += 1

    conn.close()
    time_taken = time.time() - start_time
    return {
        'Table Name': table_name,
        'Total Queries Generated': num_queries,
        'Time Taken (seconds)': time_taken,
    }

# Generate datasets with parallel processing
def generate_dataset_parallel(spatial_stats_file, output_dir, summary_csv, conn_params, num_workers=12):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(spatial_stats_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        spatial_stats = list(reader)

    summary_data = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_table, row, output_dir, conn_params)
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
    output_dir = "distanceDataset"
    summary_csv = "generation_summary_distance_2.csv"

    db_params = config["database"]
    conn_params = {
        "dbname": db_params["dbname"],
        "user": db_params["user"],
        "password": db_params["password"],
        "host": db_params["host"],
        "port": db_params["port"],
    }

    generate_dataset_parallel(spatial_stats_file, output_dir, summary_csv, conn_params, num_workers=12)
