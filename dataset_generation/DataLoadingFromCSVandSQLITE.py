import os
import sys
import csv
import sqlite3
import psycopg2
import configparser
from tqdm import tqdm
import time
from shapely import wkb

csv.field_size_limit(sys.maxsize)

# Load configuration
def load_config(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

# Insert data into PostgreSQL
def insert_into_db(cursor, table_name, batch):
    cursor.executemany(
        f"INSERT INTO {table_name} (geometry) VALUES (ST_GeomFromText(%s, 4326))",
        [(geom,) for geom in batch]
    )

# Process CSV files
def process_csv(file_path, table_name, conn, batch_size=10000):
    cursor = conn.cursor()
    batch = []
    total_rows = 0
    
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")  # Assuming tab-delimited
        for row in reader:
            if not row:
                continue
            geometry = row[0].strip()
            batch.append(geometry)
            
            if len(batch) >= batch_size:
                insert_into_db(cursor, table_name, batch)
                total_rows += len(batch)
                batch = []
    
    if batch:
        insert_into_db(cursor, table_name, batch)
        total_rows += len(batch)
    
    conn.commit()
    return total_rows

# Process SQLite (.db) files
def process_sqlite(file_path, table_name, conn, batch_size=10000):
    sqlite_conn = sqlite3.connect(file_path)
    sqlite_cursor = sqlite_conn.cursor()
    batch = []
    total_rows = 0
    
    sqlite_cursor.execute(f"SELECT s_object FROM dic_sp")
    for row in sqlite_cursor:
        wkb_geometry = row[0]
        geometry = wkb.loads(wkb_geometry).wkt  # Convert WKB to WKT
        batch.append(geometry)
        
        if len(batch) >= batch_size:
            insert_into_db(conn.cursor(), table_name, batch)
            total_rows += len(batch)
            batch = []
    
    if batch:
        insert_into_db(conn.cursor(), table_name, batch)
        total_rows += len(batch)
    
    conn.commit()
    sqlite_conn.close()
    return total_rows

# Process directory of spatial files
def process_directory(config):
    directory = config["settings"]["directory"]
    batch_size = int(config["settings"]["batch_size"])

    # Get list of spatial files (CSV and .db)
    spatial_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory) if f.endswith(".csv") or f.endswith(".db")
    ]

    total_files = len(spatial_files)
    start_time = time.time()

    # Connect to PostgreSQL
    db_config = config["database"]
    conn = psycopg2.connect(**db_config)

    for file_path in tqdm(spatial_files, desc="Processing files", unit="file"):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        table_name = file_name.replace("-", "_")
        
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                geometry GEOMETRY NOT NULL
            );
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_geometry_idx
            ON {table_name}
            USING GIST (geometry);
        """)
        conn.commit()

        if file_path.endswith(".csv"):
            rows_inserted = process_csv(file_path, table_name, conn, batch_size)
        elif file_path.endswith(".db"):
            rows_inserted = process_sqlite(file_path, table_name, conn, batch_size)
        
        tqdm.write(f"Processed {rows_inserted} rows from {file_name}.")

    conn.close()
    print("All files processed successfully.")

# Main function
if __name__ == "__main__":
    config = load_config("config.ini")
    process_directory(config)
