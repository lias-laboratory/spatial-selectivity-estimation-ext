import os
import sys
import argparse
import psycopg2
from psycopg2 import sql
import csv
import configparser
from tqdm import tqdm
from multiprocessing import Pool

csv.field_size_limit(sys.maxsize)

# Load configuration from config.ini
def load_config(file_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


# Process a batch of queries to count intersections
def process_batch(args):
    conn_params, batch_queries, table_name = args
    conn = psycopg2.connect(**conn_params)
    mbr_table = f"{table_name}_mbr"
    results = []

    with conn, conn.cursor() as cursor:
        for query_mbr, query_polygon in batch_queries:
            # Count intersections with the _mbr table
            mbr_query = sql.SQL("""
                SELECT COUNT(*)
                FROM {mbr_table}
                WHERE ST_Intersects(
                    ST_MakeEnvelope({mbr_coords}, 4326),
                    geometry
                );
            """).format(
                mbr_table=sql.Identifier(mbr_table),
                mbr_coords=sql.SQL(", ").join(sql.Placeholder() for _ in query_mbr.split(", "))
            )

            # Count intersections with the main table
            polygon_query = sql.SQL("""
                SELECT COUNT(*)
                FROM {table_name}
                WHERE ST_Intersects(
                    ST_GeomFromText(%s, 4326),
                    geometry
                );
            """).format(
                table_name=sql.Identifier(table_name)
            )

            cursor.execute(mbr_query, tuple(map(float, query_mbr.strip("()").split(", "))))
            count_mbr = cursor.fetchone()[0]

            cursor.execute(polygon_query, (query_polygon,))
            count_polygon = cursor.fetchone()[0]

            results.append((query_mbr, query_polygon, count_mbr, count_polygon))

    conn.close()
    return results


# Main function to execute the queries
def execute_queries(conn_params, dataset_path, table_name, output_file, batch_size=1000):
    results = []

    with open(dataset_path, mode="r", encoding="utf-8") as dataset:
        reader = csv.DictReader(dataset)
        total_queries = sum(1 for _ in open(dataset_path, "r", encoding="utf-8")) - 1
        dataset.seek(0)  # Reset file pointer after counting rows

        batch_queries = []
        with tqdm(total=total_queries, desc="Processing Dataset", unit="query") as progress_bar:
            for row in reader:
                query_mbr = row["MBR"]
                query_polygon = row["Polygon"]
                batch_queries.append((query_mbr, query_polygon))

                # Process the batch if batch size is reached
                if len(batch_queries) >= batch_size:
                    results.extend(run_in_parallel(conn_params, batch_queries, table_name))
                    batch_queries.clear()
                    progress_bar.update(batch_size)

            # Process remaining queries
            if batch_queries:
                results.extend(run_in_parallel(conn_params, batch_queries, table_name))
                progress_bar.update(len(batch_queries))

    # Write results to file
    write_results(output_file, results)


# Run batch processing in parallel
def run_in_parallel(conn_params, batch_queries, table_name):
    num_threads = 10
    chunk_size = len(batch_queries) // num_threads + 1
    chunks = [batch_queries[i:i + chunk_size] for i in range(0, len(batch_queries), chunk_size)]

    with Pool(processes=num_threads) as pool:
        args = [(conn_params, chunk, table_name) for chunk in chunks]
        results = pool.map(process_batch, args)

    # Flatten results from all processes
    return [item for sublist in results for item in sublist]


# Write results to a CSV file
def write_results(output_file, results):
    with open(output_file, mode="w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Query MBR", "Query Polygon", "Count MBR", "Count Polygon"])
        writer.writerows(results)


# Main function to handle the command-line arguments
def main(dataset_input):
    # Check if the input is a directory or a file
    if os.path.isdir(dataset_input):
        # If it's a directory, process all dataset files
        dataset_files = [f for f in os.listdir(dataset_input) if f.endswith("_dataset.csv")]
        for dataset_file in dataset_files:
            print(f"Processing file: {dataset_file}")
            dataset_path = os.path.join(dataset_input, dataset_file)
            table_name = dataset_file.replace("_dataset.csv", "")
            output_file = os.path.join("VMresults", f"{table_name}_results.csv")
            execute_queries(conn_params, dataset_path, table_name, output_file)
    elif os.path.isfile(dataset_input) and dataset_input.endswith("_dataset.csv"):
        # If it's a single file, process it
        dataset_path = dataset_input
        table_name = dataset_input.replace("_dataset.csv", "")
        output_file = os.path.join("VMresults", f"{table_name}_results.csv")
        execute_queries(conn_params, dataset_path, table_name, output_file)
    else:
        print("Invalid input. Please provide a valid dataset file or directory.")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process spatial queries.")
    parser.add_argument("dataset_input", help="Dataset file or directory containing _dataset.csv files")
    args = parser.parse_args()

    # Load database configuration
    config = load_config("config.ini")
    db_params = config["database"]

    # Database connection parameters
    conn_params = {
        "dbname": db_params["dbname"],
        "user": db_params["user"],
        "password": db_params["password"],
        "host": db_params["host"],
        "port": db_params["port"],
    }

    # Call the main function with the dataset input
    main(args.dataset_input)