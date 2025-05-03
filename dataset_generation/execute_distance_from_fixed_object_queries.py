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
        for object_mbr, object_true_shape, distance_min, distance_max in batch_queries:
            try:
                mbr_coords = tuple(map(float, object_mbr.strip("()").split(", ")))

                if object_true_shape.startswith("POINT"):
                    # Handle point case - compare distances with the true shape
                    mbr_query = sql.SQL("""
                        SELECT COUNT(*)
                        FROM {mbr_table}
                        WHERE ST_Intersects(
                            geometry,
                            ST_Buffer(ST_GeomFromText(%s, 4326), %s)
                        )
                        AND NOT ST_Within(
                            geometry,
                            ST_Buffer(ST_GeomFromText(%s, 4326), %s)
                        );
                    """).format(
                        mbr_table=sql.Identifier(mbr_table)
                    )
                    # print("Executing query:", mbr_query.as_string(cursor))
                    # print("With parameters:", (object_true_shape, distance_max, object_true_shape, distance_min))

                    cursor.execute(mbr_query, (object_true_shape, distance_max, object_true_shape, distance_min))
                else:
                    # Handle bounding box case - compare distances with the MBR
                    mbr_query = f"""
                        SELECT COUNT(*)
                        FROM {mbr_table}
                        WHERE ST_Intersects(
                            geometry,
                            ST_Buffer(ST_MakeEnvelope({mbr_coords[0]}, {mbr_coords[1]}, {mbr_coords[2]}, {mbr_coords[3]}, 4326),
                            {distance_max})
                        )
                        AND NOT ST_Within(
                            geometry,
                            ST_Buffer(ST_MakeEnvelope({mbr_coords[0]}, {mbr_coords[1]}, {mbr_coords[2]}, {mbr_coords[3]}, 4326),
                            {distance_min})
                        );
                    """
                    # print("Executing query:", mbr_query)

                    cursor.execute(mbr_query)

                count_mbr = cursor.fetchone()[0]

                # Count intersections with the main table
                true_shape_query = sql.SQL("""
                    SELECT COUNT(*)
                    FROM {table_name}
                    WHERE ST_DWithin(
                        geometry,
                        ST_GeomFromText(%s, 4326),
                        %s
                    )
                    AND NOT ST_Intersects(
                        geometry,
                        ST_Buffer(ST_GeomFromText(%s, 4326),
                        %s)
                    );
                """).format(
                    table_name=sql.Identifier(table_name)
                )
                # print("Executing true shape query:", true_shape_query.as_string(cursor))
                # print("With parameters:", (object_true_shape, distance_max, object_true_shape, distance_min))
                
                cursor.execute(true_shape_query, (object_true_shape, distance_max, object_true_shape, distance_min))
                count_true_shape = cursor.fetchone()[0]

                results.append((object_mbr, object_true_shape, distance_min, distance_max, count_mbr, count_true_shape))
            except Exception as e:
                print(f"Error processing query with MBR: {object_mbr} and True Shape: {object_true_shape}. Error: {e}")
                # show the specific line that caused the error
                print(f"Error on line {sys.exc_info()[-1].tb_lineno}")

    conn.close()
    return results

# Main function to execute the queries
def execute_queries(conn_params, dataset_path, table_name, output_file, batch_size=10000):
    results = []

    with open(dataset_path, mode="r", encoding="utf-8") as dataset:
        reader = csv.DictReader(dataset)
        total_queries = sum(1 for _ in open(dataset_path, "r", encoding="utf-8")) - 1
        dataset.seek(0)  # Reset file pointer after counting rows

        batch_queries = []
        with tqdm(total=total_queries, desc="Processing Dataset", unit="query") as progress_bar:
            for row in reader:
                object_mbr = row["MBR"]
                object_true_shape = row["Spatial Object"]
                distance_min = float(row["Distance Min"])
                distance_max = float(row["Distance Max"])
                batch_queries.append((object_mbr, object_true_shape, distance_min, distance_max))

                # Process the batch if batch size is reached
                if len(batch_queries) >= batch_size:
                    results.extend(run_in_parallel(conn_params, batch_queries, table_name))
                    batch_queries.clear()
                    progress_bar.update(batch_size)

            # Process remaining queries
            if batch_queries:
                results.extend(run_in_parallel(conn_params, batch_queries, table_name))
                progress_bar.update(len(batch_queries))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write results to file
    write_results(output_file, results)

# Run batch processing in parallel
def run_in_parallel(conn_params, batch_queries, table_name):
    num_threads = 64
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
        writer.writerow(["Object MBR", "Object True Shape", "Distance Min", "Distance Max", "Count MBR", "Count True Shape"])
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
            output_file = os.path.join("resultsDistance", f"{table_name}_results.csv")
            execute_queries(conn_params, dataset_path, table_name, output_file)
    elif os.path.isfile(dataset_input) and dataset_input.endswith("_dataset.csv"):
        # If it's a single file, process it
        dataset_path = dataset_input
        table_name = dataset_input.replace("_dataset.csv", "")
        output_file = os.path.join("resultsDistance", f"{table_name}_results.csv")
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