import json
import os
from google.cloud import bigquery
from google.api_core import exceptions

# It's good practice to have configuration separate.
# We'll assume a config.py file exists with PROJECT_ID.
try:
    from config import PROJECT_ID
except ImportError:
    print("Error: config.py not found. Please create it with your PROJECT_ID.")
    print("Example config.py:\nPROJECT_ID = 'your-gcp-project-id'")
    exit(1)


def create_bq_graph_from_json(
    json_file_path: str,
    project_id: str,
    pass_number: int,
    dataset_name: str = "bq_graph_db",
    location: str = "us-central1",
    chunk_size: int = 500
):
    """
    Creates and populates a BigQuery graph database directly from a JSON file.

    This function will:
    1. Create a BigQuery dataset if it doesn't exist.
    2. Create 'nodes' and 'relationships' tables for a specific pass.
    3. Load data from the JSON file into these tables in chunks.
    4. Create a BigQuery PROPERTY GRAPH for the pass combining these tables.

    Args:
        json_file_path: Path to the input JSON file.
        project_id: Your Google Cloud project ID.
        pass_number: The pass number (e.g., 1, 2, 3, 4) to generate names.
        dataset_name: The name for the BigQuery dataset.
        location: The geographic location for the BigQuery dataset and tables.
        chunk_size: The number of rows per INSERT statement for loading data.
    """
    if not project_id or project_id == "your-gcp-project-id":
        print("Error: Please set your GCP PROJECT_ID in config.py")
        return

    # 1. Initialize BigQuery Client
    try:
        client = bigquery.Client(project=project_id)
        print(f"Successfully connected to BigQuery project: {project_id}")
    except Exception as e:
        print(f"Error connecting to BigQuery: {e}")
        return

    # 2. Load JSON data
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        nodes = graph_data.get('nodes', [])
        relationships = graph_data.get('relationships', [])
        print(f"Loaded {len(nodes)} nodes and {len(relationships)} relationships from '{json_file_path}'.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return

    # 3. Create Dataset
    dataset_id = f"{project_id}.{dataset_name}"
    try:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = location
        client.create_dataset(dataset, exists_ok=True)
        print(f"Dataset '{dataset_id}' created or already exists.")
    except exceptions.GoogleAPICallError as e:
        print(f"Error creating dataset '{dataset_id}': {e}")
        return

    # Helper to run queries
    def run_query(sql: str, description: str):
        try:
            print(f"Executing: {description}...")
            job = client.query(sql)
            job.result()  # Wait for the job to complete
            print(f"SUCCESS: {description} completed.")
        except exceptions.GoogleAPICallError as e:
            print(f"ERROR executing '{description}': {e}")
            # Stop the process if a critical step fails
            raise

    try:
        # 4. Create Node Table
        nodes_table_name = f"{dataset_id}.nodes_pass{pass_number}"
        create_nodes_sql = f"""
        CREATE OR REPLACE TABLE `{nodes_table_name}` (
          entity_name STRING NOT NULL,
          entity_type STRING,
          source_document_count INT64,
          support_ticket_count INT64,
          buganizer_count INT64,
          reddit_count INT64,
          video_count INT64,
          source_document_url ARRAY<STRING>,
          support_tickets ARRAY<STRING>,
          buganizer_ids ARRAY<STRING>,
          reddit_urls ARRAY<STRING>,
          video_urls ARRAY<STRING>,
          PRIMARY KEY (entity_name) NOT ENFORCED
        );
        """
        run_query(create_nodes_sql, f"Create nodes table '{nodes_table_name}'")

        # 5. Create Relationship Table
        relationships_table_name = f"{dataset_id}.relationships_pass{pass_number}"
        create_rels_sql = f"""
        CREATE OR REPLACE TABLE `{relationships_table_name}` (
          source_entity_name STRING NOT NULL,
          target_entity_name STRING NOT NULL,
          relationship_type STRING NOT NULL,
          cooccurrence_frequency INT64,
          extraction_frequency INT64,
          relevance_score FLOAT64,
          PRIMARY KEY (source_entity_name, target_entity_name, relationship_type) NOT ENFORCED
        );
        """
        run_query(create_rels_sql, f"Create relationships table '{relationships_table_name}'")

        # 6. Upsert Data into Nodes Table
        print("\n--- Upserting data into nodes table ---")
        if not nodes:
            print("No nodes to load.")
        else:
            temp_nodes_table_ref = client.dataset(dataset_name).table(f"temp_nodes_for_merge_{pass_number}")
            node_rows = []
            for node in nodes:
                properties = node.get('properties', {})
                s_url = properties.get('source_document_url')
                urls = []
                if s_url:
                    urls = s_url if isinstance(s_url, list) else [str(s_url)]

                s_tickets = properties.get('support_tickets')
                tickets = []
                if s_tickets:
                    tickets = s_tickets if isinstance(s_tickets, list) else [str(s_tickets)]

                b_ids = properties.get('buganizer_ids')
                bug_ids = []
                if b_ids:
                    bug_ids = b_ids if isinstance(b_ids, list) else [str(b_ids)]
                
                r_urls = properties.get('reddit_urls')
                reddit_urls = []
                if r_urls:
                    reddit_urls = r_urls if isinstance(r_urls, list) else [str(r_urls)]

                v_urls = properties.get('video_urls')
                video_urls = []
                if v_urls:
                    video_urls = v_urls if isinstance(v_urls, list) else [str(v_urls)]

                node_rows.append({
                    "entity_name": node.get("entity_name"),
                    "entity_type": node.get("entity_type"),
                    "source_document_count": properties.get("source_document_count"),
                    "support_ticket_count": properties.get("support_ticket_count"),
                    "buganizer_count": properties.get("buganizer_count"),
                    "reddit_count": properties.get("reddit_count"),
                    "video_count": properties.get("video_count"),
                    "source_document_url": urls,
                    "support_tickets": tickets,
                    "buganizer_ids": bug_ids,
                    "reddit_urls": reddit_urls,
                    "video_urls": video_urls,
                })

            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("entity_name", "STRING"),
                    bigquery.SchemaField("entity_type", "STRING"),
                    bigquery.SchemaField("source_document_count", "INT64"),
                    bigquery.SchemaField("support_ticket_count", "INT64"),
                    bigquery.SchemaField("buganizer_count", "INT64"),
                    bigquery.SchemaField("reddit_count", "INT64"),
                    bigquery.SchemaField("video_count", "INT64"),
                    bigquery.SchemaField("source_document_url", "STRING", mode="REPEATED"),
                    bigquery.SchemaField("support_tickets", "STRING", mode="REPEATED"),
                    bigquery.SchemaField("buganizer_ids", "STRING", mode="REPEATED"),
                    bigquery.SchemaField("reddit_urls", "STRING", mode="REPEATED"),
                    bigquery.SchemaField("video_urls", "STRING", mode="REPEATED"),
                ],
                write_disposition="WRITE_TRUNCATE",
            )
            print(f"Loading {len(node_rows)} nodes into temp table {temp_nodes_table_ref.table_id}...")
            load_job = client.load_table_from_json(node_rows, temp_nodes_table_ref, job_config=job_config)
            load_job.result()  # Wait for the job to complete
            print("Load to temporary table complete.")

            merge_sql = f"""
            MERGE `{nodes_table_name}` T
            USING `{temp_nodes_table_ref}` S
            ON T.entity_name = S.entity_name
            WHEN MATCHED THEN
              UPDATE SET
                T.entity_type = S.entity_type,
                T.source_document_count = S.source_document_count,
                T.support_ticket_count = S.support_ticket_count,
                T.buganizer_count = S.buganizer_count,
                T.reddit_count = S.reddit_count,
                T.video_count = S.video_count,
                T.source_document_url = S.source_document_url,
                T.support_tickets = S.support_tickets,
                T.buganizer_ids = S.buganizer_ids,
                T.reddit_urls = S.reddit_urls,
                T.video_urls = S.video_urls
            WHEN NOT MATCHED THEN
              INSERT (entity_name, entity_type, source_document_count, support_ticket_count, buganizer_count, reddit_count, video_count, source_document_url, support_tickets, buganizer_ids, reddit_urls, video_urls)
              VALUES (entity_name, entity_type, source_document_count, support_ticket_count, buganizer_count, reddit_count, video_count, source_document_url, support_tickets, buganizer_ids, reddit_urls, video_urls);
            """
            run_query(merge_sql, f"Merging {len(node_rows)} nodes from temporary table")
            client.delete_table(temp_nodes_table_ref, not_found_ok=True)
            print(f"Deleted temporary nodes table {temp_nodes_table_ref.table_id}.")

        # 7. Upsert Data into Relationships Table
        print("\n--- Upserting data into relationships table ---")
        if not relationships:
            print("No relationships to load.")
        else:
            temp_rels_table_ref = client.dataset(dataset_name).table(f"temp_rels_for_merge_{pass_number}")
            rel_rows = [
                {
                    "source_entity_name": rel.get("source_entity_name"),
                    "target_entity_name": rel.get("target_entity_name"),
                    "relationship_type": rel.get("relationship_type"),
                    "cooccurrence_frequency": rel.get("properties", {}).get("frequency"),
                    "extraction_frequency": rel.get("frequency"),
                    "relevance_score": rel.get("properties", {}).get("frequency"),
                }
                for rel in relationships
            ]

            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("source_entity_name", "STRING"),
                    bigquery.SchemaField("target_entity_name", "STRING"),
                    bigquery.SchemaField("relationship_type", "STRING"),
                    bigquery.SchemaField("cooccurrence_frequency", "INT64"),
                    bigquery.SchemaField("extraction_frequency", "INT64"),
                    bigquery.SchemaField("relevance_score", "FLOAT64"),
                ],
                write_disposition="WRITE_TRUNCATE",
            )
            print(f"Loading {len(rel_rows)} relationships into temp table {temp_rels_table_ref.table_id}...")
            load_job = client.load_table_from_json(rel_rows, temp_rels_table_ref, job_config=job_config)
            load_job.result()
            print("Load to temporary table complete.")

            merge_sql = f"""
            MERGE `{relationships_table_name}` T
            USING `{temp_rels_table_ref}` S
            ON T.source_entity_name = S.source_entity_name
               AND T.target_entity_name = S.target_entity_name
               AND T.relationship_type = S.relationship_type
            WHEN MATCHED THEN
              UPDATE SET
                T.cooccurrence_frequency = S.cooccurrence_frequency,
                T.extraction_frequency = S.extraction_frequency,
                T.relevance_score = S.relevance_score
            WHEN NOT MATCHED THEN
              INSERT (source_entity_name, target_entity_name, relationship_type, cooccurrence_frequency, extraction_frequency, relevance_score)
              VALUES (source_entity_name, target_entity_name, relationship_type, cooccurrence_frequency, extraction_frequency, relevance_score);
            """
            run_query(merge_sql, f"Merging {len(rel_rows)} relationships from temporary table")
            client.delete_table(temp_rels_table_ref, not_found_ok=True)
            print(f"Deleted temporary relationships table {temp_rels_table_ref.table_id}.")

        # 8. Create Graph Table
        print("\n--- Creating property graph ---")
        graph_name = f"{dataset_id}.graph_pass{pass_number}"
        create_graph_sql = f"""
        CREATE OR REPLACE PROPERTY GRAPH `{graph_name}`
          NODE TABLES (
            `{nodes_table_name}`
              KEY (entity_name)
              LABEL entity_type PROPERTIES (entity_name, source_document_count, support_ticket_count, buganizer_count, reddit_count, video_count, source_document_url, support_tickets, buganizer_ids, reddit_urls, video_urls)
          )
          EDGE TABLES (
            `{relationships_table_name}`
              KEY (source_entity_name, target_entity_name, relationship_type)
              SOURCE KEY (source_entity_name) REFERENCES `{nodes_table_name}` (entity_name)
              DESTINATION KEY (target_entity_name) REFERENCES `{nodes_table_name}` (entity_name)
              LABEL relationship_type PROPERTIES (source_entity_name, target_entity_name, relationship_type, cooccurrence_frequency, extraction_frequency, relevance_score)
          );
        """
        run_query(create_graph_sql, f"Create property graph '{graph_name}'")

        print(f"\nBigQuery graph database creation for pass {pass_number} completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during the process for pass {pass_number}: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # The BigQuery dataset to create/use.
    DATASET_NAME = "bq_graph_db"
    # The GCP location for the dataset.
    LOCATION = "us-central1"

    # --- Execution ---
    project_root = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 5):
        json_file = f'consolidated_graph_with_counts_pass{i}.json'
        json_path = os.path.join(project_root, "json_output", json_file)

        if not os.path.exists(json_path):
            print(f"Error: Could not find '{json_file}' in the 'json_output' directory. Skipping pass {i}.")
            continue

        print(f"\n--- Starting processing for pass {i} with file '{json_file}' ---")
        create_bq_graph_from_json(
            json_file_path=json_path,
            project_id=PROJECT_ID,
            pass_number=i,
            dataset_name=DATASET_NAME,
            location=LOCATION
        )
