import os
from google.cloud import bigquery
from rich.console import Console
from rich.table import Table

# It's good practice to have configuration separate.
try:
    from config import PROJECT_ID, DATASET_ID
except ImportError:
    print("Error: config.py not found or missing variables.")
    print("Please create it with your PROJECT_ID and DATASET_ID.")
    exit(1)

# --- Configuration ---
PASS_NUMBER = 1 # Which graph pass to query
NODES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nodes_pass{PASS_NUMBER}"
TOP_N = 20 # How many top nodes to show

# --- Initialize Clients ---
console = Console()
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    print(f"Successfully connected to BigQuery project: {PROJECT_ID}")
except Exception as e:
    console.print(f"[bold red]Error connecting to BigQuery: {e}[/bold red]")
    exit(1)

def find_hotspot_nodes():
    """
    Queries the knowledge graph to find nodes with the highest number of
    associated support tickets.
    """
    console.print(f"\n[bold blue]Finding top {TOP_N} 'hotspot' nodes from '{NODES_TABLE}'...[/bold blue]")

    query = f"""
        SELECT
            name,
            label,
            ARRAY_LENGTH(support_tickets) AS ticket_count
        FROM
            `{NODES_TABLE}`
        WHERE
            support_tickets IS NOT NULL AND ARRAY_LENGTH(support_tickets) > 0
        ORDER BY
            ticket_count DESC
        LIMIT {TOP_N}
    """

    try:
        query_job = bq_client.query(query)
        results = list(query_job.result())

        if not results:
            console.print("[yellow]No nodes with associated support tickets found.[/yellow]")
            return

        table = Table(title=f"Top {TOP_N} Knowledge Graph Hotspots (Most Support Tickets)")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Node Name", style="magenta")
        table.add_column("Node Label", style="green")
        table.add_column("Ticket Count", style="bold red", justify="right")

        for i, row in enumerate(results):
            table.add_row(str(i + 1), row.name, row.label, str(row.ticket_count))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]An error occurred while querying for hotspots: {e}[/bold red]")

def main():
    """Main function to run the hotspot analysis."""
    find_hotspot_nodes()

if __name__ == "__main__":
    main()