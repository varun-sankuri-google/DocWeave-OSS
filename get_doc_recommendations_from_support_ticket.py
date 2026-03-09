import os
import csv
import re
import difflib
import html
import pandas as pd
import json
from google.cloud import bigquery
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown

# It's good practice to have configuration separate.
try:
    from config import PROJECT_ID, DATASET_ID, GOOGLE_API_KEY, GENERATION_MODEL, TEMPERATURE
except ImportError:
    print("Error: config.py not found or missing variables.")
    print("Please create it with your PROJECT_ID, DATASET_ID, GOOGLE_API_KEY, and GENERATION_MODEL.")
    print("Example config.py:\nPROJECT_ID = 'your-gcp-project-id'\nDATASET_ID = 'bq_graph_db'\nGOOGLE_API_KEY = 'your-gemini-api-key'\nGENERATION_MODEL = 'gemini-1.5-flash'")
    exit(1)

# --- Configuration ---
PASS_NUMBER = 1 # Which graph pass to query
# Set this to a list of specific ticket IDs to test with.
# If empty, it will process all tickets from the source file.
TEST_TICKET_IDS = ["500Kf00000dNL1MIAW"] # Example ticket IDs
NODES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nodes_pass{PASS_NUMBER}"
URL_CONTENT_FILE = "files/url_list_with_content.csv"
SUPPORT_TICKET_FILE = "support_tickets/support_ticket_description.csv"
PROMPT_FILE = "prompts/doc_update_recommendations_ticket_level.txt" # This will be the updated prompt
RECOMMENDATIONS_DIR = "recommendations_from_tickets"
RECOMMENDATIONS_CSV_FILE = os.path.join(RECOMMENDATIONS_DIR, "recommendations.csv")

# --- Initialize Clients ---
console = Console()
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    print(f"Successfully connected to BigQuery project: {PROJECT_ID}")
except Exception as e:
    console.print(f"[bold red]Error connecting to BigQuery: {e}[/bold red]")
    exit(1)

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    console.print("[bold red]Error: Please set your GOOGLE_API_KEY in config.py[/bold red]")
    exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel(GENERATION_MODEL)
    print(f"Successfully configured Gemini model: {GENERATION_MODEL}")
except Exception as e:
    console.print(f"[bold red]Error configuring Gemini: {e}[/bold red]")
    exit(1)

# --- Helper Functions ---
def sanitize_for_filename(text: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    # Remove protocol if it's a URL
    sanitized = re.sub(r'^https?:\/\/', '', text)
    # Replace invalid filename characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', sanitized)
    return sanitized

def load_dataframes():
    """Loads the necessary CSV files into pandas DataFrames."""
    try:
        console.print(f"Loading URL content from '{URL_CONTENT_FILE}'...")
        url_df = pd.read_csv(URL_CONTENT_FILE).set_index('page_url')
        console.print(f"Loaded {len(url_df)} URLs.")

        console.print(f"Loading support tickets from '{SUPPORT_TICKET_FILE}'...")
        # The CSV has a 'case_id' column which we will use as the index.
        ticket_df = pd.read_csv(SUPPORT_TICKET_FILE).set_index('case_id')
        console.print(f"Loaded {len(ticket_df)} support tickets.")

        return url_df, ticket_df
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: Data file not found. {e}[/bold red]")
        console.print(f"Please ensure '{URL_CONTENT_FILE}' and '{SUPPORT_TICKET_FILE}' exist.")
        exit(1)
    except KeyError as e:
        console.print(f"[bold red]Error: Missing expected column in CSV file: {e}[/bold red]")
        console.print("Please check that 'page_url' and 'case_id' are the correct index columns.")
        exit(1)

def get_nodes_for_ticket(ticket_id: str):
    """Fetches all nodes associated with a given support ticket ID."""
    query = f"""
    SELECT DISTINCT entity_name AS name
    FROM `{NODES_TABLE}`
    WHERE @ticket_id IN UNNEST(support_tickets)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("ticket_id", "STRING", str(ticket_id))]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        return [row.name for row in results]
    except Exception as e:
        console.print(f"[bold red]Error fetching nodes for ticket {ticket_id}: {e}[/bold red]")
        return []

def get_urls_for_nodes(node_names: list):
    """Fetches all unique URLs associated with a list of node names."""
    if not node_names:
        return []
    query = f"""
    SELECT DISTINCT url
    FROM `{NODES_TABLE}` AS t, UNNEST(t.source_document_url) AS url
    WHERE t.entity_name IN UNNEST(@node_names) AND url IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("node_names", "STRING", node_names)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        return [row.url for row in results]
    except Exception as e:
        console.print(f"[bold red]Error fetching URLs for nodes: {e}[/bold red]")
        return []

def get_prompt_template():
    """Loads the prompt template from the specified file."""
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
            if not template.strip():
                console.print(f"[bold red]Error: Prompt file '{PROMPT_FILE}' is empty. Cannot generate recommendations.[/bold red]")
                exit(1)
            return template
    except FileNotFoundError:
        console.print(f"[bold red]Error: Prompt file not found at '{PROMPT_FILE}'[/bold red]")
        exit(1)

def parse_gemini_response(text: str):
    """Parses the Gemini JSON response."""
    try:
        # The new prompt asks for JSON directly.
        # Clean up potential markdown formatting.
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3]
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        console.print(f"[bold red]Could not parse Gemini JSON response: {e}[/bold red]")
        console.print(f"Response text was: {text[:500]}...")
        return None

def generate_diff_and_content_files(results_df, base_dir):
    """Generates HTML diff and content files for each recommendation."""
    diff_dir = os.path.join(base_dir, "diff_files")
    content_dir = os.path.join(base_dir, "content_files")
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)
    
    console.print(f"\nGenerating HTML diff and content files in '{base_dir}'...")

    # Helper to create a simple HTML document
    def create_html_doc(title, body_content):
        # Using <pre> preserves whitespace and newlines from the text content
        # Also escape HTML characters in the content to prevent them from being rendered as HTML
        escaped_content = html.escape(body_content)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>{html.escape(title)}</title>
<meta charset="UTF-8">
<style>
  body {{ font-family: sans-serif; line-height: 1.6; padding: 2em; margin: 0 auto; max-width: 800px; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 1em; border-radius: 5px; border: 1px solid #ddd;}}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <pre>{escaped_content}</pre>
</body>
</html>"""
    
    differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    # CSS from Python's difflib.py for consistent styling
    css = """
        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
    """
    
    for _, row in results_df.iterrows():
        ticket_id = str(row['ticket_id'])
        url = row['url']
        base_filename = f"ticket_{ticket_id}_{sanitize_for_filename(url)}"

        # Generate content files
        original_filename = os.path.join(content_dir, f"{base_filename}_original.html")
        updated_filename = os.path.join(content_dir, f"{base_filename}_updated.html")
        
        with open(original_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Original: {url}", row['original_text']))
        with open(updated_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Updated: {url}", row['updated_text']))

        # Relative paths from diff_files/ to content_files/
        diff_table_html = differ.make_table(
            row['original_text'].splitlines(),
            row['updated_text'].splitlines(),
            fromdesc="Original Text",
            todesc="Updated Text"
        )

        full_html = f"""<!DOCTYPE html>
<html><head><title>Diff for: {html.escape(url)}</title>
<style>{css} body {{ font-family: sans-serif; }} h1 {{ font-size: 1.5em; }} a {{ color: #0366d6; }} p {{ margin-top: 0; }} hr {{ border: 0; border-top: 1px solid #e1e4e8; margin: 24px 0; }}</style>
</head><body>
<h1>Diff for: <a href='{html.escape(url)}' target='_blank'>{html.escape(url)}</a> (Ticket: {ticket_id})</h1>
<p><a href='../content_files/{os.path.basename(original_filename)}' target='_blank'>View Original</a> | <a href='../content_files/{os.path.basename(updated_filename)}' target='_blank'>View Updated</a></p><hr>
{diff_table_html}
</body></html>"""
        
        diff_filename = os.path.join(diff_dir, f"{base_filename}_diff.html")
        with open(diff_filename, 'w', encoding='utf-8') as f:
            f.write(full_html)

    console.print(f"[bold green]Successfully created {len(results_df)} diff files and {len(results_df)*2} content files.[/bold green]")

def main():
    """Main function to orchestrate the recommendation generation process."""
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
    url_df, ticket_df = load_dataframes()
    prompt_template = get_prompt_template()
    all_results = []

    if TEST_TICKET_IDS:
        ticket_ids_to_process = TEST_TICKET_IDS
        console.print(f"[bold yellow]Running in test mode for {len(ticket_ids_to_process)} specific ticket(s).[/bold yellow]")
    else:
        ticket_ids_to_process = ticket_df.index.tolist()

    if not ticket_ids_to_process:
        console.print("[bold yellow]No ticket IDs found to process. Exiting.[/bold yellow]")
        return

    for i, ticket_id in enumerate(ticket_ids_to_process):
        console.rule(f"[bold cyan]Processing Ticket {i+1}/{len(ticket_ids_to_process)}: {ticket_id}[/bold cyan]")
        
        try:
            ticket_info = ticket_df.loc[ticket_id]
            ticket_content_str = f"--- TICKET: {ticket_id} ---\nSUBJECT: {ticket_info.get('case_subject', 'N/A')}\nDESCRIPTION: {ticket_info.get('description', 'N/A')}\n"
        except (KeyError, ValueError):
            console.print(f"[yellow]Warning: Ticket ID '{ticket_id}' not found in support ticket file. Skipping.[/yellow]")
            continue

        # 1. Find nodes for the ticket
        node_names = get_nodes_for_ticket(ticket_id)
        if not node_names:
            console.print(f"[yellow]No associated nodes found for ticket {ticket_id}. Skipping.[/yellow]")
            continue
        console.print(f"Found {len(node_names)} associated nodes: {node_names[:5]}...")

        # 2. Find URLs for the nodes
        urls = get_urls_for_nodes(node_names)
        if not urls:
            console.print(f"[yellow]No associated URLs found for the nodes. Skipping.[/yellow]")
            continue
        console.print(f"Found {len(urls)} associated URLs.")

        # 3. Gather document content
        docs_content_list = []
        original_docs = {}
        for url in urls:
            try:
                content = url_df.loc[url].get('text')
                if pd.notna(content) and str(content).strip():
                    docs_content_list.append(f"--- DOCUMENT URL: {url} ---\n{content}\n")
                    original_docs[url] = content
                else:
                    console.print(f"[yellow]Warning: No content found for URL '{url}'. It will not be sent to the model.[/yellow]")
            except KeyError:
                console.print(f"[yellow]Warning: URL '{url}' not found in content file. Skipping.[/yellow]")
        
        if not docs_content_list:
            console.print("[yellow]No valid document content could be gathered. Skipping Gemini call.[/yellow]")
            continue

        docs_content_str = "\n".join(docs_content_list)

        # 4. Call Gemini
        console.print("Constructing prompt and sending to Gemini...")
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                response_mime_type="application/json"
            )
            final_prompt = prompt_template.format(ticket_content=ticket_content_str, docs_content=docs_content_str)
            
            response = gemini_model.generate_content(
                final_prompt,
                generation_config=generation_config
            )
            parsed_response = parse_gemini_response(response.text)
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred during Gemini call for ticket {ticket_id}: {e}[/bold red]")
            continue

        if not parsed_response or "updates" not in parsed_response:
            console.print(f"[yellow]No valid updates received from Gemini for ticket {ticket_id}.[/yellow]")
            continue

        # 5. Process and store results
        console.print(f"\n[bold green]Gemini Analysis Summary:[/bold green] {parsed_response.get('analysis_summary', 'N/A')}")
        updates = parsed_response.get("updates", {})
        if not updates:
            console.print("[yellow]Gemini determined no documents needed updates for this ticket.[/yellow]")
        
        for url, update_details in updates.items():
            if url in original_docs:
                console.print(f"  -> Recommendation to update: [blue]{url}[/blue]")
                console.print(f"     Reason: {update_details.get('reason_for_update', 'N/A')}")
                all_results.append({
                    "ticket_id": ticket_id,
                    "url": url,
                    "analysis_summary": parsed_response.get('analysis_summary'),
                    "reason_for_update": update_details.get('reason_for_update'),
                    "original_text": original_docs[url],
                    "updated_text": update_details.get('updated_content'),
                })

    if not all_results:
        console.print("\n[bold yellow]No recommendations were generated across all tickets. Exiting.[/bold yellow]")
        return

    # 6. Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RECOMMENDATIONS_CSV_FILE, index=False, quoting=csv.QUOTE_ALL)
    console.print(f"\n[bold green]Successfully saved {len(results_df)} recommendations to '{RECOMMENDATIONS_CSV_FILE}'[/bold green]")
    
    generate_diff_and_content_files(results_df, RECOMMENDATIONS_DIR)

if __name__ == "__main__":
    main()
