import os
import csv
import re
import difflib
import html
import pandas as pd
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
# Set this to a list of specific URL strings to test with.
# If set to None or an empty list, the script will process all unique URLs from the graph.
# Example: TEST_URLS = ["https://cloud.google.com/bigquery/docs/external-data-sources"]
TEST_URLS = [
    "https://cloud.google.com/bigquery/docs/migration/redshift-vpc",
    "https://cloud.google.com/bigquery/docs/using-row-level-security-with-features",
    "https://cloud.google.com/bigquery/docs/external-table-definition",
]
NODES_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nodes_pass{PASS_NUMBER}"
URL_CONTENT_FILE = "files/url_list_with_content.csv"
SUPPORT_TICKET_FILE = "support_tickets/support_ticket_description.csv"
PROMPT_FILE = "prompts/doc_update_recommendations.txt"
RECOMMENDATIONS_DIR = "recommendations"
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
def sanitize_url_for_filename(url: str) -> str:
    """Removes protocol and replaces invalid characters to create a valid filename."""
    # Remove protocol
    sanitized = re.sub(r'^https?:\/\/', '', url)
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

def get_unique_urls():
    """Fetches a list of unique source_document_url from the nodes table."""
    console.print(f"Fetching unique URLs from BigQuery table: {NODES_TABLE}...")
    query = f"SELECT DISTINCT url FROM `{NODES_TABLE}`, UNNEST(source_document_url) AS url WHERE url IS NOT NULL"
    try:
        query_job = bq_client.query(query)
        results = [row.url for row in query_job.result()]
        console.print(f"Found {len(results)} unique URLs to process.")
        return results
    except Exception as e:
        console.print(f"[bold red]Error fetching unique URLs from BigQuery: {e}[/bold red]")
        return []

def get_tickets_for_url(url):
    """Fetches the list of all unique support tickets for a given URL by aggregating from all associated nodes."""
    # This query first finds all nodes containing the URL in a Common Table
    # Expression (CTE), and then unnests the support_tickets array from that
    # filtered result. This correctly handles cases where a URL is associated
    # with a node that has no support tickets, ensuring all relevant tickets
    # from all associated nodes are aggregated.
    query = f"""
    WITH nodes_with_url AS (
      SELECT support_tickets
      FROM `{NODES_TABLE}`
      WHERE @url IN UNNEST(source_document_url)
    )
    SELECT ARRAY_AGG(DISTINCT ticket IGNORE NULLS) AS all_tickets
    FROM nodes_with_url, UNNEST(support_tickets) AS ticket
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("url", "STRING", url)]
    )
    try:
        results = list(bq_client.query(query, job_config=job_config).result())
        # The aggregated array might be NULL if no tickets are found.
        return results[0].all_tickets if results and results[0].all_tickets else []
    except Exception as e:
        console.print(f"[bold red]Error fetching tickets for URL {url}: {e}[/bold red]")
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

def parse_gemini_response(text):
    """Parses the Gemini response to extract structured data."""
    # Regex to find the content under each heading.
    # It looks for "###" followed by a number and a dot, then the title.
    # It captures everything until the next "###" or the end of the string.
    # re.DOTALL makes '.' match newlines. re.IGNORECASE handles case variations.
    key_issues_pattern = r"###\s*\d*\.\s*Key Customer Issues\s*\n(.*?)(?=\n###|\Z)"
    updated_text_pattern = r"###\s*\d*\.\s*Full Updated Documentation\s*\n(.*?)(?=\n###|\Z)"

    key_issues_match = re.search(key_issues_pattern, text, re.DOTALL | re.IGNORECASE)
    updated_text_match = re.search(updated_text_pattern, text, re.DOTALL | re.IGNORECASE)

    # The "Key Customer Issues" section now serves as the main summary/recommendation.
    key_issues = key_issues_match.group(1).strip() if key_issues_match else "Could not parse recommendations."
    updated_text = updated_text_match.group(1).strip() if updated_text_match else "Could not parse updated text."

    # We will use the "key_issues" for both the console output (recommendations_text)
    # and the CSV summary.
    return {
        "summary": key_issues,
        "recommendations_text": key_issues,
        "updated_text": updated_text,
    }

def generate_content_html_files(df, recommendations_dir):
    """Generates individual HTML files for original and updated text."""
    content_files_dir = os.path.join(recommendations_dir, "content_files")
    os.makedirs(content_files_dir, exist_ok=True)
    console.print(f"\nGenerating individual HTML content files in '{content_files_dir}'...")

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

    for _, row in df.iterrows():
        url = row['url']
        base_filename = sanitize_url_for_filename(url)
        
        original_filename = os.path.join(content_files_dir, f"{base_filename}_original.html")
        updated_filename = os.path.join(content_files_dir, f"{base_filename}_updated.html")

        with open(original_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Original: {url}", row['original_text']))
        with open(updated_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Updated: {url}", row['updated_text']))
            
    console.print(f"[bold green]Successfully created {len(df)*2} HTML content files.[/bold green]")

def generate_diff_html_files(df, recommendations_dir):
    """Generates an individual HTML diff file for each URL."""
    diff_files_dir = os.path.join(recommendations_dir, "diff_files")
    os.makedirs(diff_files_dir, exist_ok=True)
    console.print(f"\nGenerating individual HTML diff files in '{diff_files_dir}'...")

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

    for _, row in df.iterrows():
        url = row['url']
        base_filename = sanitize_url_for_filename(url)

        original_text = row['original_text'].splitlines()
        updated_text = row['updated_text'].splitlines()

        # Relative paths from diff_files/ to content_files/
        original_html_path = f"../content_files/{base_filename}_original.html"
        updated_html_path = f"../content_files/{base_filename}_updated.html"
        
        diff_table_html = differ.make_table(
            original_text, 
            updated_text, 
            fromdesc="Original Text", 
            todesc="Updated Text"
        )

        full_html = f"""<!DOCTYPE html>
<html><head><title>Diff for: {html.escape(url)}</title>
<style>{css} body {{ font-family: sans-serif; }} h1 {{ font-size: 1.5em; }} a {{ color: #0366d6; }} p {{ margin-top: 0; }} hr {{ border: 0; border-top: 1px solid #e1e4e8; margin: 24px 0; }}</style>
</head><body>
<h1>Diff for: <a href='{html.escape(url)}' target='_blank'>{html.escape(url)}</a></h1>
<p><a href='{html.escape(original_html_path)}' target='_blank'>View Original as HTML</a> | <a href='{html.escape(updated_html_path)}' target='_blank'>View Updated as HTML</a></p><hr>
{diff_table_html}
</body></html>"""
        
        diff_filename = os.path.join(diff_files_dir, f"{base_filename}_diff.html")
        with open(diff_filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
    console.print(f"[bold green]Successfully created {len(df)} HTML diff files.[/bold green]")

def main():
    """Main function to orchestrate the recommendation generation process."""
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
    url_df, ticket_df = load_dataframes()
    prompt_template = get_prompt_template()
    results_data = []

    if TEST_URLS:
        console.print(f"[bold yellow]Running in test mode for {len(TEST_URLS)} specific URL(s).[/bold yellow]")
        unique_urls = TEST_URLS
    else:
        unique_urls = get_unique_urls()

    if not unique_urls:
        console.print("[bold yellow]No URLs found to process. Exiting.[/bold yellow]")
        return

    for i, url in enumerate(unique_urls):
        console.rule(f"[bold cyan]Processing URL {i+1}/{len(unique_urls)}: {url}[/bold cyan]")
        try:
            # Using .get() is safer in case the 'text' column is missing for some reason.
            url_content = url_df.loc[url].get('text')
            # Check for empty or NaN content which can cause API errors.
            if pd.isna(url_content) or not str(url_content).strip():
                console.print(f"[yellow]Warning: URL '{url}' has empty content in the CSV. Skipping.[/yellow]")
                continue
            ticket_ids = get_tickets_for_url(url)
            if not ticket_ids:
                console.print("[yellow]No associated support tickets found for this URL. Skipping.[/yellow]")
                continue

            ticket_contents = [f"--- TICKET: {tid} ---\nSUBJECT: {ticket_df.loc[tid].get('case_subject', 'N/A')}\nDESCRIPTION: {ticket_df.loc[tid].get('description', 'N/A')}\n" for tid in ticket_ids if tid in ticket_df.index]
            if not ticket_contents:
                console.print(f"[yellow]Could not retrieve content for any associated tickets: {ticket_ids}. Skipping URL.[/yellow]")
                continue

            # --- New Debugging Prints ---
            console.print("\n[bold blue]---------- DEBUGGING INFO ----------[/bold blue]")
            console.print(f"[bold]URL Content Snippet ({len(str(url_content))} chars):[/bold]\n{str(url_content)[:500]}...")
            console.print(f"\n[bold]Support Ticket Snippets ({len(ticket_contents)} tickets):[/bold]")
            for i, ticket_text in enumerate(ticket_contents):
                # Replace newline characters for cleaner printing before the f-string formatting
                cleaned_text = ticket_text[:200].replace('\n', ' ')
                console.print(f"  Ticket {i+1}: {cleaned_text}...")
            console.print("[bold blue]------------------------------------[/bold blue]\n")

            console.print("Constructing prompt and sending to Gemini...")
            # Set a low temperature for more deterministic and consistent results
            generation_config = genai.types.GenerationConfig(
                temperature=TEMPERATURE
            )
            final_prompt = prompt_template.format(url_content=url_content, support_tickets_content="\n".join(ticket_contents))
            response = gemini_model.generate_content(
                final_prompt,
                generation_config=generation_config)

            # Parse the structured response from Gemini
            parsed_response = parse_gemini_response(response.text)

            console.print("\n[bold green]Gemini Recommendations:[/bold green]")
            # Print only the recommendations part to the console
            console.print(Markdown(parsed_response["recommendations_text"]))

            # Store all data for final output files
            results_data.append({
                "url": url,
                "gemini_summary": parsed_response["summary"],
                "original_text": url_content,
                "updated_text": parsed_response["updated_text"],
            })
        except KeyError:
            console.print(f"[yellow]Warning: URL '{url}' not found in content file. Skipping.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred for URL {url}: {e}[/bold red]")

    if not results_data:
        console.print("[bold yellow]No recommendations were generated. Exiting.[/bold yellow]")
        return

    df = pd.DataFrame(results_data)
    df.to_csv(RECOMMENDATIONS_CSV_FILE, index=False, quoting=csv.QUOTE_ALL)
    console.print(f"\n[bold green]Successfully saved {len(df)} results to '{RECOMMENDATIONS_CSV_FILE}'[/bold green]")
    generate_content_html_files(df, RECOMMENDATIONS_DIR)
    generate_diff_html_files(df, RECOMMENDATIONS_DIR)

if __name__ == "__main__":
    main()