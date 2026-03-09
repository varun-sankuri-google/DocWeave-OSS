import os
import json
import textwrap
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
from google.cloud import bigquery
import pandas as pd
import google.generativeai as genai
import argparse
import time
from collections import defaultdict
import csv
import re
import difflib
import html
from parse_url import clean_url, normalize_url

# Assuming config.py exists and contains the necessary variables
try:
    from config import GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY, TEMPERATURE
except ImportError:
    print("Error: config.py not found or is missing required variables.")
    print("Please create it with GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY, and TEMPERATURE.")
    exit(1)

# --- Configuration ---
DATASET_NAME = "bq_graph_db"
PASS_NUMBER = 1  # The pass number of the graph to query
TEST_BUGANIZER_IDS = [
    # "294626507","294844998","295079838","315385723",
"318490599"
# ,"318675627","319252892","322794137","331105631","342227144"
]
BUGANIZER_ISSUES_CSV = os.path.join(os.path.dirname(__file__), "support_tickets", "buganizer_issues.csv")
URL_CONTENT_CACHE_CSV = os.path.join(os.path.dirname(__file__), "files", "url_list_with_content.csv")
RECOMMENDATIONS_DIR = "recommendations_from_tickets"
RECOMMENDATIONS_CSV_FILE = os.path.join(RECOMMENDATIONS_DIR, "recommendations.csv")


def sanitize_for_filename(text: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    # Remove protocol if it's a URL
    sanitized = re.sub(r'^https?://', '', text)
    # Replace invalid filename characters with underscores
    sanitized = re.sub(r'[\\/*?",<>|]', '_', sanitized)
    return sanitized

def generate_diff_and_content_files(results_df, base_dir):
    """Generates HTML diff and content files for each recommendation."""
    diff_dir = os.path.join(base_dir, "diff_files")
    content_dir = os.path.join(base_dir, "content_files")
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)
    
    print(f"Generating HTML diff and content files in '{base_dir}'...")

    # Helper to create a simple HTML document
    def create_html_doc(title, body_content):
        # Using <pre> preserves whitespace and newlines from the text content
        # Also escape HTML characters in the content to prevent them from being rendered as HTML
        escaped_content = html.escape(body_content)
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<title>{html.escape(title)}</title>
<meta charset="UTF-8">
<style>
  body {{ font-family: sans-serif; line-height: 1.6; padding: 2em; margin: 0 auto; max-width: 800px; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 1em; border-radius: 5px; border: 1px solid #ddd;}}
  .summary-box {{ background-color: #eef; border-left: 5px solid #66c; padding: 1em; margin-bottom: 1em; }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <pre>{escaped_content}</pre>
</body>
</html>'''
    
    differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    # CSS from Python's difflib.py for consistent styling
    css = """        table.diff {font-family:Courier; border:medium;}
        .diff_header {background-color:#e0e0e0}
        td.diff_header {text-align:right}
        .diff_next {background-color:#c0c0c0}
        .diff_add {background-color:#aaffaa}
        .diff_chg {background-color:#ffff77}
        .diff_sub {background-color:#ffaaaa}
        .summary-box {{ background-color: #eef; border-left: 5px solid #66c; padding: 1em; margin-bottom: 1em; }}
    """
    
    for _, row in results_df.iterrows():
        buganizer_id = str(row['buganizer_id'])
        url = row['url']
        bug_summary = row.get('bug_summary', 'Not available.')
        gemini_reasoning = row.get('gemini_reasoning', 'Not available.')
        base_filename = f"bug_{buganizer_id}_{sanitize_for_filename(url)}"

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

        style_block = (
            "<style>"
            + css
            + " body { font-family: sans-serif; } h1 { font-size: 1.5em; } a { color: #0366d6; } p { margin-top: 0; } hr { border: 0; border-top: 1px solid #e1e4e8; margin: 24px 0; }"
            + "</style>"
        )

        full_html = f'''<!DOCTYPE html>
<html><head><title>Diff for: {html.escape(url)}</title>
{style_block}
</head><body>
<h1>Diff for: <a href='../content_files/{os.path.basename(original_filename)}' target='_blank'>Original</a> | <a href='../content_files/{os.path.basename(updated_filename)}' target='_blank'>Updated</a> | <a href='{html.escape(url)}' target='_blank'>{html.escape(url)}</a> (Buganizer ID: {buganizer_id})</h1>
<div class="summary-box">
    <h2>Bug Summary</h2>
    <p>{html.escape(bug_summary)}</p>
    <h2>Gemini Reasoning</h2>
    <p>{html.escape(gemini_reasoning)}</p>
</div>
<hr>
{diff_table_html}
</body></html>'''
        
        diff_filename = os.path.join(diff_dir, f"{base_filename}_diff.html")
        with open(diff_filename, 'w', encoding='utf-8') as f:
            f.write(full_html)

    print(f"Successfully created {len(results_df)} diff files and {len(results_df)*2} content files.")

def generate_index_html(base_dir):
    """Generates a root index.html to navigate all diffs with search functionality."""
    diff_dir = os.path.join(base_dir, "diff_files")
    index_filename = os.path.join(base_dir, "index.html")
    
    if not os.path.exists(diff_dir):
        print("Diff directory not found. Cannot generate index.html.")
        return

    diff_files = [f for f in os.listdir(diff_dir) if f.endswith('_diff.html')]
    
    links_html = ""
    if diff_files:
        links_html += "<ul id='report-list'>"
        for f in sorted(diff_files):
            # Add target="_blank" to open links in a new tab
            links_html += f'<li><a href="diff_files/{f}" target="_blank">{f}</a></li>'
        links_html += "</ul>"
    else:
        links_html = "<p>No diff files found.</p>"

    index_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
<title>Documentation Update Recommendations</title>
<meta charset="UTF-8">
<style>
  body {{ font-family: sans-serif; line-height: 1.6; padding: 2em; margin: 0 auto; max-width: 1000px; }}
  h1 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
  #search-box {{ width: 100%; box-sizing: border-box; padding: 10px; font-size: 16px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; }}
  ul {{ list-style-type: none; padding: 0; }}
  li {{ padding: 8px; border-bottom: 1px solid #ddd; }}
  li:hover {{ background-color: #f6f8fa; }}
  a {{ text-decoration: none; color: #0366d6; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
  <h1>Documentation Update Recommendations</h1>
  <input type="text" id="search-box" onkeyup="filterReports()" placeholder="Search for reports by URL or keyword...">
  {links_html}

<script>
function filterReports() {{
    var input, filter, ul, li, a, i, txtValue;
    input = document.getElementById('search-box');
    filter = input.value.toUpperCase();
    ul = document.getElementById("report-list");
    li = ul.getElementsByTagName('li');

    for (i = 0; i < li.length; i++) {{
        a = li[i].getElementsByTagName("a")[0];
        txtValue = a.textContent || a.innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {{
            li[i].style.display = "";
        }} else {{
            li[i].style.display = "none";
        }}
    }}
}}
</script>

</body>
</html>'''
    
    with open(index_filename, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"Successfully generated index.html with search functionality at '{{index_filename}}'")


def get_bq_client(project_id: str):
    """Initializes and returns a BigQuery client."""
    try:
        client = bigquery.Client(project=project_id)
        print(f"Successfully connected to BigQuery project: {project_id}")
        return client
    except Exception as e:
        print(f"Error connecting to BigQuery: {e}")
        return None

def load_url_content_cache(file_path: str) -> dict[str, str]:
    """Loads URL content from a CSV file into a dictionary."""
    print(f"Loading URL content cache from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        return pd.Series(df.text.values, index=df.page_url).to_dict()
    except FileNotFoundError:
        print(f"Warning: Cache file not found at {file_path}. Will fetch all content from the web.")
        return {}

def get_human_verified_urls(buganizer_ids: list[str], file_path: str) -> list[str]:
    """Reads buganizer_issues.csv to get human-verified URLs for given buganizer IDs."""
    print(f"Getting human-verified URLs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # bug_issue_id is int in csv, but string in our list
        df['bug_issue_id'] = df['bug_issue_id'].astype(str)
        filtered_df = df[df['bug_issue_id'].isin(buganizer_ids)]
        urls = []
        for links_str in filtered_df['links']:
            try:
                urls.extend(json.loads(links_str))
            except json.JSONDecodeError:
                # Handle cases where the links column is not a valid JSON array string
                pass
        return list(set(urls))
    except FileNotFoundError:
        print(f"Warning: Buganizer issues file not found at {file_path}.")
        return []

def get_all_buganizer_ids_from_bq(client: bigquery.Client, project_id: str, dataset_id: str, pass_number: int) -> list[str]:
    """
    Queries the BigQuery nodes table to get all unique Buganizer IDs.
    """
    table_name = f"{project_id}.{dataset_id}.nodes_pass{pass_number}"
    query = f'''
        SELECT DISTINCT b_id
        FROM `{table_name}`, 
        UNNEST(buganizer_ids) AS b_id
    '''
    try:
        print("Querying BigQuery for all Buganizer IDs...")
        query_job = client.query(query)
        rows = query_job.result()
        buganizer_ids = [row.b_id for row in rows]
        print(f"Found {len(buganizer_ids)} unique Buganizer IDs.")
        return buganizer_ids
    except Exception as e:
        print(f"An error occurred while querying BigQuery for all Buganizer IDs: {e}")
        return []

def get_urls_from_buganizer_ids(client: bigquery.Client, project_id: str, dataset_id: str, pass_number: int, buganizer_ids: list[str]) -> list[str]:
    """
    Queries the BigQuery nodes table to get the first URL from each node
    associated with a list of Buganizer IDs.
    """
    table_name = f"{project_id}.{dataset_id}.nodes_pass{pass_number}"
    query = f'''
        SELECT DISTINCT
            (SELECT url FROM UNNEST(t.source_document_url) AS url LIMIT 1) as url
        FROM `{table_name}` AS t,
        UNNEST(t.buganizer_ids) AS b_id
        WHERE b_id IN UNNEST(@buganizer_ids)
    '''
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("buganizer_ids", "STRING", buganizer_ids),
        ]
    )
    try:
        print("Querying BigQuery for relevant URLs...")
        query_job = client.query(query, job_config=job_config)
        rows = query_job.result()
        urls = [row.url for row in rows if row.url is not None]
        print(f"Found {len(urls)} relevant URLs from BigQuery.")
        return urls
    except Exception as e:
        print(f"An error occurred while querying BigQuery: {e}")
        return []

def get_url_content(url: str) -> str | None:
    """Fetches and parses the text content of a URL."""
    normalized_url = normalize_url(url)
    try:
        with urllib.request.urlopen(normalized_url) as response:
            if response.getcode() == 200:
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator='\n', strip=True)
            else:
                print(f"Failed to fetch {normalized_url}: Status code {response.getcode()}")
                return None
    except urllib.error.URLError as e:
        print(f"Error fetching {normalized_url}: {e.reason}")
        return None

def get_recommendations(buganizer_ids: list[str], urls: list[str], human_verified_urls: list[str], url_content_cache: dict[str, str], original_docs: dict[str, str], buganizer_issues_csv: str):
    """
    Analyzes Buganizer issues and associated documentation to provide recommendations.
    """
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GENERATION_MODEL)

    try:
        df = pd.read_csv(buganizer_issues_csv)
        df['bug_issue_id'] = df['bug_issue_id'].astype(str)
        filtered_df = df[df['bug_issue_id'].isin(buganizer_ids)]
        
        buganizer_issue_content_list = []
        for _, row in filtered_df.iterrows():
            bug_id = row['bug_issue_id']
            text = row['text']
            links = row['links']
            buganizer_issue_content_list.append(f"- ID: {bug_id}\n- Description: {text}\n- Links: {links}")
        
        buganizer_issue_content = "\n".join(buganizer_issue_content_list)

    except FileNotFoundError:
        print(f"Warning: Buganizer issues file not found at {buganizer_issues_csv}.")
        buganizer_issue_content = ""


    docs_content_list = []
    for url in urls:
        normalized_url = normalize_url(url)
        content = url_content_cache.get(normalized_url)

        if not content:
            print(f"  Content for {normalized_url} not in cache. Fetching from web...")
            content = get_url_content(normalized_url)
        
        if content:
            # IMPORTANT: Always use the normalized URL as the key for the prompt and results
            docs_content_list.append(f"\n\n---\nDOCUMENT URL: {normalized_url}\n---\n{content}")
            original_docs[normalized_url] = content
        else:
            print(f"  Could not retrieve content for {normalized_url}")

    if not docs_content_list:
        print("Could not retrieve content for any of the URLs. Aborting.")
        return

    docs_content_str = "\n".join(docs_content_list)

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "doc_update_recommendations_buganizer_level.txt")
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    prompt = prompt.format(
        human_verified_urls="\n".join(human_verified_urls),
        buganizer_issue_content=buganizer_issue_content,
        docs_content=docs_content_str
    )

    print("\nSending request to Gemini...")
    response = model.generate_content(
        [prompt],
        generation_config={
            "max_output_tokens": 58000,
            "temperature": TEMPERATURE,
            "top_p": 0.95,
            "response_mime_type": "application/json",
        },
    )

    print("\n--- Gemini Response ---")
    parsed_response = None
    try:
        # Extract what looks like a JSON object from the response text.
        # This is more robust than just finding the first '{' and last '}'
        match = re.search(r'''```json\s*(\{.*?\})\s*```''', response.text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # Fallback for cases where the JSON is not in a code block
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_text = response.text[json_start:json_end]
            else:
                json_text = None

        if json_text:
            try:
                parsed_response = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Initial JSON decode failed: {e}")
                # Attempt to fix common JSON errors (e.g., trailing commas)
                # This is a simple fix, more complex issues might still fail
                json_text_fixed = re.sub(r',(\s*[\}\]])', r'\1', json_text)
                try:
                    parsed_response = json.loads(json_text_fixed)
                    print("Successfully parsed JSON after fixing trailing commas.")
                except json.JSONDecodeError as e2:
                    print(f"JSON decode failed even after attempting to fix: {e2}")
                    print("Problematic JSON text:")
                    print(json_text)
                    return
        else:
            print("No JSON object found in the response.")
            print("Raw response:")
            print(response.text)
            return

    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        print("Raw response:")
        print(response.text)
        return

    if not parsed_response:
        print("Could not parse Gemini response. Raw response:")
        print(response.text)
        return
        
    all_results = []
    if parsed_response and "updates" in parsed_response:
        bug_summary = parsed_response.get("bug_summary", "N/A")
        print(f"\nGemini Bug Summary: {bug_summary}")
        
        updates = parsed_response.get("updates", {})
        if not updates:
            print("Gemini determined no documents needed updates for this issue.")
        
        for url, update_details in updates.items():
            normalized_url = normalize_url(url)
            if normalized_url in original_docs:
                gemini_reasoning = update_details.get('gemini_reasoning', 'N/A')
                print(f"  -> Recommendation to update: {normalized_url}")
                print(f"     Reason: {gemini_reasoning}")
                all_results.append({
                    "buganizer_id": ", ".join(buganizer_ids),
                    "url": normalized_url,
                    "bug_summary": bug_summary,
                    "gemini_reasoning": gemini_reasoning,
                    "original_text": original_docs[normalized_url],
                    "updated_text": update_details.get('updated_content'),
                })
    
    if not all_results:
        print("\nNo recommendations were generated. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RECOMMENDATIONS_CSV_FILE, mode='a', header=not os.path.exists(RECOMMENDATIONS_CSV_FILE), index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSuccessfully saved {len(results_df)} recommendations to '{RECOMMENDATIONS_CSV_FILE}'")
    
    generate_diff_and_content_files(results_df, RECOMMENDATIONS_DIR)


if __name__ == '__main__':
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
    url_cache = load_url_content_cache(URL_CONTENT_CACHE_CSV)
    bq_client = get_bq_client(PROJECT_ID)
    if bq_client:
        buganizer_ids_to_process = TEST_BUGANIZER_IDS
        if not buganizer_ids_to_process:
            print("TEST_BUGANIZER_IDS is empty. Fetching all Buganizer IDs from BigQuery.")
            buganizer_ids_to_process = get_all_buganizer_ids_from_bq(
                client=bq_client,
                project_id=PROJECT_ID,
                dataset_id=DATASET_NAME,
                pass_number=PASS_NUMBER
            )

        if not buganizer_ids_to_process:
            print("No Buganizer IDs to process.")
        else:
            for bug_id in buganizer_ids_to_process:
                print(f"--- Solving Bug ID: {bug_id} ---")
                
                # Process each bug id individually
                current_bug_id_list = [bug_id]

                human_urls = get_human_verified_urls(current_bug_id_list, BUGANIZER_ISSUES_CSV)
                human_urls = [normalize_url(url) for url in human_urls]
                print(f"Found {len(human_urls)} human-verified URLs for bug {bug_id}.")

                bq_urls = get_urls_from_buganizer_ids(
                    client=bq_client,
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_NAME,
                    pass_number=PASS_NUMBER,
                    buganizer_ids=current_bug_id_list
                )
                bq_urls = [normalize_url(url) for url in bq_urls]

                all_urls = sorted(list(set(human_urls + bq_urls)))
                
                original_docs = {}

                if all_urls:
                    get_recommendations(
                        buganizer_ids=current_bug_id_list,
                        urls=all_urls,
                        human_verified_urls=human_urls,
                        url_content_cache=url_cache,
                        original_docs=original_docs,
                        buganizer_issues_csv=BUGANIZER_ISSUES_CSV
                    )
                    print(f"--- Successfully solved Bug ID: {bug_id} ---")
                else:
                    print(f"No relevant URLs found for Bug ID: {bug_id}.")
            
            # Generate the index.html file after processing all bugs
            print("\nGenerating the main dashboard...")
            generate_index_html(RECOMMENDATIONS_DIR)