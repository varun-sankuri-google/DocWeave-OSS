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
import html
import difflib
from parse_url import clean_url, normalize_url

# Assuming config.py exists and contains the necessary variables
try:
    from config import GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY, TEMPERATURE, TOP_P
except ImportError:
    print("Error: config.py not found or is missing required variables.")
    print("Please create it with GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY, and TEMPERATURE.")
    exit(1)

# --- Configuration ---
DATASET_NAME = "bq_graph_db"
PASS_NUMBER = 1  # The pass number of the graph to query
URL_CONTENT_CACHE_CSV = os.path.join(os.path.dirname(__file__), "files", "url_list_with_content.csv")
RECOMMENDATIONS_DIR = "recommendations_from_tickets"
RECOMMENDATIONS_CSV_FILE = os.path.join(RECOMMENDATIONS_DIR, "reddit_recommendations.csv")
REDDIT_CONTENT_CSV = os.path.join(os.path.dirname(__file__), "support_tickets", "reddit_data.csv")
REDDIT_URLS_TO_PROCESS = [
   
#    first set
    # "https://www.reddit.com/r/googlecloud/comments/1mwz3ot",
    # "https://www.reddit.com/r/bigquery/comments/1lyckf0/comment/n4pcunp",
    # "https://www.reddit.com/r/bigquery/comments/1lqnln5/comment/n4d27rk",
    # "https://www.reddit.com/r/bigquery/comments/1mksbg8/comment/n85lqp3",
    # "https://www.reddit.com/r/googlecloud/comments/1m29vob/comment/n3oya0t",

    # second set
    # "https://www.reddit.com/r/dataengineering/comments/1ly3qji/comment/n2w773m",
    # "https://www.reddit.com/r/bigquery/comments/1luhtdh/comment/n20l97x",
    # "https://www.reddit.com/r/googlecloud/comments/1m5c7zv/comment/n4osxkb",
    # "https://www.reddit.com/r/agentdevelopmentkit/comments/1m9zd1w",
    # "https://www.reddit.com/r/bigquery/comments/1m2c86a/comment/n3y2lvx",

    # third set
    "https://www.reddit.com/r/bigquery/comments/1lvbsn9/comment/n3ihc9o",
    "https://www.reddit.com/r/bigquery/comments/1lumkej/comment/n22q89u",
    "https://www.reddit.com/r/bigquery/comments/1m0qsxp/comment/n3d74ss",
    "https://www.reddit.com/r/dataengineering/comments/1miaxe9/comment/n77ezzz",
    "https://www.reddit.com/r/bigquery/comments/1lqwm4h/comment/n2axwas",

    ] # Add specific Reddit URLs here to process only those

def sanitize_for_filename(text: str) -> str:
    """Sanitizes a string to be a valid filename component."""
    # Remove protocol if it's a URL
    sanitized = re.sub(r'^https?://', '', text)
    # Replace invalid filename characters with underscores
    sanitized = re.sub(r'[\\/*?"<>|]', '_', sanitized)
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
        .summary-box { background-color: #eef; border-left: 5px solid #66c; padding: 1em; margin-bottom: 1em; }
    """
    
    for _, row in results_df.iterrows():
        reddit_url = str(row['reddit_url'])
        url_to_update = row['url_to_update']
        reddit_summary = row.get('reddit_summary', 'Not available.')
        gemini_reasoning = row.get('gemini_reasoning', 'Not available.')
        base_filename = f"reddit_{sanitize_for_filename(reddit_url)}_{sanitize_for_filename(url_to_update)}"

        # Generate content files
        original_filename = os.path.join(content_dir, f"{base_filename}_original.html")
        updated_filename = os.path.join(content_dir, f"{base_filename}_updated.html")
        
        with open(original_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Original: {url_to_update}", row['original_text']))
        with open(updated_filename, 'w', encoding='utf-8') as f:
            f.write(create_html_doc(f"Updated: {url_to_update}", row['updated_text']))

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
<html><head><title>Diff for: {html.escape(url_to_update)}</title>
{style_block}
</head><body>
<h1>Diff for: <a href='../content_files/{os.path.basename(original_filename)}' target='_blank'>Original</a> | <a href='../content_files/{os.path.basename(updated_filename)}' target='_blank'>Updated</a> | <a href='{html.escape(url_to_update)}' target='_blank'>{html.escape(url_to_update)}</a> (Reddit URL: <a href='{html.escape(reddit_url)}' target='_blank'>{html.escape(reddit_url)}</a>)</h1>
<div class="summary-box">
    <h2>Reddit Summary</h2>
    <p>{html.escape(reddit_summary)}</p>
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

def load_reddit_content_from_csv(file_path: str) -> dict[str, str]:
    """Loads Reddit content from a CSV file into a dictionary."""
    print(f"Loading Reddit content from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        return pd.Series(df.text.values, index=df.page_url).to_dict()
    except FileNotFoundError:
        print(f"Error: Reddit content file not found at {file_path}.")
        return {}

def get_url_content(url: str) -> str | None:
    """Fetches and parses the text content of a URL."""
    normalized_url = normalize_url(url)
    try:
        req = urllib.request.Request(normalized_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                html_content = response.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator='\n', strip=True)
            else:
                print(f"Failed to fetch {normalized_url}: Status code {response.getcode()}")
                return None
    except urllib.error.URLError as e:
        print(f"Error fetching {normalized_url}: {e.reason}")
        return None

def get_reddit_data_from_bq(client: bigquery.Client, project_id: str, dataset_id: str, pass_number: int) -> list[str]:
    """
    Queries the BigQuery nodes table to get all unique Reddit URLs.
    """
    table_name = f"{project_id}.{dataset_id}.nodes_pass{pass_number}"
    query = f'''
        SELECT DISTINCT r_url
        FROM `{table_name}`,
             UNNEST(reddit_urls) AS r_url
        WHERE reddit_count > 0
    '''
    try:
        print("Querying BigQuery for all Reddit URLs...")
        query_job = client.query(query)
        rows = query_job.result()
        reddit_urls = [row.r_url for row in rows]
        print(f"Found {len(reddit_urls)} unique Reddit threads.")
        return reddit_urls
    except Exception as e:
        print(f"An error occurred while querying BigQuery for Reddit data: {e}")
        return []

def get_recommendations_from_reddit(reddit_url: str, reddit_content: str, level1_docs: list[str], level2_docs: list[str], url_content_cache: dict[str, str]):
    """
    Analyzes a Reddit thread and associated documentation to provide recommendations.
    """
    print("The following URLs are being sent to Gemini:")
    print("Level 1 Docs:")
    for url in level1_docs:
        print(url)
    print("Level 2 Docs:")
    for url in level2_docs:
        print(url)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GENERATION_MODEL)

    def get_docs_content(urls):
        docs_content_list = []
        original_docs = {}
        for url in urls:
            normalized_url = normalize_url(url)
            content = url_content_cache.get(normalized_url)

            if not content:
                print(f"  Content for {normalized_url} not in cache. Fetching from web...")
                content = get_url_content(normalized_url)

            if content:
                docs_content_list.append(f"\n\n---\nDOCUMENT URL: {normalized_url}\n---\n{content}")
                original_docs[normalized_url] = content
            else:
                print(f"  Could not retrieve content for {normalized_url}")
        return "\n".join(docs_content_list), original_docs

    level1_docs_content_str, original_docs1 = get_docs_content(level1_docs)
    level2_docs_content_str, original_docs2 = get_docs_content(level2_docs)

    original_docs = {**original_docs1, **original_docs2}

    if not level1_docs_content_str and not level2_docs_content_str:
        print("Could not retrieve content for any of the associated URLs. Aborting.")
        return

    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "doc_update_recommendations_reddit_level.txt")
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    prompt = prompt.format(
        reddit_thread_content=reddit_content,
        level1_docs_content=level1_docs_content_str,
        level2_docs_content=level2_docs_content_str
    )

    print("\nSending request to Gemini...")
    response = model.generate_content(
        [prompt],
        generation_config={
            "max_output_tokens": 58000,
            "temperature": TEMPERATURE,
            # "top_p": TOP_P,
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
                if not json_text_fixed.endswith('}'):
                    json_text_fixed += '}'
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

    if not parsed_response or "updates" not in parsed_response:
        print("No 'updates' field in Gemini response.")
        return

    all_results = []
    reddit_summary = parsed_response.get("reddit_summary", "N/A")
    print(f"\nGemini Reddit Summary: {reddit_summary}")

    for url, update_details in parsed_response["updates"].items():
        normalized_url = normalize_url(url)
        if normalized_url in original_docs:
            gemini_reasoning = update_details.get('gemini_reasoning', 'N/A')
            print(f"  -> Recommendation to update: {normalized_url}")
            print(f"     Reason: {gemini_reasoning}")
            all_results.append({
                "reddit_url": reddit_url,
                "url_to_update": normalized_url,
                "reddit_summary": reddit_summary,
                "gemini_reasoning": gemini_reasoning,
                "original_text": original_docs[normalized_url],
                "updated_text": update_details.get('updated_content'),
            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RECOMMENDATIONS_CSV_FILE, mode='a', header=not os.path.exists(RECOMMENDATIONS_CSV_FILE), index=False, quoting=csv.QUOTE_ALL)
        print(f"\nSuccessfully saved {len(results_df)} recommendations to '{RECOMMENDATIONS_CSV_FILE}'")
        generate_diff_and_content_files(results_df, RECOMMENDATIONS_DIR)


def get_related_urls_from_graph(client: bigquery.Client, project_id: str, dataset_id: str, pass_number: int, reddit_url: str):
    """
    Queries the BigQuery graph to find Level 1 and ranked Level 2 URLs for a given Reddit URL.
    """
    nodes_table_name = f"{project_id}.{dataset_id}.nodes_pass{pass_number}"
    relationships_table_name = f"{project_id}.{dataset_id}.relationships_pass{pass_number}"

    query = f"""
    WITH Level1Entities AS (
        -- Find all entities (nodes) directly associated with the input Reddit URL
        SELECT DISTINCT
            entity_name
        FROM
            `{nodes_table_name}`
        WHERE
            '{reddit_url}' IN UNNEST(reddit_urls)
    ),
    Level1URLs AS (
        -- Get all URLs from the Level 1 entities
        SELECT DISTINCT
            url
        FROM
            `{nodes_table_name}`,
            UNNEST(source_document_url) as url
        WHERE
            entity_name IN (SELECT entity_name FROM Level1Entities)
    ),
    Level2Entities AS (
        -- Find all entities connected to our Level 1 entities
        SELECT DISTINCT
            r.target_entity_name AS entity_name,
            COUNT(r.source_entity_name) AS relevance_score
        FROM
            `{relationships_table_name}` r
        WHERE
            r.source_entity_name IN (SELECT entity_name FROM Level1Entities)
        GROUP BY
            r.target_entity_name
    ),
    RankedLevel2URLs AS (
        -- Get all URLs from the Level 2 entities and rank them by relevance
        SELECT
            url,
            l2.relevance_score
        FROM
            `{nodes_table_name}` n,
            UNNEST(source_document_url) as url
        JOIN
            Level2Entities l2 ON n.entity_name = l2.entity_name
        WHERE
            url NOT IN (SELECT url FROM Level1URLs)
    )
    -- Final selection
    SELECT
        (SELECT ARRAY_AGG(url) FROM Level1URLs) AS level1_urls,
        (SELECT ARRAY_AGG(url ORDER BY relevance_score DESC, url LIMIT 50) FROM RankedLevel2URLs) AS level2_urls;
    """

    try:
        print(f"Querying graph for Reddit URL: {reddit_url} with new logic...")
        query_job = client.query(query)
        rows = list(query_job.result())

        if not rows:
            print("No results from graph query.")
            return [], []

        row = rows[0]
        level1_urls = row.level1_urls if row.level1_urls is not None else []
        level2_urls = row.level2_urls if row.level2_urls is not None else []

        print(f"Found {len(level1_urls)} Level 1 URL(s).")
        print(f"Found {len(level2_urls)} relevant Level 2 URL(s) (top 50).")

        return level1_urls, level2_urls

    except Exception as e:
        print(f"An error occurred while querying the graph: {e}")
        return [], []

if __name__ == '__main__':
    os.makedirs(RECOMMENDATIONS_DIR, exist_ok=True)
    url_cache = load_url_content_cache(URL_CONTENT_CACHE_CSV)
    reddit_content_cache = load_reddit_content_from_csv(REDDIT_CONTENT_CSV)
    bq_client = get_bq_client(PROJECT_ID)

    if bq_client:
        reddit_urls = get_reddit_data_from_bq(
            client=bq_client,
            project_id=PROJECT_ID,
            dataset_id=DATASET_NAME,
            pass_number=PASS_NUMBER
        )

        if REDDIT_URLS_TO_PROCESS:
            reddit_urls = [url for url in reddit_urls if url in REDDIT_URLS_TO_PROCESS]

        if not reddit_urls:
            print("No Reddit data found in BigQuery or the specified URLs were not found.")
        else:
            for reddit_url in reddit_urls:
                print(f"--- Processing Reddit URL: {reddit_url} ---")

                level1_urls, level2_urls = get_related_urls_from_graph(
                    client=bq_client,
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_NAME,
                    pass_number=PASS_NUMBER,
                    reddit_url=reddit_url
                )

                reddit_content = reddit_content_cache.get(normalize_url(reddit_url))

                if reddit_content:
                    get_recommendations_from_reddit(
                        reddit_url=reddit_url,
                        reddit_content=reddit_content,
                        level1_docs=level1_urls,
                        level2_docs=level2_urls,
                        url_content_cache=url_cache
                    )
                    print(f"--- Successfully processed Reddit URL: {reddit_url} ---")
                else:
                    print(f"Could not retrieve content for Reddit URL: {reddit_url}")