# activate venv
# source /home/user/devrel-kg/venv/bin/activate

# python /home/varunsankuri/devrel-kg/extract_graph_data_from_docs.py --product gemini-cli


import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
import google.generativeai as genai
import argparse # Added for command-line arguments
from config import GENERATION_MODEL, PROJECT_ID, REGION , GOOGLE_API_KEY,TEMPERATURE

product = "gemini-cli"  #either change the product name here or in the command line

# --- New Functions for Graph Data Extraction ---

def _read_graph_prompt_template(filename="extract_graph_data.txt", base_path="prompts"):
    """Reads the graph extraction prompt template."""
    # Construct path relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, base_path, filename)
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading prompt file {filepath}: {e}")
        return None

def _extract_single_text_graph_with_gemini(url: str, text_content: str, model_name: str, prompt_template: str, product_name: str):
    """
    Helper function to call Gemini for graph extraction for a single text.
    """
    if not url or not text_content or not prompt_template:
        return None

    # Limit content size to avoid exceeding model token limits.
    # Adjust max_chars based on the model and typical content length.
    # The prompt itself also consumes tokens.
    max_chars = 1000000 # Example limit, adjust as needed for GENERATION_MODEL
    if len(text_content) > max_chars:
        print(f"Warning: Content length ({len(text_content)} chars) exceeds max_chars ({max_chars}). Truncating.")
        text_content = text_content[:max_chars]
    
    prompt = prompt_template.format(url=url, text=text_content, product_name=product_name)

    try:
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            temperature=TEMPERATURE
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config)

        graph_data_str = response.text.strip()

        # Clean up potential markdown formatting (e.g., ```json ... ```)
        if graph_data_str.startswith("```json"):
            graph_data_str = graph_data_str[7:].strip()
        if graph_data_str.endswith("```"):
            graph_data_str = graph_data_str[:-3].strip()

        return json.loads(graph_data_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Gemini response text was: '{response.text if hasattr(response, 'text') else 'N/A'}'")
        return None
    except Exception as e:
        print(f"Error extracting graph data using Gemini: {e}")
        if hasattr(response, 'text'):
            print(f"Gemini response text was: '{response.text}'")
        return None

def extract_graph_data_from_dataframe(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """
    Takes a DataFrame, extracts graph data from the 'text' column for each row
    using Gemini, and adds it as a new 'graph_data' column.

    Args:
        df (pd.DataFrame): DataFrame containing 'page_url' and 'text' columns.
        product_name (str): The name of the product to be used in the prompt.

    Returns:
        pd.DataFrame: DataFrame with an added 'graph_data' column.
    """
    if 'text' not in df.columns or 'page_url' not in df.columns:
        print("Error: 'text' and/or 'page_url' column not found in DataFrame.")
        df['graph_data'] = None
        return df

    # Configure Gemini using the API Key from config
    if not GOOGLE_API_KEY:
        print("Fatal: GOOGLE_API_KEY not found in environment/config.py.")
        print("Ensure it is set in your .env file and loaded by config.py.")
        print("Cannot proceed with graph extraction.")
        df['graph_data'] = None
        return df

    try:
        print(f"Configuring Gemini with API Key. (Project context for API enablement: {PROJECT_ID})")
        genai.configure(api_key=GOOGLE_API_KEY)

        # Attempt to get the model to verify configuration and access.
        model_to_check = genai.get_model(GENERATION_MODEL)
        if not model_to_check:
            print(f"Fatal: Model '{GENERATION_MODEL}' not found or accessible after configuration. Cannot proceed.")
            print(f"Ensure the model name is correct and the API key has permissions for it.")
            df['graph_data'] = None
            return df
        print(f"Successfully configured Gemini and verified access to model '{GENERATION_MODEL}'.")

    except Exception as e:
        print(f"Fatal: Error configuring or verifying Gemini: {e}. Cannot proceed with graph extraction.")
        df['graph_data'] = None # Add empty column
        return df

    prompt_template_content = _read_graph_prompt_template()
    if not prompt_template_content:
        print("Fatal: Could not load graph extraction prompt. Cannot proceed.")
        df['graph_data'] = None # Add empty column
        return df

    graph_results = []
    total_rows = len(df)
    print(f"\nStarting graph data extraction for {total_rows} text entries using model '{GENERATION_MODEL}'...")

    # Iterate using df.index to ensure correct alignment when updating later
    for idx in df.index:
        page_text = df.loc[idx, 'text']
        page_url = df.loc[idx, 'page_url']
        # Using original DataFrame index for progress reporting if df is a slice
        original_index_for_reporting = idx + 1 # Assuming 0-based index from original df if it was sliced
        print(f"  Processing text for original index {original_index_for_reporting}/{len(df)} (current batch)...")


        if pd.isna(page_text) or not str(page_text).strip():
            print(f"    Skipping original index {original_index_for_reporting} due to empty or NaN text.")
            graph_results.append(None)
            continue
        
        current_product_name = product_name if product_name else df.loc[idx, 'product']

        extracted_data = _extract_single_text_graph_with_gemini(
            str(page_url),
            str(page_text),
            GENERATION_MODEL,
            prompt_template_content,
            current_product_name
        )
        graph_results.append(extracted_data)

        if extracted_data:
            nodes_count = len(extracted_data.get('nodes', []))
            rels_count = len(extracted_data.get('relationships', []))
            print(f"    Successfully extracted for original index {original_index_for_reporting}: {nodes_count} nodes, {rels_count} relationships.")
        else:
            print(f"    Failed to extract graph data for original index {original_index_for_reporting}.")

    # Assign results back to the correct slice of the DataFrame
    # This assumes 'graph_results' has the same length and order as 'df.index'
    df.loc[df.index, 'graph_data'] = graph_results
    print("Graph data extraction process complete for this batch.")
    return df


if __name__ == "__main__":
    # Define a product variable in the code.
    # This can be overridden by a command-line argument.


    # Input CSV now directly points to the file with existing text content
    INPUT_CSV_WITH_CONTENT = '/home/varunsankuri/devrel-kg/files/url_list_with_content.csv'
    # Output CSV for the DataFrame that will include graph data
    OUTPUT_CSV_GRAPH = '/home/varunsankuri/devrel-kg/files/url_list_with_graph_data.csv'

    parser = argparse.ArgumentParser(description="Extract graph data for a specific product.")
    parser.add_argument(
        "--product",
        type=str,
        default=product,
        help=f"The product to filter by (e.g., 'gemini-cli'). Defaults to '{product}'. Use 'NULL' to process all rows."
    )
    args = parser.parse_args()

    Product = args.product
    
    try:
        content_df = pd.read_csv(INPUT_CSV_WITH_CONTENT)
        print(f"Successfully loaded '{INPUT_CSV_WITH_CONTENT}'. Shape: {content_df.shape}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{INPUT_CSV_WITH_CONTENT}'. Cannot proceed.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading '{INPUT_CSV_WITH_CONTENT}': {e}")
        exit(1)

    if 'product' not in content_df.columns and Product != "NULL":
        print(f"Error: 'product' column not found in '{INPUT_CSV_WITH_CONTENT}'. Cannot filter by product.")
        exit(1)

    if Product == "NULL":
        print("--- Starting Graph Data Extraction for all products ---")
        df_to_process = content_df.copy()
    else:
        print(f"--- Starting Graph Data Extraction for product: {Product} ---")
        df_to_process = content_df[content_df['product'] == Product].copy()

    if df_to_process.empty:
        if Product != "NULL":
            print(f"No rows found for product '{Product}'. Nothing to process.")
        else:
            print("Input CSV is empty. Nothing to process.")
        exit(0)

    if Product != "NULL":
        print(f"Found {len(df_to_process)} rows for product '{Product}'.")
    else:
        print(f"Processing all {len(df_to_process)} rows.")

    if 'graph_data' not in df_to_process.columns:
        df_to_process['graph_data'] = pd.NA

    product_name_for_prompt = Product if Product != "NULL" else ""
    extract_graph_data_from_dataframe(df_to_process, product_name_for_prompt)

    try:
        file_exists = os.path.exists(OUTPUT_CSV_GRAPH)
        df_to_process.to_csv(OUTPUT_CSV_GRAPH, mode='a', header=not file_exists, index=False)
        print(f"Successfully appended processed data to '{OUTPUT_CSV_GRAPH}'")
    except Exception as e:
        print(f"Error saving final DataFrame: {e}")

    # Final check and print sample
    if os.path.exists(OUTPUT_CSV_GRAPH):
        try:
            final_df_check = pd.read_csv(OUTPUT_CSV_GRAPH)
            if not final_df_check.empty:
                print(f"\n--- Final Output Sample (from '{OUTPUT_CSV_GRAPH}') ---")
                preview_cols = ['page_url', 'title', 'product']
                if 'text' in final_df_check.columns: preview_cols.append('text')
                if 'graph_data' in final_df_check.columns: preview_cols.append('graph_data')

                valid_preview_cols = [col for col in preview_cols if col in final_df_check.columns]
                if Product != "NULL":
                    print(final_df_check[final_df_check['product'] == Product].tail(2).to_string())
                else:
                    print(final_df_check.tail(2).to_string())
            else:
                print(f"Final output file '{OUTPUT_CSV_GRAPH}' is empty.")
        except Exception as e:
            print(f"Error reading or previewing final output file '{OUTPUT_CSV_GRAPH}': {e}")

    print("--- Graph Data Extraction Complete ---")