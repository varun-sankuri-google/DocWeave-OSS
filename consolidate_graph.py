import pandas as pd
import json
import os
import google.generativeai as genai
import ast # For safely evaluating string literals like dicts
import argparse
import time
import numpy as np # For checking NaN and array_split

# Assuming your config.py is in the same directory or accessible via PYTHONPATH
# and contains GENERATION_MODEL, GOOGLE_API_KEY, PROJECT_ID
from config import GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY,TEMPERATURE

# --- Configuration ---
CSV_FILE_PATH = '/home/varunsankuri/devrel-kg/files/url_list_with_graph_data_bq.csv'
GRAPH_DATA_COLUMN = 'graph_data'
PROMPT_FILENAME_PASS_1 = "consolidate_graph_prompt.txt"       # For the first pass (chunk-level)
PROMPT_FILENAME_PASS_2 = "final_consolidate_graph_prompt.txt" # For the second pass (inter-chunk)
PROMPT_FILENAME_PASS_3 = "final_consolidate_graph_prompt.txt" # For the third pass (refinement)
PROMPT_FILENAME_PASS_4 = "final_consolidate_graph_prompt.txt" # For the fourth pass (final polish)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "json_output")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists
PROMPT_FILE_PATH_PASS_1 = os.path.join(SCRIPT_DIR, "prompts", PROMPT_FILENAME_PASS_1)
PROMPT_FILE_PATH_PASS_2 = os.path.join(SCRIPT_DIR, "prompts", PROMPT_FILENAME_PASS_2)
PROMPT_FILE_PATH_PASS_3 = os.path.join(SCRIPT_DIR, "prompts", PROMPT_FILENAME_PASS_3)
PROMPT_FILE_PATH_PASS_4 = os.path.join(SCRIPT_DIR, "prompts", PROMPT_FILENAME_PASS_4)

# Output filenames for each pass
OUTPUT_FILENAME_PASS_1 = os.path.join(OUTPUT_DIR, "consolidated_graph_pass1.json")
OUTPUT_FILENAME_PASS_2 = os.path.join(OUTPUT_DIR, "consolidated_graph_pass2.json")
OUTPUT_FILENAME_PASS_3 = os.path.join(OUTPUT_DIR, "consolidated_graph_pass3.json")
FINAL_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "consolidated_graph_pass4.json")

# Configuration for multi-stage consolidation
N_CHUNKS_PASS_1 = 40  # Number of chunks for the initial consolidation
N_CHUNKS_PASS_2 = 30  # Number of chunks for the second consolidation pass
N_CHUNKS_PASS_3 = 20  # Number of chunks for the third consolidation pass
N_CHUNKS_PASS_4 = 15   # Number of chunks for the final consolidation pass
MAX_CHARS_PER_CHUNK_PASS_1 = 2_000_000  # Character limit for pass 1 chunks
MAX_CHARS_PER_CHUNK_PASS_2 = 1_500_000  # Character limit for pass 2 chunks
MAX_CHARS_PER_CHUNK_PASS_3 = 1_000_000  # Character limit for pass 3 chunks
MAX_CHARS_PER_CHUNK_PASS_4 = 1_000_000  # Character limit for pass 4 chunks
MAX_RETRIES_PER_CHUNK = 2 # 1 initial attempt + 2 retries
MAX_OUTPUT_TOKENS_GEMINI = 58000 # Set a generous output limit. The main defense against MAX_TOKENS is a good prompt.


def read_prompt_template(filepath):
    """Reads the graph consolidation prompt template."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading prompt file {filepath}: {e}")
        return None

def load_graph_from_json(filepath: str) -> dict:
    """Loads a graph dictionary from a JSON file."""
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph from '{filepath}'")
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data or 'relationships' not in graph_data:
            print(f"Warning: JSON file '{filepath}' is not in the expected graph format.")
            return None
        return graph_data
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{filepath}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}")
        return None
    except Exception as e:
        print(f"Error loading JSON file '{filepath}': {e}")
        return None

def aggregate_graph_data(df: pd.DataFrame, column: str) -> dict:
    """
    Aggregates graph data (nodes and relationships) from a DataFrame column
    into a single graph dictionary.
    """
    all_nodes = []   
    all_relationships = []
    processed_count = 0

    print(f"\nAggregating graph data from column '{column}'...")

    for index, row_value in df[column].items():
        if pd.notna(row_value) and isinstance(row_value, str) and row_value.strip():
            try:
                # Use ast.literal_eval to parse string representation of Python dict
                graph_dict = ast.literal_eval(row_value)
                if isinstance(graph_dict, dict):
                    nodes = graph_dict.get('nodes', [])
                    relationships = graph_dict.get('relationships', [])

                    if isinstance(nodes, list) and isinstance(relationships, list):
                        all_nodes.extend(nodes)
                        all_relationships.extend(relationships)
                        processed_count += 1
                    else:
                         print(f"Warning: Row {index} graph_data (after eval) does not contain expected 'nodes' or 'relationships' lists. Skipping. Content: {str(row_value)[:100]}...")
                else:
                    print(f"Warning: Row {index} graph_data (after eval) is not a dictionary. Skipping. Content: {str(row_value)[:100]}...")

            except (ValueError, SyntaxError, TypeError) as e_eval:
                print(f"Warning: Row {index} contains data in '{column}' that could not be evaluated by ast.literal_eval. Error: {e_eval}. Skipping. Content: {str(row_value)[:100]}...")
            except Exception as e:
                 print(f"Warning: Unexpected error processing row {index} graph_data: {e}. Skipping. Content: {str(row_value)[:100]}...")


    print(f"Aggregated data from {processed_count} valid graph entries.")
    print(f"Total nodes collected: {len(all_nodes)}")
    print(f"Total relationships collected: {len(all_relationships)}")

    return {"nodes": all_nodes, "relationships": all_relationships}


def _format_error_report(response_obj):
    """Helper to create a detailed error string from a Gemini response object."""
    if not response_obj:
        return "No response object available."

    report_parts = []
    if hasattr(response_obj, 'prompt_feedback') and response_obj.prompt_feedback:
        report_parts.append(f"  Prompt Feedback: {response_obj.prompt_feedback}")
    if hasattr(response_obj, 'candidates') and response_obj.candidates:
        for i, candidate in enumerate(response_obj.candidates):
            report_parts.append(f"  Candidate {i} Finish Reason: {candidate.finish_reason}")
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                report_parts.append(f"  Candidate {i} Safety Ratings: {candidate.safety_ratings}")
    if hasattr(response_obj, 'usage_metadata'):
        report_parts.append(f"  Usage Metadata: {response_obj.usage_metadata}")
    if hasattr(response_obj, 'text') and response_obj.text:
        response_text_preview = response_obj.text
        if len(response_text_preview) > 500:
            response_text_preview = response_text_preview[:500] + "... (truncated in log)"
        report_parts.append(f"  Response Text (preview): '{response_text_preview}'")
    
    return "\n".join(report_parts) if report_parts else "No detailed error information found in response object."

def consolidate_graph_with_gemini(combined_graph: dict, model_name: str, prompt_template: str, max_chars_limit: int):
    """
    Calls Gemini to consolidate the combined graph data.
    """
    if not combined_graph or not prompt_template:
        return None, 0, "Error: Combined graph data or prompt template was missing."

    initial_node_count = len(combined_graph.get('nodes', []))
    nodes_prepared_for_gemini = initial_node_count # Start with the full count, will be updated if truncated

    # Convert the combined graph dictionary into a JSON string for the prompt
    # Use compact JSON to save tokens
    combined_graph_json_string = json.dumps(combined_graph, separators=(',', ':'))

    print(f"Total length of combined graph data string for prompt: {len(combined_graph_json_string)} characters.")
    
    # Apply character limit to the combined data
    if len(combined_graph_json_string) > max_chars_limit:
        print(f"Warning: Combined graph data length ({len(combined_graph_json_string)} chars) "
              f"exceeds max_chars_limit ({max_chars_limit}). Truncating input to Gemini.")
        
        # Original code's truncation logic is left here for reference but should ideally be avoided
        # by better chunking. If chunks are still too large after improved chunking, consider
        # further reducing N_CHUNKS_PASS_1 or MAX_CHARS_PER_CHUNK_PASS_1 or using a different model.
        total_elements = len(combined_graph.get('nodes', [])) + len(combined_graph.get('relationships', []))
        if total_elements > 0:
            approx_chars_per_element = len(combined_graph_json_string) / total_elements
            max_elements_to_keep = int(max_chars_limit / approx_chars_per_element)
            # Distribute the limit between nodes and relationships based on their original proportion
            node_proportion = len(combined_graph.get('nodes', [])) / total_elements
            rel_proportion = 1 - node_proportion

            truncated_nodes_count = int(max_elements_to_keep * node_proportion)
            truncated_relationships_count = max_elements_to_keep - truncated_nodes_count # Ensure total is max_elements_to_keep

            truncated_nodes = combined_graph.get('nodes', [])[:truncated_nodes_count]
            truncated_relationships = combined_graph.get('relationships', [])[:truncated_relationships_count]

            truncated_graph = {"nodes": truncated_nodes, "relationships": truncated_relationships}
            combined_graph_json_string = json.dumps(truncated_graph, separators=(',', ':'))
            nodes_prepared_for_gemini = len(truncated_nodes) # Update with actual count after truncation

            print(f"Truncated combined graph data length: {len(combined_graph_json_string)} characters.")
            print(f"Kept {len(truncated_nodes)} nodes and {len(truncated_relationships)} relationships.")
            print("This truncation might lead to an incomplete or inaccurate consolidation.")
        else:
             print("Warning: No nodes or relationships to truncate, but combined string is too large? This is unexpected.")


    prompt = prompt_template.format(graph_data_json=combined_graph_json_string)

    try:
        print(f"\nSending prompt (length: {len(prompt)} chars) to Gemini model '{model_name}'...")
        model = genai.GenerativeModel(model_name)

        # Adjust generation_config for JSON output. Using current SDK syntax.
        generation_config = genai.GenerationConfig(
             temperature=TEMPERATURE, # Keep low for factual consolidation
            response_mime_type="application/json",
            max_output_tokens=MAX_OUTPUT_TOKENS_GEMINI,
        )

        response = model.generate_content(prompt, generation_config=generation_config)

        # Check if the response text is empty, which can happen for various reasons (e.g. safety filters)
        if not response.text or not response.candidates:
            error_report = _format_error_report(response)
            return None, nodes_prepared_for_gemini, f"Gemini API returned an empty or invalid response.\n{error_report}"

        consolidated_graph_str = response.text.strip()

        # Even with mime_type="application/json", sometimes markdown ```json can appear. Clean it up.
        if consolidated_graph_str.startswith("```json"):
            consolidated_graph_str = consolidated_graph_str[7:].strip()
        if consolidated_graph_str.endswith("```"):
            consolidated_graph_str = consolidated_graph_str[:-3].strip()

        # Parse the JSON string from Gemini
        consolidated_graph_dict = json.loads(consolidated_graph_str)

        # Basic validation of the output structure
        if not isinstance(consolidated_graph_dict, dict) or 'nodes' not in consolidated_graph_dict or 'relationships' not in consolidated_graph_dict:
             print(f"Error: Gemini response is not in the expected JSON format. Response: {consolidated_graph_str[:500]}...")
             error_report = _format_error_report(response)
             return None, nodes_prepared_for_gemini, f"Response was not in the expected JSON graph format.\n{error_report}"
        
        if hasattr(response, 'usage_metadata'):
            print(f"  Successfully processed. Usage Metadata: {response.usage_metadata}")

        return consolidated_graph_dict, nodes_prepared_for_gemini, None

    except json.JSONDecodeError as e:
        response_obj = locals().get('response')
        error_report = _format_error_report(response_obj)
        # Check if the failure was due to reaching the token limit, which causes truncated (invalid) JSON.
        if response_obj and response_obj.candidates and response_obj.candidates[0].finish_reason.value == 2: # 2 corresponds to MAX_TOKENS
            return None, nodes_prepared_for_gemini, f"MAX_TOKENS_ERROR: Gemini response was truncated, leading to JSON decode error. {e}\n{error_report}"

        return None, nodes_prepared_for_gemini, f"Error decoding JSON from Gemini response: {e}\n{error_report}"
    except Exception as e:
        error_report = _format_error_report(locals().get('response'))
        return None, nodes_prepared_for_gemini, f"An unexpected error occurred during Gemini call: {e}\n{error_report}"

def create_relationship_chunks(graph_data: dict, num_chunks: int) -> list:
    """Splits a graph into chunks based on relationships."""
    all_nodes = graph_data.get('nodes', [])
    all_relationships = graph_data.get('relationships', [])

    if not all_relationships:
        # If there are no relationships, but there are nodes, create a single chunk with all nodes.
        if all_nodes:
            return [{"nodes": all_nodes, "relationships": []}]
        return []

    # Ensure num_chunks is not greater than the number of relationships
    effective_num_chunks = min(num_chunks, len(all_relationships))
    if effective_num_chunks == 0:
        return []

    relationship_chunks_np = np.array_split(all_relationships, effective_num_chunks)
    
    graph_chunks = []
    print(f"\n--- Splitting data into {effective_num_chunks} relationship-centric chunks ---")

    node_lookup = {node['entity_name']: node for node in all_nodes if isinstance(node, dict) and 'entity_name' in node}

    for i, rel_chunk_np in enumerate(relationship_chunks_np):
        current_relationships = list(rel_chunk_np)
        current_nodes_in_chunk_set = set()
        current_nodes_in_chunk_list = []

        for rel in current_relationships:
            if not isinstance(rel, dict): continue

            for key in ['source_entity_name', 'target_entity_name']:
                node_name = rel.get(key)
                if node_name and node_name not in current_nodes_in_chunk_set:
                    node_obj = node_lookup.get(node_name)
                    if node_obj:
                        current_nodes_in_chunk_list.append(node_obj)
                        current_nodes_in_chunk_set.add(node_name)
        
        graph_chunks.append({
            "nodes": current_nodes_in_chunk_list,
            "relationships": current_relationships
        })
        print(f"Chunk {i+1} prepared: {len(current_nodes_in_chunk_list)} nodes, {len(current_relationships)} relationships.")
    
    return graph_chunks

def run_consolidation_pass(
    pass_name: str,
    input_chunks: list,
    model_name: str,
    prompt_template: str,
    max_chars_limit: int,
    max_retries: int
) -> (list, list, list):
    """Runs a consolidation pass over a list of graph chunks."""
    consolidated_graphs = []
    failed_chunk_reports = []
    failed_chunks_data = []

    print(f"\n--- Starting {pass_name} ({len(input_chunks)} chunks) ---")

    for i, chunk_graph in enumerate(input_chunks):
        print(f"\nConsolidating Chunk {i+1}/{len(input_chunks)} for {pass_name}...")
        if not chunk_graph.get('nodes') and not chunk_graph.get('relationships'):
            print(f"Chunk {i+1} is empty. Skipping.")
            consolidated_graphs.append({"nodes": [], "relationships": []})
            continue
        
        consolidated_chunk = None
        last_error_report = "No error report generated."
        for attempt in range(max_retries + 1):
            current_consolidated_chunk, nodes_sent_chunk, error_report = consolidate_graph_with_gemini(
                chunk_graph, model_name, prompt_template, max_chars_limit
            )
            last_error_report = error_report

            if current_consolidated_chunk:
                consolidated_chunk = current_consolidated_chunk
                print(f"Chunk {i+1} consolidated successfully on attempt {attempt + 1}: "
                      f"{len(consolidated_chunk.get('nodes',[]))} nodes, "
                      f"{len(consolidated_chunk.get('relationships',[]))} relationships "
                      f"(from {nodes_sent_chunk} nodes sent to Gemini).")
                break # Success, exit retry loop
            else:
                # Check for the specific non-recoverable MAX_TOKENS error
                if error_report and "MAX_TOKENS_ERROR" in error_report:
                    print(f"Attempt {attempt + 1} failed for chunk {i+1} due to MAX_TOKENS. "
                          f"This is not a transient error. Aborting retries for this chunk.")
                    break

                print(f"Attempt {attempt + 1}/{max_retries + 1} failed for chunk {i+1}.")
                if attempt < max_retries:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)

        if consolidated_chunk:
            consolidated_graphs.append(consolidated_chunk)
        else:
            # This block is reached if the loop finished without success
            consolidated_graphs.append({"nodes": [], "relationships": []})
            print(f"Failed to consolidate chunk {i+1} after all attempts. It will be excluded from the next stage.")
            report = f"--- Chunk {i+1} Failure Report ({pass_name}) ---\n{last_error_report}"
            failed_chunk_reports.append(report)
            failed_chunks_data.append({'pass_name': pass_name, 'chunk_index': i + 1, 'reason': last_error_report})

    return consolidated_graphs, failed_chunk_reports, failed_chunks_data

def combine_graphs(graph_list: list) -> dict:
    """
    Combines a list of graph dictionaries into a single graph.
    - De-duplicates nodes by 'entity_name'.
    - Merges 'source_document_url' properties into a unique list.
    - Sums 'source_document_count' for merged nodes.
    - Aggregates 'frequency' for identical relationships.
    """
    aggregated_nodes_dict = {}
    aggregated_rels_dict = {} # To aggregate relationships

    for graph in graph_list:
        if not isinstance(graph, dict): continue

        # Process nodes
        for node in graph.get("nodes", []):
            if not isinstance(node, dict): continue
            node_name = node.get("entity_name")
            if not node_name: continue

            if node_name not in aggregated_nodes_dict:
                # First time seeing this node. Initialize it.
                # Ensure properties exist and source_document_url is a list
                node_props = node.get('properties', {})
                urls = node_props.get('source_document_url', [])
                if isinstance(urls, str):
                    node_props['source_document_url'] = [urls]
                else:
                    # Ensure it's a mutable list, not a tuple from ast.literal_eval
                    node_props['source_document_url'] = list(urls) 
                
                # Ensure source_document_count exists
                if 'source_document_count' not in node_props:
                    node_props['source_document_count'] = 1
                
                node['properties'] = node_props
                aggregated_nodes_dict[node_name] = node
            else:
                # Node already exists, merge properties.
                existing_node = aggregated_nodes_dict[node_name]
                new_props = node.get('properties', {})

                # Merge source_document_url
                existing_urls = set(existing_node.get('properties', {}).get('source_document_url', []))
                new_urls = new_props.get('source_document_url', [])
                if isinstance(new_urls, str):
                    existing_urls.add(new_urls)
                else:
                    existing_urls.update(new_urls)
                existing_node.get('properties', {})['source_document_url'] = sorted(list(existing_urls))

                # Sum source_document_count
                existing_count = existing_node.get('properties', {}).get('source_document_count', 1)
                new_count = new_props.get('source_document_count', 1)
                existing_node.get('properties', {})['source_document_count'] = existing_count + new_count

        # Process relationships (existing logic seems fine for frequency)
        for rel in graph.get("relationships", []):
            if not isinstance(rel, dict): continue
            source = rel.get("source_entity_name")
            target = rel.get("target_entity_name")
            rel_type = rel.get("relationship_type")
            
            if not (source and target and rel_type): continue

            rel_key = (source, target, rel_type)
            frequency = int(rel.get("frequency", 1))

            if rel_key not in aggregated_rels_dict:
                aggregated_rels_dict[rel_key] = rel.copy()
                aggregated_rels_dict[rel_key]['frequency'] = frequency # Ensure it's an int
            else:
                # Add to the existing frequency
                aggregated_rels_dict[rel_key]['frequency'] += frequency

    return {
        "nodes": list(aggregated_nodes_dict.values()),
        "relationships": list(aggregated_rels_dict.values())
    }

def save_graph_to_json(graph_data: dict, filename: str, is_final: bool = False):
    """Saves a graph dictionary to a JSON file with pretty printing."""
    if not graph_data or (not graph_data.get('nodes') and not graph_data.get('relationships')):
        print(f"\nSkipping save for '{filename}' as there is no graph data.")
        return

    num_nodes = len(graph_data.get('nodes', []))
    num_rels = len(graph_data.get('relationships', []))
    prefix = "Final" if is_final else "Intermediate"

    print(f"\nSaving {prefix.lower()} graph with {num_nodes} nodes and {num_rels} relationships to '{filename}'...")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving graph to file '{filename}': {e}")

def write_failures_to_csv(failed_data: list, filename: str):
    """Writes a list of failure data dictionaries to a CSV file."""
    if not failed_data:
        return

    print(f"\n--- Writing Failed Chunk Reasons to '{filename}' ---")
    try:
        failed_df = pd.DataFrame(failed_data)
        failed_csv_path = os.path.join(SCRIPT_DIR, filename)
        failed_df.to_csv(failed_csv_path, index=False, encoding='utf-8')
        print(f"Saved details for {len(failed_data)} failed chunks to '{failed_csv_path}'")
    except Exception as e:
        print(f"Error writing failed chunks to CSV '{filename}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Run the graph consolidation pipeline in stages.")
    parser.add_argument(
        "--start-pass",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="The pass number to start processing from. Defaults to 1 (full run)."
    )
    parser.add_argument(
        "--end-pass",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="The pass number to end processing at. Defaults to 4 (full run)."
    )
    args = parser.parse_args()

    if args.start_pass > args.end_pass:
        print("Error: --start-pass cannot be greater than --end-pass.")
        exit(1)

    print("--- Starting Graph Data Consolidation ---")

    # Configure Gemini
    if not GOOGLE_API_KEY:
        print("Fatal: GOOGLE_API_KEY not found. Set it in your environment or config.py.")
        exit(1)
    try:
        print(f"Configuring Gemini with API Key. (Project context for API enablement: {PROJECT_ID})")
        genai.configure(api_key=GOOGLE_API_KEY)
        model_to_check = genai.get_model(GENERATION_MODEL)
        if not model_to_check:
            print(f"Fatal: Model '{GENERATION_MODEL}' not found or accessible. Cannot proceed.")
            exit(1)
        print(f"Successfully configured Gemini and verified access to model '{GENERATION_MODEL}'.")
    except Exception as e:
        print(f"Fatal: Error configuring or verifying Gemini: {e}. Cannot proceed.")
        exit(1)

    # Read prompt templates
    prompt_template_pass1_content = read_prompt_template(PROMPT_FILE_PATH_PASS_1)
    if not prompt_template_pass1_content:
        print("Fatal: Could not load first-pass consolidation prompt. Cannot proceed.")
        exit(1)
    
    prompt_template_pass2_content = read_prompt_template(PROMPT_FILE_PATH_PASS_2)
    if not prompt_template_pass2_content:
        print("Fatal: Could not load second-pass consolidation prompt. Cannot proceed.")
        exit(1)

    prompt_template_pass3_content = read_prompt_template(PROMPT_FILE_PATH_PASS_3)
    if not prompt_template_pass3_content:
        print("Fatal: Could not load third-pass consolidation prompt. Cannot proceed.")
        exit(1)

    prompt_template_pass4_content = read_prompt_template(PROMPT_FILE_PATH_PASS_4)
    if not prompt_template_pass4_content:
        print("Fatal: Could not load fourth-pass consolidation prompt. Cannot proceed.")
        exit(1)

    # --- Data Loading & Processing ---
    current_graph_data = None
    all_failed_reports = []
    all_failed_data = []
    num_input_nodes = -1
    num_input_relationships = -1

    if args.start_pass == 1:
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            print(f"\nSuccessfully loaded '{CSV_FILE_PATH}'. Shape: {df.shape}")
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{CSV_FILE_PATH}'.")
            exit(1)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            exit(1)

        if GRAPH_DATA_COLUMN not in df.columns:
            print(f"Error: Column '{GRAPH_DATA_COLUMN}' not found in the CSV.")
            exit(1)

        current_graph_data = aggregate_graph_data(df, GRAPH_DATA_COLUMN)
        num_input_nodes = len(current_graph_data.get('nodes', []))
        num_input_relationships = len(current_graph_data.get('relationships', []))

        print(f"\n--- Aggregated Input Graph Summary ---")
        print(f"Number of input nodes before consolidation: {num_input_nodes}")
        print(f"Number of input relationships before consolidation: {num_input_relationships}")

        if num_input_nodes == 0 and num_input_relationships == 0:
            print("No valid graph data found to consolidate.")
            exit(0)
    else:
        input_filename = f"consolidated_graph_pass{args.start_pass - 1}.json"
        print(f"\n--- Starting from Pass {args.start_pass} ---")
        print(f"Loading input data from '{input_filename}'...")
        current_graph_data = load_graph_from_json(input_filename)
        if not current_graph_data or (not current_graph_data.get('nodes') and not current_graph_data.get('relationships')):
            print(f"Could not load valid data for Pass {args.start_pass} from '{input_filename}'. Halting.")
            exit(1)

    # --- Pass 1 ---
    if args.start_pass <= 1 and args.end_pass >= 1:
        pass1_input_chunks = create_relationship_chunks(current_graph_data, N_CHUNKS_PASS_1)
        pass1_consolidated_graphs, pass1_failed_reports, pass1_failed_data = run_consolidation_pass(
            pass_name="Pass 1",
            input_chunks=pass1_input_chunks,
            model_name=GENERATION_MODEL,
            prompt_template=prompt_template_pass1_content,
            max_chars_limit=MAX_CHARS_PER_CHUNK_PASS_1,
            max_retries=MAX_RETRIES_PER_CHUNK
        )
        all_failed_reports.extend(pass1_failed_reports)
        all_failed_data.extend(pass1_failed_data)

        current_graph_data = combine_graphs(pass1_consolidated_graphs)
        print(f"\n--- After Pass 1 Combination ---")
        print(f"Graph contains {len(current_graph_data['nodes'])} nodes and {len(current_graph_data['relationships'])} relationships.")
        save_graph_to_json(current_graph_data, OUTPUT_FILENAME_PASS_1)
        write_failures_to_csv(pass1_failed_data, "chunk_failed_reasons_pass1.csv")

    # --- Pass 2 ---
    if args.start_pass <= 2 and args.end_pass >= 2:
        if not current_graph_data.get('nodes') and not current_graph_data.get('relationships'):
            print("No data from previous pass. Halting process.")
        else:
            pass2_input_chunks = create_relationship_chunks(current_graph_data, N_CHUNKS_PASS_2)
            pass2_consolidated_graphs, pass2_failed_reports, pass2_failed_data = run_consolidation_pass(
                pass_name="Pass 2",
                input_chunks=pass2_input_chunks,
                model_name=GENERATION_MODEL,
                prompt_template=prompt_template_pass2_content,
                max_chars_limit=MAX_CHARS_PER_CHUNK_PASS_2,
                max_retries=MAX_RETRIES_PER_CHUNK
            )
            all_failed_reports.extend(pass2_failed_reports)
            all_failed_data.extend(pass2_failed_data)

            current_graph_data = combine_graphs(pass2_consolidated_graphs)
            print(f"\n--- After Pass 2 Combination ---")
            print(f"Graph contains {len(current_graph_data['nodes'])} nodes and {len(current_graph_data['relationships'])} relationships.")
            save_graph_to_json(current_graph_data, OUTPUT_FILENAME_PASS_2)
            write_failures_to_csv(pass2_failed_data, "chunk_failed_reasons_pass2.csv")

    # --- Pass 3 ---
    if args.start_pass <= 3 and args.end_pass >= 3:
        if not current_graph_data.get('nodes') and not current_graph_data.get('relationships'):
            print("No data from previous pass. Halting process.")
        else:
            pass3_input_chunks = create_relationship_chunks(current_graph_data, N_CHUNKS_PASS_3)
            pass3_consolidated_graphs, pass3_failed_reports, pass3_failed_data = run_consolidation_pass(
                pass_name="Pass 3",
                input_chunks=pass3_input_chunks,
                model_name=GENERATION_MODEL,
                prompt_template=prompt_template_pass3_content,
                max_chars_limit=MAX_CHARS_PER_CHUNK_PASS_3,
                max_retries=MAX_RETRIES_PER_CHUNK
            )
            all_failed_reports.extend(pass3_failed_reports)
            all_failed_data.extend(pass3_failed_data)

            current_graph_data = combine_graphs(pass3_consolidated_graphs)
            print(f"\n--- After Pass 3 Combination ---")
            print(f"Graph contains {len(current_graph_data['nodes'])} nodes and {len(current_graph_data['relationships'])} relationships.")
            save_graph_to_json(current_graph_data, OUTPUT_FILENAME_PASS_3)
            write_failures_to_csv(pass3_failed_data, "chunk_failed_reasons_pass3.csv")

    # --- Pass 4 ---
    if args.start_pass <= 4 and args.end_pass >= 4:
        if not current_graph_data.get('nodes') and not current_graph_data.get('relationships'):
            print("No data from previous pass. Halting process.")
        else:
            pass4_input_chunks = create_relationship_chunks(current_graph_data, N_CHUNKS_PASS_4)
            pass4_consolidated_graphs, pass4_failed_reports, pass4_failed_data = run_consolidation_pass(
                pass_name="Pass 4",
                input_chunks=pass4_input_chunks,
                model_name=GENERATION_MODEL,
                prompt_template=prompt_template_pass4_content,
                max_chars_limit=MAX_CHARS_PER_CHUNK_PASS_4,
                max_retries=MAX_RETRIES_PER_CHUNK
            )
            all_failed_reports.extend(pass4_failed_reports)
            all_failed_data.extend(pass4_failed_data)

            current_graph_data = combine_graphs(pass4_consolidated_graphs)
            write_failures_to_csv(pass4_failed_data, "chunk_failed_reasons_pass4.csv")

    # --- Save Final Results ---
    final_graph = current_graph_data
    num_final_nodes = len(final_graph.get('nodes', []))
    num_final_rels = len(final_graph.get('relationships', []))

    print(f"\n--- Final Combined Graph Summary ---")
    print(f"Final graph contains {num_final_nodes} nodes and {num_final_rels} relationships.")
    if num_input_nodes != -1:
        print(f"  (Original input: {num_input_nodes} nodes, {num_input_relationships} relationships)")

    # Save the final consolidated graph
    if args.end_pass == 4:
        save_graph_to_json(final_graph, FINAL_OUTPUT_FILENAME, is_final=True)

    if all_failed_reports:
        print("\n\n" + "="*80)
        print("--- Summary of All Failed Chunk Consolidations ---")
        print("="*80)
        for report in all_failed_reports:
            print(report)
            print("-"*80)

    # --- Write all failed chunks to a consolidated CSV ---
    if all_failed_data:
        print("\n\n--- Writing All Failed Chunk Reasons to a Consolidated CSV ---")
        try:
            failed_df = pd.DataFrame(all_failed_data)
            failed_csv_path = os.path.join(SCRIPT_DIR, 'chunk_failed_reasons_all_passes.csv')
            failed_df.to_csv(failed_csv_path, index=False, encoding='utf-8')
            print(f"Saved consolidated details for {len(all_failed_data)} failed chunks to '{failed_csv_path}'")
        except Exception as e:
            print(f"Error writing consolidated failed chunks to CSV: {e}")

    print("\n\n--- Graph Data Consolidation Complete ---")

if __name__ == "__main__":
    main()