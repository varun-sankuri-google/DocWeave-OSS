import pandas as pd
import json
import os
import google.generativeai as genai
import argparse
import time
from config import GENERATION_MODEL, PROJECT_ID, GOOGLE_API_KEY, TEMPERATURE
from collections import defaultdict

# --- Configuration ---
CSV_FILE_PATH = 'support_tickets/buganizer_issues.csv'
TICKET_SUBJECT_COLUMN = ''
TICKET_DESCRIPTION_COLUMN = 'text'
CASE_ID_COLUMN = 'bug_issue_id'
PROMPT_FILENAME = "buganizer_nodes.txt"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "json_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROMPT_FILE_PATH = os.path.join(SCRIPT_DIR, "prompts", PROMPT_FILENAME)
TICKET_ASSIGNMENTS_FILENAME = os.path.join(OUTPUT_DIR, "buganizer_nodes.json")


CHUNK_SIZE = 100
MAX_NODES_PER_PROMPT = 10000
MAX_RETRIES_PER_CHUNK = 2
MAX_OUTPUT_TOKENS_GEMINI = 58000

def read_prompt_template(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Escape braces for .format() by doubling them
            content = f.read().strip()
            return content.replace('{', '{{').replace('}', '}}').replace('{{tickets}}', '{tickets}').replace('{{nodes}}', '{nodes}')

    except FileNotFoundError:
        print(f"Error: Prompt file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading prompt file {filepath}: {e}")
        return None

def load_unique_nodes_from_json(filepath):
    """Loads a list of unique node names from a JSON file with a 'nodes' key."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "nodes" not in data:
                raise ValueError("Expected a JSON object with a 'nodes' key.")
            
            nodes_list = data["nodes"]
            if not isinstance(nodes_list, list):
                raise ValueError("Expected 'nodes' to be a JSON array.")

            node_names = [node.get("entity_name") for node in nodes_list if isinstance(node, dict) and "entity_name" in node]
            return list(set(node_names))

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading or processing node list from '{filepath}': {e}")
        return []


def assign_tickets_to_nodes(ticket_data, node_list, model_name, prompt_template):
    """Calls Gemini to assign tickets to nodes and count assignments."""
    if not ticket_data or not prompt_template or not node_list:
        return None, "Error: Ticket data, node list, or prompt template missing."

    formatted_tickets = "\n\n".join([
        f"Ticket {case_id}: Subject: {subject}\nDescription: {description}"
        for case_id, subject, description in ticket_data
    ])

    if len(node_list) > MAX_NODES_PER_PROMPT:
        print(f"Warning: Node list is too long ({len(node_list)} > {MAX_NODES_PER_PROMPT}). Processing a subset.")
        node_list = node_list[:MAX_NODES_PER_PROMPT]

    formatted_nodes = "\n".join([
        f"- {node_name}"
        for node_name in node_list
    ])

    prompt = prompt_template.format(tickets=formatted_tickets, nodes=formatted_nodes)

    try:
        print(f"Sending prompt (length: {len(prompt)} chars) to Gemini model '{model_name}'...")
        model = genai.GenerativeModel(model_name)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=MAX_OUTPUT_TOKENS_GEMINI,
            temperature=TEMPERATURE,
        )
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        if not response.text or not response.candidates:
            return None, f"Gemini API returned an empty or invalid response.\n{format_error_report(response)}"

        assignments_str = response.text.strip()
        if assignments_str.startswith("```json"):
            assignments_str = assignments_str[7:].strip()
        if assignments_str.endswith("```"):
            assignments_str = assignments_str[:-3].strip()

        assignments_data = json.loads(assignments_str)
        if not isinstance(assignments_data, dict):
            return None, f"Response was not a JSON object with node assignments. Response:\n{assignments_str[:500]}..."
        
        if hasattr(response, 'usage_metadata'):
            print(f"Successfully processed. Usage Metadata: {response.usage_metadata}")

        return assignments_data, None

    except json.JSONDecodeError as e:
        return None, f"Error decoding JSON from Gemini response: {e}\n{format_error_report(locals().get('response'))}"
    except Exception as e:
        return None, f"An unexpected error occurred during Gemini call: {e}\n{format_error_report(locals().get('response'))}"

def format_error_report(response_obj):
    if not response_obj:
        return "No response object available."

    report_parts = []
    if hasattr(response_obj, 'prompt_feedback') and response_obj.prompt_feedback:
        report_parts.append(f"Prompt Feedback: {response_obj.prompt_feedback}")
    if hasattr(response_obj, 'candidates') and response_obj.candidates:
        for i, candidate in enumerate(response_obj.candidates):
            report_parts.append(f"Candidate {i} Finish Reason: {candidate.finish_reason}")
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                report_parts.append(f"Candidate {i} Safety Ratings: {candidate.safety_ratings}")
    if hasattr(response_obj, 'usage_metadata'):
        report_parts.append(f"Usage Metadata: {response_obj.usage_metadata}")
    if hasattr(response_obj, 'text') and response_obj.text:
        response_text_preview = response_obj.text
        if len(response_text_preview) > 500:
            response_text_preview = response_text_preview[:500] + "... (truncated in log)"
        report_parts.append(f"Response Text (preview): '{response_text_preview}'")

    return "\n".join(report_parts) if report_parts else "No detailed error information found in response object."

def process_tickets_in_chunks(df, model_name, prompt_template, unique_nodes):
    all_chunk_results = []
    num_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"\n--- Processing {len(df)} tickets in {num_chunks} chunks ---")

    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(df))      
        ticket_chunk = list(zip(
            df[CASE_ID_COLUMN][start_idx:end_idx],
            df[TICKET_SUBJECT_COLUMN][start_idx:end_idx] if TICKET_SUBJECT_COLUMN else [''] * (end_idx - start_idx),
            df[TICKET_DESCRIPTION_COLUMN][start_idx:end_idx]
        ))

        print(f"\nProcessing chunk {i + 1}/{num_chunks} (tickets {start_idx + 1}-{end_idx})...")
        chunk_assignments = None
        last_error_report = "No error report generated."
        for attempt in range(MAX_RETRIES_PER_CHUNK + 1):
            chunk_assignments, error_report = assign_tickets_to_nodes(
                ticket_chunk, unique_nodes, model_name, prompt_template
            )
            last_error_report = error_report
            if chunk_assignments:
                print(f"Chunk {i + 1} processed successfully on attempt {attempt + 1}.")
                break
            else:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES_PER_CHUNK + 1} failed for chunk {i}.\n{last_error_report}")
                if attempt < MAX_RETRIES_PER_CHUNK:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
        if not chunk_assignments:
            print(f"Failed to process chunk {i + 1} after all attempts. Skipping this chunk's tickets.")
        else:
            all_chunk_results.append(chunk_assignments)

    return all_chunk_results

def save_chunk_results_to_json(chunk_data, filename): 
    if not chunk_data:
        print(f"No chunk data to save. Skipping '{filename}'.")
        return

    print(f"\nSaving {len(chunk_data)} chunk results to '{filename}'...")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, indent=2)
        print(f"Successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving chunk results to file '{filename}': {e}")

def aggregate_and_merge_buganizer_data(ticket_assignments_file, consolidated_graph_file, output_file):
    """
    Aggregates buganizer issue assignments from chunks and merges them into the main knowledge graph.
    """
    print("\n--- Aggregating and Merging Buganizer Data ---")

    # Step 1: Load and aggregate ticket assignments from all chunks
    try:
        with open(ticket_assignments_file, 'r', encoding='utf-8') as f:
            ticket_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse ticket assignments file '{ticket_assignments_file}': {e}")
        return

    # Use defaultdict to aggregate lists of ticket IDs
    aggregated_tickets = defaultdict(list)
    total_assignments = 0
    for chunk_result in ticket_data:
        for node_name, ticket_ids in chunk_result.items():
            if isinstance(ticket_ids, list):
                aggregated_tickets[node_name].extend(ticket_ids)
                total_assignments += len(ticket_ids)

    print(f"Aggregated {total_assignments} ticket assignments across {len(aggregated_tickets)} unique nodes.")

    # Step 2: Load the main knowledge graph
    try:
        with open(consolidated_graph_file, 'r', encoding='utf-8') as f:
            consolidated_graph = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse consolidated graph file '{consolidated_graph_file}': {e}")
        return

    # Step 3: Merge counts and ticket lists into the main graph nodes
    nodes_in_graph = consolidated_graph.get("nodes", [])
    updated_nodes_count = 0
    total_tickets_in_graph = 0
    for node in nodes_in_graph:
        entity_name = node.get("entity_name")
        if not entity_name:
            continue

        # Get the aggregated list of tickets, ensure uniqueness, and sort them
        ticket_list = sorted([str(x) for x in set(aggregated_tickets.get(entity_name, []))])
        ticket_count = len(ticket_list)
        total_tickets_in_graph += ticket_count

        # Ensure the 'properties' dictionary exists
        if "properties" not in node:
            node["properties"] = {}

        # Add the new fields for the list of tickets and the total count
        node["properties"]["buganizer_count"] = ticket_count
        node["properties"]["buganizer_ids"] = ticket_list
        if ticket_count > 0:
            updated_nodes_count += 1

    print(f"Merged ticket data into the consolidated graph. {updated_nodes_count} nodes were updated.")
    print(f"A total of {total_tickets_in_graph} unique buganizer issue references were added to the graph nodes.")

    # Step 4: Save the final, updated graph to a new file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_graph, f, indent=2)
        print(f"Successfully saved final graph with ticket data to '{output_file}'")
    except Exception as e:
        print(f"Error saving the final graph file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract and process Buganizer nodes.")
    args = parser.parse_args()

    print("--- Starting Buganizer Node Extraction ---")

    if not GOOGLE_API_KEY:
        print("Fatal: GOOGLE_API_KEY not found.")
        exit(1)
    try:
        print(f"Configuring Gemini with API Key. (Project context: {PROJECT_ID})")
        genai.configure(api_key=GOOGLE_API_KEY)
        genai.get_model(GENERATION_MODEL)
        print(f"Successfully configured Gemini and verified access to model '{GENERATION_MODEL}'.")
    except Exception as e:
        print(f"Fatal: Error configuring or verifying Gemini: {e}.")
        exit(1)

    prompt_template_content = read_prompt_template(PROMPT_FILE_PATH)
    if not prompt_template_content:
        print("Fatal: Could not load prompt template.")
        exit(1)

    # Load the CSV file containing support tickets
    try: 
        df = pd.read_csv(CSV_FILE_PATH) 
        print(f"Successfully loaded '{CSV_FILE_PATH}'. Shape: {df.shape}") 
    except FileNotFoundError: 
        print(f"Error: CSV file not found at '{CSV_FILE_PATH}'.") 
        exit(1) 
    except Exception as e: 
        print(f"Error loading CSV file: {e}") 
        exit(1) 

    required_columns = [CASE_ID_COLUMN, TICKET_DESCRIPTION_COLUMN]
    if TICKET_SUBJECT_COLUMN:
        required_columns.append(TICKET_SUBJECT_COLUMN)

    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the CSV.")
            exit(1)

    if df.empty:
        print(f"Warning: Input CSV '{CSV_FILE_PATH}' is empty.")
        save_chunk_results_to_json([], TICKET_ASSIGNMENTS_FILENAME)
        exit(0)

    for i in range(1, 5):
      consolidated_graph_file = os.path.join(OUTPUT_DIR, f"consolidated_graph_with_counts_pass{i}.json")
      ticket_assignments_filename = os.path.join(OUTPUT_DIR, f"buganizer_nodes_pass{i}.json")
      
      unique_nodes = load_unique_nodes_from_json(consolidated_graph_file)
      if not unique_nodes:
          print(f"Fatal: Could not load unique nodes from JSON: {consolidated_graph_file}")
          continue
      print(f"Loaded {len(unique_nodes)} unique nodes from consolidated graph: {consolidated_graph_file}")

      all_chunk_results = process_tickets_in_chunks(df, GENERATION_MODEL, prompt_template_content, unique_nodes)
      save_chunk_results_to_json(all_chunk_results, ticket_assignments_filename)

      aggregate_and_merge_buganizer_data(ticket_assignments_filename, consolidated_graph_file, consolidated_graph_file)

    

    print("\n--- Buganizer Node Processing Complete ---")


if __name__ == "__main__":
    main()