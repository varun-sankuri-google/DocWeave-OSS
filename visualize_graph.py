# command to create HTTP SERVER
# python3 -m http.server 8080
import json
from pyvis.network import Network
import networkx as nx
import os
import matplotlib.colors as mcolors
import matplotlib

# --- Global Configuration Constants ---

# Level of Detail (LOD) settings for dynamic node visibility.
# This avoids hardcoding values in multiple places.
LOD_DEGREE_THRESHOLD = 3  # Nodes with degree <= this are affected by LOD.
LOD_ZOOM_THRESHOLD = 0.8  # Zoom level (scale) at which nodes appear.

def _inject_client_side_enhancements(html_path, lod_config=None, freeze_layout=False):
    """
    Injects JavaScript into the generated HTML file to add client-side enhancements
    like Level of Detail (LOD) and freezing the layout after stabilization.
    """
    if not lod_config and not freeze_layout:
        return

    try:
        print(f"Injecting client-side enhancements into '{html_path}'...")
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # The JS code to inject. It finds the 'network' object created by vis.js
        # and adds a 'zoom' event listener to show/hide nodes.
        js_script = f"""
<script type="text/javascript">
  document.addEventListener("DOMContentLoaded", function() {{
    // Wait for the 'network' object to be initialized by PyVis
    var checkExist = setInterval(function() {{
       if (typeof network !== 'undefined' && network) {{
          console.log("Network object found, attaching client-side enhancements.");
          clearInterval(checkExist);

          // --- 1. Level of Detail (LOD) Enhancement ---
          {''.join([f'''
          (function() {{
            const nodeDataSet = network.body.data.nodes; // The DataSet object
            const allNodeIds = nodeDataSet.getIds(); // Get all IDs, including hidden ones
            const zoomThreshold = {lod_config['zoom_threshold']};
            const degreeThreshold = {lod_config['degree_threshold']};

            function updateNodeVisibility() {{
              const currentScale = network.getScale();
              let nodesToUpdate = [];

              // Iterate over all node IDs to ensure we check every node, even hidden ones.
              allNodeIds.forEach(id => {{
                const node = nodeDataSet.get(id); // Get the full node object by ID

                // Apply LOD logic only to low-degree nodes
                if (node && node.full_degree !== undefined && node.full_degree <= degreeThreshold) {{
                  // Nodes should be hidden when zoomed OUT (scale is small)
                  const shouldBeHidden = currentScale < zoomThreshold;

                  // Update only if the state needs to change
                  if (node.hidden !== shouldBeHidden) {{
                    nodesToUpdate.push({{ id: node.id, hidden: shouldBeHidden }});
                  }}
                }}
              }});

              if (nodesToUpdate.length > 0) {{
                console.log(`LOD: Updating visibility of ${{nodesToUpdate.length}} nodes based on zoom (${{currentScale.toFixed(2)}}).`);
                nodeDataSet.update(nodesToUpdate);
              }}
            }}

            // Attach listener to the zoom event
            network.on('zoom', updateNodeVisibility);

            // Run once on load to set the initial state
            console.log("LOD: Performing initial node visibility check.");
            updateNodeVisibility();
          }})();
          ''' if lod_config else ""])}

          // --- 2. Freeze Layout Enhancement ---
          {''.join([f'''
          (function() {{
            // The 'stabilizationIterationsDone' event can be unreliable on complex graphs.
            // Instead, we'll forcefully turn off physics after a timeout to ensure
            // the graph becomes static and usable.
            const freezeTimeout = 10000; // 10 seconds
            setTimeout(function() {{
                if (network.physics.options.enabled) {{
                    console.log(`Timeout of ${{freezeTimeout / 1000}}s reached. Freezing physics to prevent further movement.`);
                    network.setOptions( {{ physics: false }} );
                }}
            }}, freezeTimeout);
          }})();
          ''' if freeze_layout else ""])}
       }}
    }}, 100); // Check for the network object every 100ms
  }});
</script>
        """

        # Inject the script just before the closing </body> tag for robustness
        body_end_tag = "</body>"
        if body_end_tag in html_content:
            html_content = html_content.replace(body_end_tag, js_script + "\n" + body_end_tag)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("Successfully injected enhancements script.")
        else:
            print("Warning: Could not find </body> tag. Script not injected.")

    except Exception as e:
        print(f"Error injecting script into '{html_path}': {e}")

def visualize_graph_from_json(json_filepath, output_html_filename="graph_visualization.html", dynamic_lod=True, freeze_layout_after_load=True):
    """
    Loads consolidated graph data from a JSON file containing 'nodes' and
    'relationships', and generates an interactive HTML visualization using PyVis.
    """
    try:
        with open(json_filepath, 'r') as f:
            graph_data_json = json.load(f)
        print(f"Successfully loaded '{json_filepath}'.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_filepath}'")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_filepath}': {e}")
        return
    except Exception as e:
        print(f"Error loading JSON '{json_filepath}': {e}")
        return

    # --- 1. Build NetworkX graph for metrics calculation ---
    G = nx.DiGraph()
    nodes_in_json = graph_data_json.get('nodes', [])
    relationships_in_json = graph_data_json.get('relationships', [])
    node_properties_lookup = {node['entity_name']: node for node in nodes_in_json if 'entity_name' in node}
    for node_name in node_properties_lookup.keys():
        G.add_node(node_name)
    for rel_data in relationships_in_json:
        source_name = rel_data.get('source_entity_name')
        target_name = rel_data.get('target_entity_name')
        if source_name and target_name and G.has_node(source_name) and G.has_node(target_name):
            G.add_edge(source_name, target_name)
    if not G.nodes():
        print("No nodes found to visualize after processing the JSON file.")
        return

    # --- 2. Calculate graph metrics ---
    print("Calculating graph metrics (degree, betweenness centrality)...")
    betweenness_centrality = nx.betweenness_centrality(G)
    degrees = dict(G.degree())

    # --- 3. Prepare for color mapping based on Betweenness Centrality ---
    max_betweenness = max(betweenness_centrality.values()) if betweenness_centrality else 0
    norm_betweenness = {node: (val / max_betweenness if max_betweenness > 0 else 0) for node, val in betweenness_centrality.items()}
    colormap = matplotlib.colormaps['hot_r']

    # Initialize PyVis Network
    net = Network(notebook=False, height="800px", width="100%", directed=True, cdn_resources='remote', select_menu=True, filter_menu=True)

    # --- Physics and Interaction Configuration ---
    net.options.physics.solver = 'barnesHut'
    net.options.physics.barnesHut = {
        "gravitationalConstant": -8000, "centralGravity": 0.15, "springLength": 150,
        "springConstant": 0.05, "damping": 0.09, "avoidOverlap": 1.0
    }
    net.options.physics.stabilization.iterations = 1000
    net.options.physics.stabilization.fit = True
    net.options.interaction.hover = True
    net.options.interaction.tooltipDelay = 200

    # --- 4. Add nodes to PyVis network with calculated sizes and colors ---
    print("Adding nodes to visualization with calculated sizes and colors...")
    for node_name in G.nodes():
        # Get original properties
        node_data = node_properties_lookup.get(node_name, {})
        properties = node_data.get('properties', {})
        node_type = node_data.get('entity_type', 'Unknown')
        
        # === ATTENTION SCORE LOGIC START ===
        # Get metrics for calculation
        source_urls = properties.get('source_document_url', [])
        support_tickets = properties.get('support_tickets', [])

        # Deduplicate to get accurate counts, making the calculation more robust
        # against potential inconsistencies in the source JSON.
        source_doc_count = len(set(source_urls))
        support_ticket_count = len(set(support_tickets))
        
        # Calculate the Attention Score
        attention_score = support_ticket_count / (source_doc_count + 0.1)
        
        # Determine node size based on the new Attention Score
        # Base size 15, scaled by the score. Adjust the multiplier (e.g., 10) as needed.
        node_size = 15 + (attention_score * 10)
        # === ATTENTION SCORE LOGIC END ===

        # Get other metrics for coloring and tooltips
        betweenness = betweenness_centrality.get(node_name, 0)
        full_degree = degrees.get(node_name, 0)

        # Determine node color based on betweenness centrality
        normalized_b = norm_betweenness.get(node_name, 0)
        node_color_rgba = colormap(normalized_b)
        node_color_hex = mcolors.to_hex(node_color_rgba)

        # Create rich hover title with the new Attention Score
        title_parts = [
            f"Name: {node_name}",
            f"Type: {node_type}",
            "--------------------",
            f"Attention Score: {attention_score:.2f} (determines size)",
            f"Support Ticket Count: {support_ticket_count}",
            f"Source Document Count: {source_doc_count}",
            "--------------------",
            f"Betweenness Centrality: {betweenness:.4f} (determines color)",
            f"Connections (Degree): {full_degree}"
        ]

        # Add top 3 support tickets if available
        if support_tickets:
            title_parts.append("--------------------")
            title_parts.append("Support Tickets (Top 3):")
            for ticket in support_tickets[:3]:
                title_parts.append(f"  - {ticket}")
            if len(support_tickets) > 3:
                title_parts.append("  ...")

        # Add top 3 source document URLs if available
        if source_urls:
            title_parts.append("--------------------")
            title_parts.append("Source URLs (Top 3):")
            for url in source_urls[:3]:
                title_parts.append(f"  - {url}")
            if len(source_urls) > 3:
                title_parts.append("  ...")

        node_hover_title = "\n".join(title_parts)

        # For LOD, hide low-degree nodes initially
        is_hidden = dynamic_lod and full_degree <= LOD_DEGREE_THRESHOLD

        net.add_node(node_name, label=node_name, group=node_type, 
                     title=node_hover_title, size=node_size, color=node_color_hex,
                     hidden=is_hidden,
                     full_degree=full_degree # Custom attribute for our JS
                     )

    # --- 5. Add edges to the PyVis network ---
    print("Adding relationships to visualization...")
    for rel_data in relationships_in_json:
        source_name = rel_data.get('source_entity_name')
        target_name = rel_data.get('target_entity_name')
        rel_type = rel_data.get('relationship_type', 'RELATED_TO')
        frequency = rel_data.get('frequency', 1)

        if G.has_edge(source_name, target_name):
            edge_width = 1 + min(frequency / 2.0, 8.0)
            title = f"Relationship: {rel_type}\nFrequency: {frequency} (determines thickness)"
            net.add_edge(source_name, target_name, label=rel_type, title=title, width=edge_width)

    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    net.save_graph(output_html_filename)
    print(f"\\nGraph visualization saved to '{os.path.abspath(output_html_filename)}'")

    # --- 6. Inject client-side enhancements ---
    if dynamic_lod or freeze_layout_after_load:
        lod_params = {
            "zoom_threshold": LOD_ZOOM_THRESHOLD,
            "degree_threshold": LOD_DEGREE_THRESHOLD
        } if dynamic_lod else None
        _inject_client_side_enhancements(output_html_filename, lod_config=lod_params, freeze_layout=freeze_layout_after_load)

    print("Open this HTML file in your web browser to view the interactive graph.")

if __name__ == "__main__":
    # Build paths relative to the script's location to avoid permission errors.
    project_root = os.path.dirname(os.path.abspath(__file__))
    json_input_dir = os.path.join(project_root, "json_output")
    html_output_dir = os.path.join(project_root, "html_output")

    os.makedirs(html_output_dir, exist_ok=True)

    # Use the filename of the graph that now contains the support ticket counts
    files_to_process = ["consolidated_graph_with_counts_pass1.json",
                        "consolidated_graph_with_counts_pass2.json",
                        "consolidated_graph_with_counts_pass3.json",
                        "consolidated_graph_with_counts_pass4.json"]

    for i, json_filename in enumerate(files_to_process, 1):
        json_filepath = os.path.join(json_input_dir, json_filename)        
        output_html_path = os.path.join(html_output_dir, f"graph_visualization_with_attention_score_{i}.html")

        print("\\n" + "="*80)
        print(f"VISUALIZING: Reading from '{json_filepath}'")
        print(f"             Outputting to '{output_html_path}'")
        print("="*80)

        if not os.path.exists(json_filepath):
            print(f"Warning: Input file not found, skipping: {json_filepath}")
            continue

        visualize_graph_from_json(json_filepath, output_html_path, dynamic_lod=True, freeze_layout_after_load=True)

    print("\\nAll visualization passes complete.")