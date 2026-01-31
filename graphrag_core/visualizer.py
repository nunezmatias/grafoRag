import networkx as nx
from pyvis.network import Network
import random

def visualize_results(results, title="RAG Knowledge Subgraph", height="600px", dark_mode=True):
    """
    Visualizes the subgraph of retrieved results using PyVis.
    
    Args:
        results (dict): The results object returned by engine.search()
        title (str): Title of the visualization
        height (str): Height of the widget (e.g. "600px")
        dark_mode (bool): If True, uses a dark premium theme.
        
    Returns:
        IPython.display.HTML: Interactive HTML object ready to be displayed in Notebooks.
    """
    if not results.get("graph_links"):
        print("No graph data to visualize.")
        return None

    # 1. Create Network
    net = Network(height=height, width="100%", notebook=True, cdn_resources='in_line', bgcolor="#1a1a1a" if dark_mode else "#ffffff", font_color="white" if dark_mode else "black")
    
    # 2. Extract Data
    G = nx.DiGraph()
    
    # Track node types for coloring
    # We don't have explicit "anchor" vs "expanded" in results dict unless we infer it.
    # Heuristic: Nodes in 'node1' of the first link are likely anchors if hops > 0, but let's just color by community or connectivity.
    # Better: Use a reliable set of colors.
    
    nodes = set()
    for link in results["graph_links"]:
        u, v = link["node1"], link["node2"]
        rel = link["relation"]
        
        # Add Edge
        G.add_edge(u, v, title=rel, label=rel)
        
        # Add Node Attributes if available
        # PyVis uses 'title' for tooltips (supports HTML)
        if u not in nodes:
            u_meta = link.get("source_metadata", {})
            u_tooltip = _format_tooltip(u, u_meta)
            G.add_node(u, title=u_tooltip, label=u_meta.get('label', u))
            nodes.add(u)
            
        if v not in nodes:
            v_meta = link.get("target_metadata", {})
            v_tooltip = _format_tooltip(v, v_meta)
            G.add_node(v, title=v_tooltip, label=v_meta.get('label', v))
            nodes.add(v)

    net.from_nx(G)

def _format_tooltip(node_id, meta):
    """
    Creates a clean HTML tooltip from node metadata.
    """
    if not meta: return f"<b>{node_id}</b>"
    
    html = f"<div style='font-family: Arial; font-size: 12px; min-width: 200px;'>"
    html += f"<b style='font-size: 14px; color: #333;'>{meta.get('label', node_id)}</b><hr style='border-top: 1px solid #ccc; margin: 5px 0;'>"
    
    # Prioritize fields to show
    priority_fields = ['type', 'description', 'definition', 'source']
    
    for key in priority_fields:
        if key in meta:
            val = meta[key]
            if len(str(val)) > 300: val = str(val)[:300] + "..."
            html += f"<b>{key.capitalize()}:</b> {val}<br>"
            
    # Show counts if available
    if 'degree' in meta: html += f"<b>Degree:</b> {meta['degree']}<br>"
    
    html += "</div>"
    return html
    
    # 3. Apply Premium Styling
    # Node Styling
    for node in net.nodes:
        node['shape'] = 'dot'
        node['size'] = 20
        node['borderWidth'] = 2
        node['shadow'] = True
        # Random elegant pastel colors for nodes
        node['color'] = random.choice(["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7B731", "#A3CB38"])
        node['font'] = {'size': 14, 'face': 'Verdana'}

    # Edge Styling
    for edge in net.edges:
        edge['color'] = "#555555" if dark_mode else "#cccccc"
        edge['arrows'] = 'to'
        edge['font'] = {'size': 10, 'align': 'middle'}
        
    # Physics Options (Stabilization)
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)

    # 4. Return as HTML
    try:
        # Save to a temporary string buffer? PyVis saves to file.
        # simpler: let it generate html string
        html = net.generate_html()
        from IPython.display import HTML
        return HTML(html)
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None
