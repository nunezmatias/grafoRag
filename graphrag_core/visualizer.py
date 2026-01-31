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
        # PyVis handles node creation automatically when adding edges, but we want custom attributes
        G.add_edge(u, v, title=rel, label=rel)
        nodes.add(u)
        nodes.add(v)

    net.from_nx(G)
    
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
