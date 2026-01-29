# 🌍 GraphRAG Core: Climate Edition

**Agnostic Graph Retrieval-Augmented Generation Engine with Embedded Climate Intelligence.**

This library provides a powerful "Node-Centric" retrieval engine for Knowledge Graphs. 
It comes **pre-loaded with a Climate Adaptation Knowledge Base** (vectorized papers + causal graph), making it instantly usable for research.

## 📦 Installation

```bash
pip install git+https://github.com/nunezmatias/grafoRag.git
```

## 🚀 Quick Start (Zero Config)

Since the data is embedded, you can start searching immediately:

```python
from graphrag_core import GraphRAGEngine

# Initialize (No paths needed -> Uses internal Climate DB)
engine = GraphRAGEngine()

# 1. Search for a complex topic
results = engine.search(
    query="cascading risks of heatwaves and power outages",
    top_k=3,       # Find 3 main topics
    context_k=4,   # Read 4 deep-dive papers per topic
    hops=2         # Traverse 2 levels of causal links
)

# 2. Generate an Expert Prompt
prompt = engine.format_prompt(
    results, 
    query="cascading risks...",
    system_role="You are a UN Crisis Response Specialist."
)

print(prompt)
```

## 🛠️ Advanced: Using Your Own Data

This engine is agnostic. You can plug in your own Medical, Legal, or Financial data:

```python
# Initialize with custom paths
custom_engine = GraphRAGEngine(
    vector_db_path="./my_medical_db",
    graph_json_path="./my_medical_graph.json"
)
```

## 🏗️ Building a New Graph

If you have raw JSON data, use the Builder to create a compatible Vector DB:

```python
from graphrag_core import GraphBuilder

builder = GraphBuilder(output_vector_db_path="./new_db")
builder.build(
    input_json_path="./raw_data.json",
    output_json_skeleton_path="./new_skeleton.json"
)
```
