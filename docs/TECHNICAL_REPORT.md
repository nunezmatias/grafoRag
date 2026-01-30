# 📘 GraphRAG Core: Technical Documentation & Developer Guide

**Date:** January 29, 2026
**Version:** 1.0 (Node-Centric V2)
**Maintainer:** Matias Nuñez

---

## 1. System Overview

**GraphRAG Core** is a domain-agnostic retrieval engine designed to bridge the gap between Unstructured Text (Vector Search) and Structured Knowledge (Knowledge Graphs). Unlike traditional RAG, which retrieves independent text chunks, this system retrieves **Concepts** and their **Causal Chains**.

### Core Architecture: "Node-Centric Retrieval"
The system operates on a specific philosophy distinguished from standard RAG:
1.  **Identify the Node:** First, find the "Concept" (e.g., *Heat Wave*) relevant to the query.
2.  **Deep Dive (Intra-Node):** Once the concept is found, perform a secondary semantic search *inside* that node's pool of documents to find the best evidence.
3.  **Traverse (Inter-Node):** Follow edges (e.g., *CAUSES*) to find downstream impacts.

---

## 2. Data Engineering & Graph Construction

The Knowledge Graph is the heart of the system. It is built using `graphrag_core.builder.GraphBuilder`.

### 2.1. Input Data Schema (JSON)
The system expects a JSON file with the following structure:

```json
{
  "nodes": [
    {
      "id": "concept_id",
      "label": "Human Readable Label",
      "type": "category",
      "properties": {
        "references": [
          {
            "text": "Full text or abstract from a scientific paper...",
            "title": "Paper Title",
            "url": "https://doi.org/...",
            "year": 2024
          }
        ]
      }
    }
  ],
  "edges": [
    {
      "source": "concept_id_A",
      "target": "concept_id_B",
      "relation": "CAUSES",
      "evidence": {
        "snippet": "Text proving A causes B...",
        "justification": "Explanation..."
      }
    }
  ]
}
```

### 2.2. Vectorization Strategy (`builder.py`)
*   **Model:** `BAAI/bge-base-en-v1.5` (768 dimensions). Selected for its high MTEB performance and compact size.
*   **Storage:** ChromaDB (Persistent).
*   **Collections:**
    1.  `nodes_references_base`: Stores the text content of nodes (References). Metadata includes `node_id`, `url`, `year`.
    2.  `edges_evidence_base`: Stores the text justification for edges. Metadata includes `edge_id`, `source`, `target`, `relation`.

**Key Technical Detail:** Metadata is "flattened". Nested JSON objects in input properties are converted to string fields in ChromaDB metadata to ensure compatibility.

---

## 3. Retrieval Engine Logic (`engine.py`)

The `GraphRAGEngine` class implements a multi-stage retrieval pipeline.

### Stage 1: Primary Vector Search (`top_k`)
*   **Action:** Query `nodes_references_base` with the user question.
*   **Output:** Returns the top $k$ chunks globally.
*   **Purpose:** To identify the "Anchor Nodes" (Themes).

### Stage 2: Context Expansion (`context_k`) - *The V2 Innovation*
*   **Problem:** A global search might return a random paragraph from a relevant node.
*   **Solution:** For each identified Anchor Node, the engine executes a **new, restricted vector search** specifically filtering `where={"node_id": anchor_id}`.
*   **Action:** Retrieves the top $context_k$ chunks *within* that node that best match the query.
*   **Result:** High-precision evidence specific to the node's topic.

### Stage 3: Graph Traversal (`hops`)
*   **Action:** Uses NetworkX to perform a BFS (Breadth-First Search) starting from Anchor Nodes.
*   **Logic:**
    *   Find neighbors (Forward direction).
    *   Retrieve the specific Edge Evidence from `edges_evidence_base` using the unique `edge_id`.
*   **Output:** A list of causal triplets (`A --[REL]--> B`) supported by textual evidence.

---

## 4. Prompt Engineering ("Expert Template")

The system does not just dump data to the LLM. It structures it using a specialized template (`format_prompt`):

1.  **Data Source Legend:** Explicitly defines `[REF_PRI]` (Direct Match), `[REF_CTX]` (Context), and `[GRAPH]` (Causality).
2.  **Triangulation:** Instructions to cross-reference Graph Evidence with Textual Evidence.
3.  **Chain Detection:** Instructions to identify transitive chains ($A \rightarrow B \rightarrow C$) for "Cascading Risk" analysis.

---

## 5. Replication & Development Guide

### How to Rebuild the Graph (New Data)
If you modify the source JSON, you must rebuild the DB:

```python
from graphrag_core import GraphBuilder

# 1. Define paths
RAW_JSON = "./my_new_data.json"
OUTPUT_DB = "./new_vectordb"
OUTPUT_SKELETON = "./new_skeleton.json"

# 2. Build
builder = GraphBuilder(output_vector_db_path=OUTPUT_DB)
builder.build(RAW_JSON, OUTPUT_SKELETON)

# 3. Zip and Upload (If distributing via Drive)
# zip -r data.zip new_vectordb new_skeleton.json
```

### Context for AI Agents (LLMs)
If you are an AI assistant tasked with modifying this code:
*   **`engine.py`**: Contains the retrieval logic. Modify `search()` to change how data is fetched. Modify `format_prompt()` to change how the LLM sees the data.
*   **`builder.py`**: Contains ingestion logic. Modify this if the input JSON schema changes.
*   **`setup.py`**: Controls dependencies.
*   **`data/`**: This directory is NOT tracked by git (except the skeleton). It is downloaded at runtime from Google Drive.


### Environment Setup
*   **Python:** 3.9+
*   **Dependencies:** `chromadb`, `sentence-transformers`, `networkx`, `torch`, `gdown`.
*   **Hardware:** Optimized for Apple Silicon (`mps`) and CUDA. Defaults to CPU if neither is available.

---

## 6. Known Limitations
*   **Static Graph:** The graph structure is loaded into memory (NetworkX). For extremely massive graphs (>1M nodes), this needs to be migrated to a Graph Database (Neo4j).
---

## 7. Customizing Data Sources

The engine is designed to be "Plug & Play" with the default Climate Database, but it is fully agnostic. You can swap the "brain" of the system by providing your own data.

### 7.1. Expected Data Package (ZIP)
If you want to host your own data on Google Drive for auto-download, package a `.zip` file containing:

1.  **`optimized_graph_base.json`**: The topology file.
2.  **`climate_knowledge_vectordb_base/`**: The ChromaDB folder.

**Zip Structure:**
```text
my_data.zip
├── optimized_graph_base.json
└── climate_knowledge_vectordb_base/
    ├── chroma.sqlite3
    └── ... (other chroma files)
```
*Note: The engine's smart extractor will find these files even if they are nested inside folders within the zip, but flat is better.*

### 7.2. Using a Custom Database
You can initialize the engine with a custom Google Drive ID to download your own data:

```python
engine = GraphRAGEngine(
    gdrive_id="YOUR_NEW_FILE_ID_HERE"
)
```

Or point to local paths if you have the files already:

```python
engine = GraphRAGEngine(
    vector_db_path="./my_local_db",
    graph_json_path="./my_local_graph.json"
)
```

### 7.3. ChromaDB Collections Schema
If you build the database yourself (using `builder.py`), ensure these collection names are used:

*   **`nodes_references_base`**:
    *   **Document**: Text chunk of the node.
    *   **Metadata**: `node_id`, `url`, `year`, `title`.
*   **`edges_evidence_base`**:
    *   **Document**: Text snippet justifying the link.
    *   **Metadata**: `edge_id`, `source`, `target`, `relation`.
