import json
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
import os
import sys
import subprocess
import zipfile
import shutil
import random
import glob

# --- CONFIGURACIÓN ---
DEFAULT_CLIMATE_ID = "1jxCFQ9yxAE8IvYlFvRJkHUVemeSJXS1T"

class GraphRAGEngine:
    def __init__(self, vector_db_path=None, graph_json_path=None, gdrive_id=None, model_name="BAAI/bge-base-en-v1.5", device=None):
        """
        Motor GraphRAG Agnóstico con Soporte para Datasets en la Nube.
        """
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Initializing GraphRAG Engine on {self.device.upper()}...")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        # CASO 1: Cerebro Personalizado desde Drive
        if gdrive_id and gdrive_id != DEFAULT_CLIMATE_ID:
            print(f"   🧠 Loading CUSTOM Brain from Drive ID: {gdrive_id}")
            custom_dir = os.path.join("/tmp", f"graphrag_{gdrive_id}")
            
            # Limpieza y descarga fresca si no existe
            if not os.path.exists(custom_dir):
                self._download_custom_data(custom_dir, gdrive_id)
            
            # Descubrimiento automático de archivos (Busca en toda la estructura extraída)
            print(f"   🔍 Searching for Knowledge Base files in {custom_dir}...")
            
            # Buscar el archivo JSON del grafo
            json_files = glob.glob(os.path.join(custom_dir, "**", "*.json"), recursive=True)
            # Buscar la carpeta que contiene chroma.sqlite3
            sqlite_files = glob.glob(os.path.join(custom_dir, "**", "chroma.sqlite3"), recursive=True)
            
            if not json_files:
                raise FileNotFoundError(f"No .json graph file found in the extracted package at {custom_dir}")
            if not sqlite_files:
                raise FileNotFoundError(f"No 'chroma.sqlite3' database found in the extracted package at {custom_dir}")
            
            graph_json_path = json_files[0]
            vector_db_path = os.path.dirname(sqlite_files[0])
            
            print(f"   ✅ Discovered Graph: {os.path.basename(graph_json_path)}")
            print(f"   ✅ Discovered Vector DB: {os.path.basename(vector_db_path)}")

        # CASO 2: Defaults o Rutas Locales
        else:
            if vector_db_path is None:
                vector_db_path = os.path.join(data_dir, "climate_knowledge_vectordb_base")
            if graph_json_path is None:
                graph_json_path = os.path.join(data_dir, "optimized_graph_base.json")
            
            # Descargar Dataset Climático si faltan archivos
            if not os.path.exists(vector_db_path) or not os.path.exists(graph_json_path):
                print(f"   ℹ️  Default Climate Data missing. Downloading...")
                self._download_custom_data(data_dir, DEFAULT_CLIMATE_ID)

        # Validación Final de Rutas
        if not os.path.exists(vector_db_path): raise FileNotFoundError(f"Vector DB not found at: {vector_db_path}")
        if not os.path.exists(graph_json_path): raise FileNotFoundError(f"Graph JSON not found at: {graph_json_path}")
        
        print(f"   ℹ️  Knowledge Base paths verified.")

        # 1. Cargar Topología del Grafo
        print(f"   > Loading Graph Topology...")
        with open(graph_json_path, 'r') as f:
            data = json.load(f)
        self.G = nx.DiGraph()
        for n in data['nodes']: 
            self.G.add_node(n['id'], label=n.get('label', n['id']))
        for e in data['edges']: 
            # Soportar formatos 'source/target' o 'node1/node2'
            src = e.get('source') or e.get('node1')
            dst = e.get('target') or e.get('node2')
            self.G.add_edge(src, dst, id=e['id'], relation=e.get('relation'))

        # 2. Cargar IA (Sentence Transformers)
        print(f"   > Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        model_ref = self.model
        class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                return model_ref.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
        
        # 3. Conectar a ChromaDB
        print(f"   > Connecting to Vector Database...")
        client = chromadb.PersistentClient(path=vector_db_path)
        self.node_coll = client.get_collection("nodes_references_base", embedding_function=LocalEmbeddingFunction())
        self.edge_coll = client.get_collection("edges_evidence_base", embedding_function=LocalEmbeddingFunction())
        print("✅ System Ready.")

    def _download_custom_data(self, target_dir, file_id):
        """
        Descarga datos desde Google Drive y los extrae en el directorio objetivo.
        """
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        # Ubicación neutral para el zip para evitar colisiones
        temp_zip = f"/tmp/graphrag_temp_{file_id}.zip"
        
        try:
            import gdown
        except ImportError:
            print("   📦 Installing 'gdown'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"   ⬇️  Downloading package (ID: {file_id})...")
        gdown.download(url, temp_zip, quiet=False)

        print(f"   📦 Extracting package...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Limpiar el zip temporal
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        print("   ✅ Extraction complete.")

    def _extract_title(self, text):
        try:
            start = text.find("Paper Title:") + 13
            end = text.find("\n", start)
            if start > 12 and end > start: return text[start:end].strip()
        except: pass
        return "Document Snippet"

    def search(self, query, top_k=3, hops=1, context_k=2):
        """
        Búsqueda híbrida (Vectorial + Expansión Temática + Grafo).
        """
        knowledge = {
            "papers": [],
            "graph_links": [],
            "stats": {"primary": 0, "context": 0, "graph": 0}
        }
        seen_docs = set()
        anchor_ids = set()

        # 1. Búsqueda Vectorial Primaria
        results = self.node_coll.query(query_texts=[query], n_results=top_k)
        if results['ids']:
            for i, doc in enumerate(results['documents'][0]):
                if doc in seen_docs: continue
                seen_docs.add(doc)
                meta = results['metadatas'][0][i]
                anchor_ids.add(meta['node_id'])
                
                knowledge["papers"].append({
                    "ref_id": f"REF_PRI_{i+1}",
                    "source_id": meta.get('url', meta.get('doi', 'N/A')),
                    "title": self._extract_title(doc),
                    "year": meta.get('year', 'N/A'),
                    "content": doc.strip(),
                    "type": "PRIMARY",
                    "relevance": "Direct match"
                })
        knowledge["stats"]["primary"] = len(knowledge["papers"])

        # 2. Expansión de Contexto por Nodo
        if context_k > 0 and anchor_ids:
            print(f"      > Expanding context for {len(anchor_ids)} topics...")
            for nid in anchor_ids:
                try:
                    res = self.node_coll.query(query_texts=[query], n_results=context_k+2, where={"node_id": nid})
                    if res['ids']:
                        added_count = 0
                        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                            if doc in seen_docs: continue
                            if added_count >= context_k: break
                            seen_docs.add(doc)
                            added_count += 1
                            knowledge["papers"].append({
                                "ref_id": f"REF_CTX_{nid}_{added_count}",
                                "source_id": meta.get('url', meta.get('doi', 'N/A')),
                                "title": self._extract_title(doc),
                                "year": meta.get('year', 'N/A'),
                                "content": doc.strip(),
                                "type": "CONTEXT",
                                "relevance": f"Context for {nid}"
                            })
                except: pass
        knowledge["stats"]["context"] = len(knowledge["papers"]) - knowledge["stats"]["primary"]

        # 3. Travesía del Grafo
        if hops > 0 and anchor_ids:
            visited_edges = set()
            count = 1
            current_frontier = list(anchor_ids)
            visited_nodes = set(anchor_ids)
            for _ in range(hops):
                next_frontier = []
                for anchor in current_frontier:
                    if anchor not in self.G: continue
                    for neighbor in self.G.successors(anchor):
                        edge_data = self.G.get_edge_data(anchor, neighbor)
                        if edge_data['id'] not in visited_edges:
                            edocs = self.edge_coll.get(where={"edge_id": edge_data['id']}, limit=1)
                            evidence = edocs['documents'][0] if edocs['documents'] else "Structural link."
                            knowledge["graph_links"].append({
                                "graph_id": f"GRAPH_{count}",
                                "node1": anchor,
                                "node2": neighbor,
                                "relation": edge_data.get('relation', 'RELATED'),
                                "evidence": evidence.strip()
                            })
                            visited_edges.add(edge_data['id'])
                            count += 1
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            next_frontier.append(neighbor)
                current_frontier = next_frontier
        knowledge["stats"]["graph"] = len(knowledge["graph_links"])
        return knowledge

    def format_prompt(self, knowledge, query):
        """
        Genera el prompt experto para el LLM.
        """
        if not knowledge["papers"] and not knowledge["graph_links"]:
            return "No evidence found."
        
        papers_block = ""
        for p in knowledge["papers"]:
            papers_block += f"[{p['ref_id']}] | SOURCE: {p['source_id']} | {p['title']} ({p['year']})\nKey excerpt: {p['content']}\n\n"
        
        graph_block = ""
        for g in knowledge["graph_links"]:
            graph_block += f"[{g['graph_id']}] {g['node1']} --[{g['relation']}]--> {g['node2']}\nEvidence: {g['evidence']}\n\n"
        
        return f"""# ROLE
You are a Climate Adaptation Knowledge Synthesizer with expertise in Systems Thinking.

# USER QUESTION
"{query}"

# KNOWLEDGE BASE
## 1. Scientific Literature
{papers_block}

## 2. Structural Context (Graph)
{graph_block}

# INSTRUCTIONS
1. Triangulate data: Identify where literature and graph connections overlap.
2. Causal reasoning: Explain mechanistic pathways (A -> B -> C).
3. Citation: Use [REF_PRI_x] and [GRAPH_x] tags.
"""