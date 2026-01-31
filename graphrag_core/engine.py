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
        
        # 1. DETERMINAR ID Y RUTAS
        target_id = gdrive_id if gdrive_id else DEFAULT_CLIMATE_ID
        is_custom = target_id != DEFAULT_CLIMATE_ID
        
        # Si es custom y no hay rutas, usamos /tmp para evitar problemas de permisos
        if is_custom and vector_db_path is None:
            work_dir = os.path.join("/tmp", f"graphrag_{target_id}")
        else:
            work_dir = data_dir

        if vector_db_path is None:
            vector_db_path = os.path.join(work_dir, "climate_knowledge_vectordb_base")
        if graph_json_path is None:
            graph_json_path = os.path.join(work_dir, "optimized_graph_base.json")

        # 2. CARGA O DESCARGA
        # Si los archivos no existen, disparamos la descarga
        if not os.path.exists(vector_db_path) or not os.path.exists(graph_json_path):
            print(f"   ℹ️  Data missing at {work_dir}. Downloading (ID: {target_id})...")
            self._download_and_extract(work_dir, target_id)
            
            # Tras descargar un CUSTOM, los nombres pueden ser distintos, así que escaneamos
            if is_custom:
                print(f"   🔍 Scanning for custom files in {work_dir}...")
                jsons = glob.glob(os.path.join(work_dir, "**", "*.json"), recursive=True)
                sqlites = glob.glob(os.path.join(work_dir, "**", "chroma.sqlite3"), recursive=True)
                if jsons and sqlites:
                    graph_json_path = jsons[0]
                    vector_db_path = os.path.dirname(sqlites[0])
                    print(f"   ✅ Discovered: {os.path.basename(graph_json_path)}")

        # 3. VALIDACIÓN FINAL
        if not os.path.exists(vector_db_path): raise FileNotFoundError(f"Vector DB not found at: {vector_db_path}")
        if not os.path.exists(graph_json_path): raise FileNotFoundError(f"Graph JSON not found at: {graph_json_path}")
        
        print(f"   ℹ️  Using Graph Data from: {os.path.dirname(graph_json_path)}")

        # 4. CARGAR COMPONENTES
        with open(graph_json_path, 'r') as f:
            data = json.load(f)
        self.G = nx.DiGraph()
        for n in data['nodes']: 
            # Store full node metadata
            self.G.add_node(n['id'], **n)
        for e in data['edges']: 
            src = e.get('source') or e.get('node1')
            dst = e.get('target') or e.get('node2')
            self.G.add_edge(src, dst, id=e['id'], relation=e.get('relation'))
            
        self.model = SentenceTransformer(model_name, device=self.device)
        model_ref = self.model
        class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                return model_ref.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
        
        client = chromadb.PersistentClient(path=vector_db_path)
        self.node_coll = client.get_collection("nodes_references_base", embedding_function=LocalEmbeddingFunction())
        self.edge_coll = client.get_collection("edges_evidence_base", embedding_function=LocalEmbeddingFunction())
        print("✅ System Ready.")

    def _download_and_extract(self, target_dir, file_id):
        os.makedirs(target_dir, exist_ok=True)
        temp_zip = f"/tmp/temp_graph_{file_id}.zip"
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, temp_zip, quiet=False)

        print(f"   📦 Extracting and flattening...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename: continue
                
                # Lógica de aplanado: si es el json o parte de la DB, lo ponemos en la raíz de target_dir o su carpeta
                if filename.endswith(".json") or "optimized_graph" in filename:
                    with zip_ref.open(member) as source, open(os.path.join(target_dir, filename), "wb") as target:
                        shutil.copyfileobj(source, target)
                elif "climate_knowledge_vectordb_base" in member or "test_db" in member:
                    # Preservar estructura interna de la DB pero aplanar la raíz
                    db_root = "climate_knowledge_vectordb_base" if "climate_knowledge_vectordb_base" in member else "test_db"
                    parts = member.split(f"{db_root}/")
                    if len(parts) > 1 and parts[1]:
                        final_path = os.path.join(target_dir, db_root, parts[1])
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        with zip_ref.open(member) as source, open(final_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                else:
                    # Otros archivos (README, etc) a la raíz
                    with zip_ref.open(member) as source, open(os.path.join(target_dir, filename), "wb") as target:
                        shutil.copyfileobj(source, target)
        
        if os.path.exists(temp_zip): os.remove(temp_zip)

    def search(self, query, top_k=3, hops=1, context_k=2):
        knowledge = {"papers": [], "graph_links": [], "stats": {"primary": 0, "context": 0, "graph": 0}}
        seen_docs = set()
        anchor_ids = set()

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

        if context_k > 0 and anchor_ids:
            print(f"      > Expanding context for {len(anchor_ids)} topics...")
            for nid in anchor_ids:
                try:
                    res = self.node_coll.query(query_texts=[query], n_results=context_k+2, where={"node_id": nid})
                    if res['ids']:
                        added = 0
                        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                            if doc in seen_docs: continue
                            if added >= context_k: break
                            seen_docs.add(doc)
                            added += 1
                            knowledge["papers"].append({
                                "ref_id": f"REF_CTX_{nid}_{added}",
                                "source_id": meta.get('url', meta.get('doi', 'N/A')),
                                "title": self._extract_title(doc),
                                "year": meta.get('year', 'N/A'),
                                "content": doc.strip(),
                                "type": "CONTEXT",
                                "relevance": f"Context for {nid}"
                            })
                except: pass
        knowledge["stats"]["context"] = len(knowledge["papers"]) - knowledge["stats"]["primary"]

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
                            
                            # Retrieve metadata for tooltip
                            n1_data = self.G.nodes[anchor]
                            n2_data = self.G.nodes[neighbor]
                            
                            knowledge["graph_links"].append({
                                "graph_id": f"GRAPH_{count}",
                                "node1": anchor,
                                "node2": neighbor,
                                "relation": edge_data.get('relation', 'RELATED'),
                                "evidence": evidence.strip(),
                                "source_metadata": n1_data,
                                "target_metadata": n2_data
                            })
                            visited_edges.add(edge_data['id'])
                            count += 1
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            next_frontier.append(neighbor)
                current_frontier = next_frontier
        knowledge["stats"]["graph"] = len(knowledge["graph_links"])
        return knowledge

    def _extract_title(self, text):
        try:
            start = text.find("Paper Title:") + 13
            end = text.find("\n", start)
            if start > 12 and end > start: return text[start:end].strip()
        except: pass
        return "Document Snippet"

    def format_prompt(self, knowledge, query, role=None, instructions=None, template=None):
        if not knowledge["papers"] and not knowledge["graph_links"]: return "No evidence found."

        # Defaults
        role = role if role else "You are a Climate Adaptation Knowledge Synthesizer with expertise in Systems Thinking."
        instructions = instructions if instructions else "1. Triangulate data from both sections.\n2. Identify causal chains.\n3. Cite using [REF_PRI_x] and [GRAPH_x]."

        # Build Blocks
        papers_block = ""
        for p in knowledge["papers"]:
            papers_block += f"[{p['ref_id']}] | SOURCE: {p['source_id']} | {p['title']} ({p['year']})\nKey excerpt: {p['content']}\n\n"
        
        graph_block = ""
        for g in knowledge["graph_links"]:
            graph_block += f"[{g['graph_id']}] {g['node1']} --[{g['relation']}]--> {g['node2']}\nEvidence: {g['evidence']}\n\n"
            
        # Use Custom or Default Template
        if template:
            try:
                return template.format(role=role, query=query, papers_block=papers_block, graph_block=graph_block, instructions=instructions)
            except KeyError as e:
                return f"Error in template formatting: Missing key {e}"
        
        return f"# ROLE\n{role}\n\n# USER QUESTION\n\"{query}\"\n\n# DATA\n## 1. Literature\n{papers_block}## 2. Graph\n{graph_block}\n# INSTRUCTIONS\n{instructions}\n"

    def visualize(self, results, title="Knowledge Subgraph", height="600px", dark_mode=True):
        """
        Visualizes the search results as an interactive graph.
        """
        try:
            from .visualizer import visualize_results
            return visualize_results(results, title=title, height=height, dark_mode=dark_mode)
        except ImportError:
            print("❌ PyVis not installed. Please install it via pip or setup.py.")
            return None
        except Exception as e:
            print(f"❌ Visualization Error: {e}")
            return None
