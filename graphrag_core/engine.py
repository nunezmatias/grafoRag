import json
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
import os
import random

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

# --- CONFIGURACIÓN ---
# IMPORTANTE: Reemplaza este ID con el ID real de tu archivo en Google Drive
GDRIVE_FILE_ID = "1jxCFQ9yxAE8IvYlFvRJkHUVemeSJXS1T" 

class GraphRAGEngine:
    def __init__(self, vector_db_path=None, graph_json_path=None, model_name="BAAI/bge-base-en-v1.5", device=None):
        """
        Inicializa el motor GraphRAG.
        Si no se proveen rutas, intenta usar/descargar el DATASET CLIMÁTICO interno.
        """
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Initializing GraphRAG Engine on {self.device.upper()}...")
        
        # 0. Resolución de Rutas Automática
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        if vector_db_path is None:
            vector_db_path = os.path.join(data_dir, "climate_knowledge_vectordb_base")
            
        if graph_json_path is None:
            graph_json_path = os.path.join(data_dir, "optimized_graph_base.json")

        # AUTO-DESCARGA: Si las rutas por defecto no existen, intentamos bajar los datos
        if not os.path.exists(vector_db_path) or not os.path.exists(graph_json_path):
            print(f"   ℹ️  Default data not found locally. Attempting to download from Google Drive...")
            self._download_default_data(data_dir)

        # Validar existencia final
        if not os.path.exists(vector_db_path) or not os.path.exists(graph_json_path):
            raise FileNotFoundError(f"Data files missing! Please ensure 'climate_data.zip' is downloaded and extracted to: {data_dir}")
        
        print(f"   ℹ️  Using Graph Data from: {data_dir}")

        # 1. Cargar Topología
        print(f"   > Loading Graph Topology from: {os.path.basename(graph_json_path)}")
        with open(graph_json_path, 'r') as f:
            data = json.load(f)
        self.G = nx.DiGraph()
        for n in data['nodes']: 
            self.G.add_node(n['id'], label=n.get('label', n['id']))
        for e in data['edges']: 
            self.G.add_edge(e['source'], e['target'], id=e['id'], relation=e.get('relation'))

        # 2. Cargar IA
        print(f"   > Loading Embedding Model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Wrapper para Chroma
        model_ref = self.model
        class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                return model_ref.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
        
        # 3. Conectar a Chroma
        print(f"   > Connecting to Vector DB at: {os.path.basename(vector_db_path)}")
        client = chromadb.PersistentClient(path=vector_db_path)
        self.node_coll = client.get_collection("nodes_references_base", embedding_function=LocalEmbeddingFunction())
        self.edge_coll = client.get_collection("edges_evidence_base", embedding_function=LocalEmbeddingFunction())
        print("✅ System Ready.")

    def _download_default_data(self, target_dir):
        """
        Descarga y descomprime los datos desde Google Drive.
        """
        # Asegurar directorio
        os.makedirs(target_dir, exist_ok=True)
        zip_path = os.path.join(target_dir, "climate_data.zip")
        
        # 1. Instalar gdown si no existe
        try:
            import gdown
        except ImportError:
            print("   📦 Installing 'gdown' for file download...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        # 2. Descargar
        if GDRIVE_FILE_ID == "YOUR_REAL_FILE_ID_FROM_GOOGLE_DRIVE":
            print("   ⚠️  WARNING: You haven't set the Real File ID in engine.py yet!")
            print("   Please download the data manually or update GDRIVE_FILE_ID in the code.")
            return

        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        print(f"   ⬇️  Downloading embedded data (ID: {GDRIVE_FILE_ID})...")
        gdown.download(url, zip_path, quiet=False)

        # 3. Descomprimir
        print(f"   📦 Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Asumiendo que el zip tiene la estructura graphRagClima/graphrag_core/data/...
            # Ajustamos para extraer en target_dir plano
            for member in zip_ref.namelist():
                # Eliminamos los directorios padres del path del zip para aplanar
                filename = os.path.basename(member)
                # Skip directories
                if not filename: continue
                
                # Copiar archivo
                source = zip_ref.open(member)
                target = open(os.path.join(target_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
        
        # Limpieza
        os.remove(zip_path)
        print("   ✅ Data ready.")

    def _extract_title(self, text):
        # Heurística simple para extraer títulos si siguen el formato "Paper Title: ..."
        try:
            start = text.find("Paper Title:") + 13
            end = text.find("\n", start)
            if start > 12 and end > start: return text[start:end].strip()
        except: pass
        return "Document Snippet"

    def search(self, query, top_k=3, hops=1, context_k=2):
        """
        Ejecuta la búsqueda completa (Vectorial + Contexto Semántico + Grafo).
        Devuelve un diccionario estructurado con los datos crudos.
        """
        knowledge = {
            "papers": [],
            "graph_links": [],
            "stats": {"primary": 0, "context": 0, "graph": 0}
        }
        
        seen_docs = set()
        anchor_ids = set()

        # --- FASE 1: Búsqueda Vectorial (Primary Hits) ---
        results = self.node_coll.query(query_texts=[query], n_results=top_k)
        
        if results['ids']:
            for i, doc in enumerate(results['documents'][0]):
                if doc in seen_docs: continue
                seen_docs.add(doc)
                
                meta = results['metadatas'][0][i]
                node_id = meta['node_id']
                anchor_ids.add(node_id)
                
                # Extracción agnóstica de metadatos (si existen)
                url = meta.get('url', 'N/A')
                doi = meta.get('doi', 'N/A')
                source_id = meta.get('source_doc_id', 'N/A')
                human_id = url if url != 'N/A' else (doi if doi != 'N/A' else source_id)
                title = self._extract_title(doc)
                
                knowledge["papers"].append({
                    "ref_id": f"REF_PRI_{i+1}",
                    "source_id": human_id,
                    "title": title,
                    "year": meta.get('year', 'N/A'),
                    "content": doc.strip(),
                    "type": "PRIMARY",
                    "relevance_note": f"Direct semantic match"
                })

        knowledge["stats"]["primary"] = len(knowledge["papers"])

        # --- FASE 2: Expansión de Contexto (Node-Centric Semantic) ---
        if context_k > 0 and anchor_ids:
            for nid in anchor_ids:
                fetch_limit = context_k + 5 # Buffer para duplicados
                try:
                    extra_res = self.node_coll.query(
                        query_texts=[query],
                        n_results=fetch_limit,
                        where={"node_id": nid}
                    )
                    
                    if extra_res['ids']:
                        docs_list = extra_res['documents'][0]
                        metas_list = extra_res['metadatas'][0]
                        
                        added_count = 0
                        for doc, meta in zip(docs_list, metas_list):
                            if doc in seen_docs: continue
                            if added_count >= context_k: break 
                            
                            seen_docs.add(doc)
                            added_count += 1
                            
                            url = meta.get('url', 'N/A')
                            human_id = url if url != 'N/A' else meta.get('doi', 'N/A')
                            title = self._extract_title(doc)
                            
                            knowledge["papers"].append({
                                "ref_id": f"REF_CTX_{nid}_{added_count}",
                                "source_id": human_id,
                                "title": title,
                                "year": meta.get('year', 'N/A'),
                                "content": doc.strip(),
                                "type": "CONTEXT",
                                "relevance_note": f"Contextual evidence for node '{nid}'"
                            })
                except Exception:
                    pass # Si falla un nodo, seguimos

        knowledge["stats"]["context"] = len(knowledge["papers"]) - knowledge["stats"]["primary"]

        # --- FASE 3: Travesía de Grafo (Causal Hops) ---
        if hops > 0 and anchor_ids:
            visited_edges = set()
            count = 1
            
            # BFS limitado por hops
            current_frontier = list(anchor_ids)
            visited_nodes = set(anchor_ids)

            for _ in range(hops):
                next_frontier = []
                for anchor in current_frontier:
                    neighbors = list(self.G.successors(anchor))
                    for neighbor in neighbors:
                        # Recuperar arista
                        edge_data = self.G.get_edge_data(anchor, neighbor)
                        edge_id = edge_data.get('id')
                        
                        if edge_id and edge_id not in visited_edges:
                            # Buscar evidencia de la arista en VectorDB
                            edocs = self.edge_coll.get(where={"edge_id": edge_id}, limit=1)
                            evidence_text = "Relation defined in graph topology."
                            if edocs['documents']:
                                evidence_text = edocs['documents'][0]

                            rel = edge_data.get('relation', 'RELATED')
                            
                            knowledge["graph_links"].append({
                                "graph_id": f"GRAPH_{count}",
                                "source": anchor,
                                "target": neighbor,
                                "relation": rel,
                                "evidence": evidence_text.strip()
                            })
                            visited_edges.add(edge_id)
                            count += 1
                        
                        if neighbor not in visited_nodes:
                            visited_nodes.add(neighbor)
                            next_frontier.append(neighbor)
                current_frontier = next_frontier

        knowledge["stats"]["graph"] = len(knowledge["graph_links"])
        return knowledge

    def format_prompt(self, knowledge, query, system_role="Expert Analyst", custom_instructions=None):
        """
        Convierte el objeto 'knowledge' en un Prompt formateado para LLM.
        Agnóstico al dominio: Usa 'system_role' para definir el tono.
        """
        
        # 1. Formatear Papers
        papers_block = ""
        for p in knowledge["papers"]:
            tag = "[PRIMARY]" if p['type'] == "PRIMARY" else "[CONTEXT]"
            papers_block += (f"[{p['ref_id']}] {tag} Title: {p['title']} ({p['year']})\n"
                             f"Source: {p['source_id']}\n"
                             f"Content: {p['content']}\n\n")

        # 2. Formatear Grafo
        graph_block = ""
        for g in knowledge["graph_links"]:
            graph_block += (f"[{g['graph_id']}] {g['source']} --[{g['relation']}]--> {g['target']}\n"
                            f"Evidence: {g['evidence']}\n\n")

        # 3. Instrucciones por defecto (Si no se dan custom)
        if not custom_instructions:
            custom_instructions = """
            1. Analyze the Primary Sources to answer the core question.
            2. Use Context Sources to broaden the perspective.
            3. Use Graph Links to identify causal chains and systemic effects.
            4. Triangulate information: Do graph links support the text evidence?
            """

        # 4. Ensamblar Prompt
        prompt = f"""# SYSTEM ROLE
{system_role}

# USER QUERY
"{query}"

# RETRIEVED KNOWLEDGE BASE

## A. Textual Evidence (Scientific/Domain Documents)
{papers_block}

## B. Structural Evidence (Knowledge Graph Topology)
{graph_block}

# INSTRUCTIONS
{custom_instructions}

# OUTPUT FORMAT
Provide a comprehensive, evidence-based response citing specific references (e.g., [REF_PRI_1], [GRAPH_3]).
"""
        return prompt
