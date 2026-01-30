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
        Descarga y descomprime los datos desde Google Drive, corrigiendo estructuras anidadas.
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
            return

        # Solo descargar si no existe o es muy pequeño (error de descarga previa)
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            print(f"   ⬇️  Downloading embedded data (ID: {GDRIVE_FILE_ID})...")
            gdown.download(url, zip_path, quiet=False)

        # 3. Descomprimir Inteligente (Aplanando rutas)
        print(f"   📦 Extracting data and fixing paths...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                
                # Ignorar directorios vacíos
                if not filename:
                    continue

                # Caso A: El JSON del grafo
                if filename == "optimized_graph_base.json":
                    source = zip_ref.open(member)
                    target_file = os.path.join(target_dir, filename)
                    with open(target_file, "wb") as f:
                        shutil.copyfileobj(source, f)
                
                # Caso B: Base de datos Vectorial
                elif "climate_knowledge_vectordb_base" in member:
                    # Encontrar la parte de la ruta DESPUÉS de 'climate_knowledge_vectordb_base/'
                    parts = member.split("climate_knowledge_vectordb_base/")
                    if len(parts) > 1 and parts[1]:
                        # Reconstruir la ruta relativa interna
                        relative_path = os.path.join("climate_knowledge_vectordb_base", parts[1])
                        final_path = os.path.join(target_dir, relative_path)
                        
                        # Crear subdirectorios necesarios
                        os.makedirs(os.path.dirname(final_path), exist_ok=True)
                        
                        source = zip_ref.open(member)
                        with open(final_path, "wb") as f:
                            shutil.copyfileobj(source, f)
        
        # Limpieza
        # os.remove(zip_path) # Dejamos el zip por si acaso falla algo más
        print("   ✅ Data ready and structured.")

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
                    "relevance": f"HIGH - Direct semantic match"
                })

        knowledge["stats"]["primary"] = len(knowledge["papers"])

        # --- FASE 2: Expansión de Contexto (Node-Centric Semantic) ---
        if context_k > 0 and anchor_ids:
            print(f"      > Expandiendo contexto para {len(anchor_ids)} temas clave (max {context_k} docs extra por tema)...")
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
                                "relevance": f"CONTEXT (High Similarity) - Evidence for concept '{nid}'"
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
                                "node1": anchor,
                                "node2": neighbor,
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

    def format_prompt(self, knowledge, query):
        """
        Convierte el objeto 'knowledge' en el Prompt Experto desarrollado originalmente.
        """
        # Construir bloques de texto dinámicos
        papers_block = ""
        for p in knowledge["papers"]:
            papers_block += (f"[{p['ref_id']}] | SOURCE_ID: {p['source_id']} | {p['title']} ({p['year']})\n"
                             f"Key excerpt: {p['content']}\n" 
                             f"Relevance: {p['relevance']}\n\n")

        graph_block = ""
        for g in knowledge["graph_links"]:
            graph_block += (f"[{g['graph_id']}] {g['node1']} --[{g['relation']}]--> {g['node2']}\n"
                            f"Evidence: {g['evidence']}\n\n")

        # Template Maestro Optimizado (V2)
        prompt = f"""# ROLE
You are a Climate Adaptation Knowledge Synthesizer with expertise in Systems Thinking. You translate complex scientific data into actionable, high-level technical narratives.

# CORE TASK
Synthesize a response to the User Question by triangulating data from Scientific Literature and Graph Topology. You must distinguish between direct evidence and contextual background, and actively identify causal cascades.

# USER QUESTION
"{query}"

# KNOWLEDGE BASE

## DATA SOURCE LEGEND (Read Carefully)
- **[REF_PRI_x]**: PRIMARY SOURCES. These documents are direct semantic matches to the user's query. Give them the highest weight.
- **[REF_CTX_x]**: CONTEXT EXPANSION. These documents were retrieved because they belong to the same topical nodes as the primary hits. Use them to provide breadth, consensus, or background, but prioritize PRI sources for specific claims.
- **[GRAPH_x]**: CAUSAL EVIDENCE. These are validated structural links with associated text snippets proving the relationship.

## 1. Scientific Literature (Papers)
{papers_block}

## 2. Structural Context (Knowledge Graph)
{graph_block}

# SYNTHESIS GUIDELINES

## A. Structural & Causal Analysis (Crucial)
1.  **Chain Detection (Cascades):** Look for transitive patterns in the graph data.
    *   *Example:* If you see `Heat -> Drought` and `Drought -> Food Insecurity`, explicitly analyze the "Heat-driven Food Insecurity" pathway.
    *   Do not just list edges; weave them into causal chains.
2.  **Mechanism Extraction:** Use the 'Evidence' text within [GRAPH_x] to explain *how* A affects B. (e.g., "Heat affects labor NOT just by discomfort, but by inducing physiological heat stress limits...").

## B. Data Triangulation
1.  **Cross-Verification:** Does the graph evidence [GRAPH_x] support the claims in the primary papers [REF_PRI]? Mention where sources converge.
2.  **Handling Contradiction:** If [REF_CTX] suggests a different outcome than [REF_PRI], note this as scientific complexity or regional variation.

## C. Narrative Quality
1.  **Technical Precision:** Use domain-specific terminology (e.g., "Compound Events", "Teleconnections", "Maladaptation").
2.  **Flow:** Avoid "According to paper X...". Instead, make the claim and cite it: "Climate change exacerbates coastal erosion through sea-level rise [REF_PRI_1], a process accelerated by urban subsidence [GRAPH_3]."

# OUTPUT STRUCTURE

Please organize your response as follows:

1.  **Executive Synthesis**: Direct answer to the question, summarizing the key consensus.
2.  **Mechanistic Pathways & Cascades**:
    *   Detail the causal chains identified in the graph (A -> B -> C).
    *   Explain the mechanisms using the evidence snippets.
3.  **Evidence Deep Dive**:
    *   Synthesize findings from Primary Sources [REF_PRI].
    *   Enrich with Context [REF_CTX].
4.  **Critical Gaps/Uncertainties**: What is missing or uncertain based on the retrieved data?
5.  **References**: List full citations at the end.

**IMPORTANT:** You are writing for a policy/scientific audience. Be rigorous.
"""
        return prompt
