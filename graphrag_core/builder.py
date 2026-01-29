import json
import hashlib
import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class GraphBuilder:
    def __init__(self, output_vector_db_path, model_name="BAAI/bge-base-en-v1.5", device=None):
        self.db_path = output_vector_db_path
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🏗️  Initializing GraphBuilder on {self.device.upper()}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Wrapper Chroma
        model_ref = self.model
        class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                return model_ref.encode(input, normalize_embeddings=True, show_progress_bar=False).tolist()
        
        self.embedding_fn = LocalEmbeddingFunction()

    def build(self, input_json_path, output_json_skeleton_path, reset_db=True):
        """
        Ingesta un grafo JSON agnóstico.
        Requisitos JSON:
          - nodes: [{id, label, properties: {references: [{text, ...}]}}]
          - edges: [{source, target, relation, evidence: {snippet, ...}}]
        """
        if reset_db and os.path.exists(self.db_path):
            print("   ♻️  Resetting Vector DB...")
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)

        client = chromadb.PersistentClient(path=self.db_path)
        node_coll = client.create_collection("nodes_references_base", embedding_function=self.embedding_fn)
        edge_coll = client.create_collection("edges_evidence_base", embedding_function=self.embedding_fn)

        print(f"   📂 Reading {input_json_path}...")
        with open(input_json_path, 'r') as f:
            raw_data = json.load(f)

        # 1. Procesar Nodos
        vector_nodes = []
        skeleton_nodes = []
        
        for n in tqdm(raw_data.get('nodes', []), desc="Ingesting Nodes"):
            # Guardamos estructura ligera
            skeleton_nodes.append({
                "id": n['id'],
                "label": n.get('label', n['id']),
                "type": n.get('type', 'undefined')
            })

            # Extraemos texto de 'references' o cualquier campo de texto lista
            # (Lógica adaptada para ser flexible pero compatible con tu grafo actual)
            refs = n.get('properties', {}).get('references', [])
            if isinstance(refs, list):
                for i, ref in enumerate(refs):
                    text = ref.get('text', '')
                    if not text: continue
                    
                    # Flatten metadata
                    meta = {k: str(v) for k, v in ref.items() if k != 'text'}
                    meta['node_id'] = n['id']
                    
                    # ID único
                    vec_id = f"{n['id']}_ref_{i}"
                    vector_nodes.append((text, meta, vec_id))

        if vector_nodes:
            self._batch_add(node_coll, vector_nodes)

        # 2. Procesar Aristas
        vector_edges = []
        skeleton_edges = []
        
        for e in tqdm(raw_data.get('edges', []), desc="Ingesting Edges"):
            edge_id = hashlib.md5(f"{e['source']}-{e.get('relation')}-{e['target']}".encode()).hexdigest()
            
            skeleton_edges.append({
                "id": edge_id,
                "source": e['source'],
                "target": e['target'],
                "relation": e.get('relation', 'RELATED')
            })

            evidence = e.get('evidence', {})
            text = evidence.get('snippet', '') or evidence.get('justification', '')
            
            if text:
                meta = {k: str(v) for k, v in evidence.items() if k not in ['snippet', 'justification']}
                meta['edge_id'] = edge_id
                meta['relation'] = e.get('relation', 'RELATED')
                
                vec_id = f"{edge_id}_ev"
                vector_edges.append((text, meta, vec_id))

        if vector_edges:
            self._batch_add(edge_coll, vector_edges)

        # 3. Guardar Esqueleto
        skeleton = {"nodes": skeleton_nodes, "edges": skeleton_edges}
        with open(output_json_skeleton_path, 'w') as f:
            json.dump(skeleton, f, indent=2)
            
        print(f"✅ Build Complete. Skeleton saved to {output_json_skeleton_path}")

    def _batch_add(self, collection, data, batch_size=32):
        total = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            collection.add(
                documents=[b[0] for b in batch],
                metadatas=[b[1] for b in batch],
                ids=[b[2] for b in batch]
            )
