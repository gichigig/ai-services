import os
import json
import numpy as np
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception as e:
    import traceback
    traceback.print_exc()
    faiss = None
    SentenceTransformer = None

DATA_DIR = "/data/rag"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")

class PersistentRAG:
    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        if SentenceTransformer is None or faiss is None:
            raise ImportError("faiss and sentence_transformers are required for RAG")
            
        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load existing index if present
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            print(f"Loading FAISS index from {INDEX_PATH}")
            self.index = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, "r") as f:
                self.documents = json.load(f)
        else:
            print("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
        
    def save(self):
        """Save the index and documents to the persistent volume."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "w") as f:
            json.dump(self.documents, f)
            
    def add_documents(self, docs):
        if not docs:
            return
            
        embeddings = self.embedder.encode(docs, convert_to_numpy=True)
        self.index.add(embeddings)
        self.documents.extend(docs)
        self.save()
        
    def search(self, query, top_k=3):
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(distances[0][i])
                })
        return results

# Initialize a global RAG instance
rag_store = None

def get_rag_store():
    global rag_store
    if rag_store is None:
        rag_store = PersistentRAG()
    return rag_store

def retrieve_context(query: str, top_k: int = 3, history: list = None) -> str:
    store = get_rag_store()
    
    # If we have history, form a contextual query to help FAISS find the right document
    search_query = query
    if history and len(history) > 0:
        # Grab the last user message from history if it exists
        user_msgs = [m["content"] for m in history if m.get("role") == "user"]
        if user_msgs:
            search_query = f"{user_msgs[-1]} {query}"

    results = store.search(search_query, top_k=top_k)
    
    if not results:
        return "No relevant context found."
        
    context_parts = [r["document"] for r in results]
    return "\n".join(context_parts)

async def index_file_url(file_url: str, ext: str) -> int:
    """
    Downloads a file, extracts its text using PyMuPDF (if PDF) or raw text,
    chunks it, and adds to FAISS index.
    """
    import urllib.request
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        urllib.request.urlretrieve(file_url, tmp.name)
        file_path = tmp.name
        
    text = ""
    try:
        if ext == "pdf":
            import fitz # PyMuPDF
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text() + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    finally:
        os.remove(file_path)
        
    if not text.strip():
        return 0
        
    # Chunking by roughly 500 words
    words = text.split()
    chunks = []
    chunk_size = 500
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunk_with_meta = f"Source URL: {file_url}\n\n{chunk_text}"
        chunks.append(chunk_with_meta)
        
    store = get_rag_store()
    store.add_documents(chunks)
    return len(chunks)
