from __future__ import annotations
import os
import pickle
import numpy as np
from pathlib import Path

KNOWLEDGE_DIR = Path("knowledge_base")
EMBEDDINGS_FILE = Path("data/rag_embeddings.pkl")

# Cache model and index in memory for fast querying
_model = None
_index = None

def chunk_document(filepath: Path) -> list[dict]:
    """
    Splits a markdown document into clean section-level chunks by looking for heading indicators.
    Prepends the product name and section title to each chunk's text to preserve semantic context.
    """
    text = filepath.read_text(encoding="utf-8")
    product_name = ""
    
    # Extract product name from the first line (# Heading)
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        product_name = lines[0][2:].strip()
        
    # Split by "## " headings
    sections = text.split("\n## ")
    chunks = []
    
    # First section contains document title, note, and introduction/overview
    intro_text = sections[0].strip()
    chunks.append({
        "product_name": product_name,
        "section_name": "Overview",
        "text": intro_text,
        "source_file": filepath.name
    })
    
    for section in sections[1:]:
        sec_lines = section.split("\n")
        section_name = sec_lines[0].strip()
        section_text = "\n".join(sec_lines[1:]).strip()
        
        # Prepend product context to help the embedding model search more accurately
        chunk_content = f"{product_name} — {section_name}\n{section_text}"
        chunks.append({
            "product_name": product_name,
            "section_name": section_name,
            "text": chunk_content,
            "source_file": filepath.name
        })
        
    return chunks

def build_rag_index():
    """
    Reads all markdown product documents in knowledge_base/, generates local embeddings,
    and stores them in a serialized pickle file for lightweight vector search.
    """
    print("[RAG Engine] Building vector index from documents in knowledge_base/...")
    if not KNOWLEDGE_DIR.exists():
        print(f"[RAG Engine] Error: Directory '{KNOWLEDGE_DIR}' does not exist.")
        return
        
    all_chunks = []
    for filepath in KNOWLEDGE_DIR.glob("*.md"):
        if filepath.name.lower() == "readme.md":
            continue
        try:
            chunks = chunk_document(filepath)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[RAG Engine] Error reading {filepath}: {e}")
            
    if not all_chunks:
        print("[RAG Engine] Error: No document chunks found to index.")
        return
        
    print(f"[RAG Engine] Found {len(all_chunks)} chunks to embed.")
    
    # Load model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate embeddings
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save to pickle file
    EMBEDDINGS_FILE.parent.mkdir(exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({
            "chunks": all_chunks,
            "embeddings": embeddings
        }, f)
        
    print(f"[RAG Engine] Index built successfully and saved to {EMBEDDINGS_FILE}")

def load_rag_index() -> tuple[SentenceTransformer, dict]:
    """
    Helper to lazily load/initialize the embedding model and vector index into memory.
    If the index pickle file doesn't exist on disk, it triggers an index rebuild.
    """
    global _index, _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        if not EMBEDDINGS_FILE.exists():
            build_rag_index()
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "rb") as f:
                _index = pickle.load(f)
        else:
            raise FileNotFoundError(f"Embeddings file {EMBEDDINGS_FILE} could not be loaded or created.")
    return _model, _index

def _keyword_search(query: str, chunks: list, top_k: int = 3) -> list[dict]:
    """Fallback keyword search when model unavailable."""
    query_words = set(query.lower().split())
    scores = []
    for i, chunk in enumerate(chunks):
        text_words = set(chunk["text"].lower().split())
        score = len(query_words & text_words) / (len(query_words) + 1)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [{"chunk": chunks[i], "score": s} for s, i in scores[:top_k]]

def search_product_policy(query: str, top_k: int = 3) -> list[dict]:
    # Skip RAG entirely if disabled (e.g. low-memory environments)
    if os.environ.get("DISABLE_RAG") == "1":
        return []
    """
    Performs semantic search across the product knowledge base using cosine similarity.
    Returns the top-k matches with similarity scores.
    """
    try:
        model, index = load_rag_index()
    except Exception:
        if _index is not None:
            return _keyword_search(query, _index["chunks"], top_k)
        return []
    chunks = index["chunks"]
    embeddings = index["embeddings"]
    
    # Embed the query
    query_emb = model.encode(query, convert_to_numpy=True)
    
    # Compute cosine similarities manually: np.dot(A, B) / (norm(A) * norm(B))
    # Normalize query embedding
    norm_query_emb = query_emb / np.linalg.norm(query_emb)
    # Normalize index embeddings
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    similarities = np.dot(norm_embeddings, norm_query_emb)
    
    # Get top_k indices sorted descending by similarity score
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        chunk = chunks[idx]
        results.append({
            "product_name": chunk["product_name"],
            "section_name": chunk["section_name"],
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "similarity_score": score
        })
        
    return results

if __name__ == "__main__":
    # If run directly, build/rebuild the index
    build_rag_index()
