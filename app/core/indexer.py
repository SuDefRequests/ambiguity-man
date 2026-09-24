import pymupdf as fitz
import chromadb
import re

PDF_FILE = "baws_vol_1.pdf"
CHROMA_PATH = "./ambedkar_archive_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="ambedkar_writings")

def search_archive(query_text: str, n_results: int = 4):
    """Performs semantic vector search across the historical archive."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    matches = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [[0] * len(docs)])[0]
        
        for doc, meta, dist in zip(docs, metas, distances):
            matches.append({
                "source": meta["source"],
                "page": meta["page"],
                "volume": meta["volume"],
                "snippet": doc,
                "relevance_score": round(1 - dist, 4) if dist else 1.0
            })
    return matches