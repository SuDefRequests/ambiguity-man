import os
import re
from typing import List, Dict, Any, Optional
import pymupdf as fitz
import chromadb

# --- Paths & Collection Identifiers ---
BAWS_CHROMA_PATH = os.getenv("BAWS_CHROMA_PATH", "./ambedkar_archive_db")
BAWS_COLLECTION_NAME = "ambedkar_writings"

CAD_CHROMA_PATH = os.getenv("CAD_CHROMA_PATH", "./cad_archive_db")
CAD_COLLECTION_NAME = "cad_debates"

# --- Client Initializations ---
# 1. BAWS Client (Writings & Speeches)
baws_client = chromadb.PersistentClient(path=BAWS_CHROMA_PATH)
baws_collection = baws_client.get_or_create_collection(
    name=BAWS_COLLECTION_NAME,
    metadata={"description": "Writings and Speeches of Dr. B. R. Ambedkar"}
)

# 2. CAD Client (Constituent Assembly Debates)
cad_client = chromadb.PersistentClient(path=CAD_CHROMA_PATH)
cad_collection = cad_client.get_or_create_collection(
    name=CAD_COLLECTION_NAME,
    metadata={"description": "Constituent Assembly Debates (Official Transcripts)"}
)


# --- Text Processing Helpers ---
def clean_text(text: str) -> str:
    """Normalize whitespace and strip common digital header/footer artifacts."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Sliding window chunking with overlap to preserve semantic continuity.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap
    return chunks


def _calculate_relevance(distance: float) -> float:
    """
    Normalizes distance to an intuitive 0.0 - 1.0 relevance score.
    Chroma uses Euclidean/L2 distance by default.
    """
    return round(1.0 / (1.0 + distance), 4)


# --- BAWS Ingestion Pipeline ---
def ingest_document(
    file_path: str,
    source_title: str,
    volume: int,
    start_page: int = 0,
    end_page: Optional[int] = None,
    batch_size: int = 100
) -> int:
    """
    Extracts, chunks, and persists a BAWS PDF document into ChromaDB.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    actual_end_page = min(end_page or total_pages, total_pages)

    print(f"[*] Ingesting '{source_title}' (Vol {volume}) | Pages {start_page} to {actual_end_page} of {total_pages}...")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    total_chunks = 0

    for page_idx in range(start_page, actual_end_page):
        page = doc[page_idx]
        raw_text = page.get_text()
        cleaned = clean_text(raw_text)

        if len(cleaned) < 80:
            continue

        page_number = page_idx + 1
        page_chunks = chunk_text(cleaned)

        for chunk_idx, chunk in enumerate(page_chunks):
            chunk_id = f"baws_v{volume}_p{page_number}_c{chunk_idx}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": source_title,
                "volume": volume,
                "page": page_number,
                "chunk_index": chunk_idx,
                "type": "baws"
            })

            if len(ids) >= batch_size:
                baws_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                total_chunks += len(ids)
                ids, documents, metadatas = [], [], []

    if ids:
        baws_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(ids)

    print(f"[✓] Completed '{source_title}': Indexed {total_chunks} chunks.")
    return total_chunks


# --- Search Implementations ---
def search_baws_archive(
    query_text: str,
    n_results: int = 4,
    volume_filter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Searches Dr. Ambedkar's Writings and Speeches (BAWS)."""
    where_filter = {"volume": volume_filter} if volume_filter is not None else None

    query_params: Dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": n_results,
    }
    if where_filter:
        query_params["where"] = where_filter

    raw_results = baws_collection.query(**query_params)
    results = []

    if raw_results and raw_results.get("documents") and raw_results["documents"][0]:
        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0] if raw_results.get("distances") else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            results.append({
                "source": meta.get("source", "BAWS Archive"),
                "archive_type": "baws",
                "volume": meta.get("volume", 0),
                "page": meta.get("page", 0),
                "title": meta.get("source", "BAWS"),
                "url": "",
                "snippet": doc_text,
                "relevance_score": _calculate_relevance(dist),
            })

    return results


def search_cad_archive(
    query_text: str,
    n_results: int = 4,
    volume_filter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Searches the Constituent Assembly Debates (CAD)."""
    where_filter = {"volume": volume_filter} if volume_filter is not None else None

    query_params: Dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": n_results,
    }
    if where_filter:
        query_params["where"] = where_filter

    raw_results = cad_collection.query(**query_params)
    results = []

    if raw_results and raw_results.get("documents") and raw_results["documents"][0]:
        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0] if raw_results.get("distances") else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            results.append({
                "source": meta.get("source", "CAD Archive"),
                "archive_type": "cad",
                "volume": meta.get("volume", 0),
                "page": meta.get("page", meta.get("chunk_idx", 0)),
                "title": meta.get("title", "Debate Session"),
                "url": meta.get("url", ""),
                "snippet": doc_text,
                "relevance_score": _calculate_relevance(dist),
            })

    return results


def search_archive(
    query_text: str,
    n_results: int = 4,
    archive: str = "baws",
    volume_filter: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Unified router for archive search.
    - archive="baws": Searches BAWS collection
    - archive="cad": Searches Constituent Assembly Debates collection
    - archive="all": Searches both and sorts by relevance score
    """
    archive_normalized = archive.lower().strip()

    if archive_normalized == "baws":
        return search_baws_archive(query_text, n_results, volume_filter)
    elif archive_normalized == "cad":
        return search_cad_archive(query_text, n_results, volume_filter)
    elif archive_normalized in ("all", "both"):
        # Split top_k evenly across both collections
        half_k = max(1, n_results // 2)
        baws_res = search_baws_archive(query_text, half_k, volume_filter)
        cad_res = search_cad_archive(query_text, half_k, volume_filter)
        combined = baws_res + cad_res
        combined.sort(key=lambda x: x["relevance_score"], reverse=True)
        return combined[:n_results]
    
    else:
        return search_baws_archive(query_text, n_results, volume_filter)