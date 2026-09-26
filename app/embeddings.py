"""
Embedding + vector store layer.

Why BGE-M3: it is multilingual (English, Hindi, Marathi all score well),
so a Hindi question can retrieve an English-language 1948 debate passage
without a separate translation step before search.

Why Qdrant: simple to self-host, filters on payload fields (speaker,
date, doc_id) work out of the box, and it is fast enough for a kiosk.
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, EMBED_DIM


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Loaded once per process; this is the slow part (~2-4s cold start).
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )


def embed(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    # normalize_embeddings=True lets us use cosine distance safely
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def upsert_passages(passages: list[dict]) -> None:
    """passages: list of dicts with passage_id, text, and payload fields
    (doc_id, speaker, date, page_start, language, article_refs)."""
    ensure_collection()
    client = get_qdrant()
    vectors = embed([p["text"] for p in passages])

    points = [
        qm.PointStruct(
            id=_stable_int_id(p["passage_id"]),
            vector=vec,
            payload={
                "passage_id": p["passage_id"],
                "doc_id": p.get("doc_id"),
                "speaker": p.get("speaker"),
                "date": p.get("date"),
                "page_start": p.get("page_start"),
                "language": p.get("language", "en"),
                "article_refs": p.get("article_refs", []),
            },
        )
        for p, vec in zip(passages, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def vector_search(query: str, top_k: int = 20) -> list[dict]:
    ensure_collection()
    client = get_qdrant()
    qvec = embed([query])[0]
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    return [
        {"passage_id": h.payload["passage_id"], "score": h.score, "payload": h.payload}
        for h in hits
    ]


def _stable_int_id(passage_id: str) -> int:
    # Qdrant point IDs must be int or UUID; derive a stable int from the
    # text passage_id so re-running the loader upserts instead of duplicating.
    import hashlib
    return int(hashlib.sha256(passage_id.encode()).hexdigest(), 16) % (2**63)
