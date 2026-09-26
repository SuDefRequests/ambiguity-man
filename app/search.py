"""
Hybrid retrieval: BM25 catches exact wording (crucial for quotes and
Article numbers), vector search catches paraphrase and cross-language
meaning. Neither alone is enough — see the design notes below each
function. Reciprocal Rank Fusion (RRF) merges the two ranked lists
without needing to normalize incomparable score scales.
"""

from app.db import get_conn
from app.embeddings import vector_search


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """Exact / near-exact word match via Postgres full-text search.
    Why: embeddings blur exact wording, so a search for a specific phrase
    ("fraternity assuring the dignity of the individual") is far more
    reliable here than in vector space."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.passage_id, p.text, p.page_start, d.doc_date AS date,
                       s.canonical_name AS speaker,
                       ts_rank_cd(p.tsv, plainto_tsquery('english', %(q)s)) AS score
                FROM passages p
                LEFT JOIN speakers s ON s.speaker_id = p.speaker_id
                LEFT JOIN documents d ON d.doc_id = p.doc_id
                WHERE p.tsv @@ plainto_tsquery('english', %(q)s)
                ORDER BY score DESC
                LIMIT %(k)s
                """,
                {"q": query, "k": top_k},
            )
            return cur.fetchall()


def hybrid_search(query: str, top_k: int = 8) -> list[dict]:
    """Merge BM25 + vector results with Reciprocal Rank Fusion (RRF).

    RRF score for a passage = sum over each ranked list it appears in of
    1 / (k + rank). This is preferred over averaging raw scores because
    BM25 scores and cosine similarities live on different, incomparable
    scales — RRF only needs *rank position*, which is always comparable.
    """
    RRF_K = 60  # standard smoothing constant from the RRF literature

    bm25_hits = bm25_search(query, top_k=20)
    vec_hits = vector_search(query, top_k=20)

    fused: dict[str, dict] = {}

    for rank, row in enumerate(bm25_hits):
        pid = row["passage_id"]
        fused.setdefault(pid, {"passage_id": pid, "rrf": 0.0, "via": set()})
        fused[pid]["rrf"] += 1.0 / (RRF_K + rank + 1)
        fused[pid]["via"].add("bm25")

    for rank, hit in enumerate(vec_hits):
        pid = hit["passage_id"]
        fused.setdefault(pid, {"passage_id": pid, "rrf": 0.0, "via": set()})
        fused[pid]["rrf"] += 1.0 / (RRF_K + rank + 1)
        fused[pid]["via"].add("vector")

    ranked_ids = sorted(fused.values(), key=lambda r: r["rrf"], reverse=True)[:top_k]

    # Hydrate full passage rows from Postgres (the source of truth) —
    # never trust vector-store payloads for anything beyond search filters.
    return _hydrate(ranked_ids)


def graph_expand(passage_ids: list[str], article_number: str | None) -> list[dict]:
    """Pull in passages connected via the graph (e.g. speeches by people
    who OPPOSED/SUPPORTED the same Article) that pure text search would
    likely miss because they don't share vocabulary with the query."""
    if not article_number:
        return []
    from app.graph.query import passages_for_article
    extra_ids = passages_for_article(article_number)
    return _hydrate([{"passage_id": pid, "rrf": 0.0, "via": {"graph"}} for pid in extra_ids])


def _hydrate(ranked: list[dict]) -> list[dict]:
    if not ranked:
        return []
    ids = [r["passage_id"] for r in ranked]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.passage_id, p.text, p.page_start,
                       s.canonical_name AS speaker, d.doc_date AS date
                FROM passages p
                LEFT JOIN speakers s ON s.speaker_id = p.speaker_id
                LEFT JOIN documents d ON d.doc_id = p.doc_id
                WHERE p.passage_id = ANY(%s)
                """,
                (ids,),
            )
            rows = {r["passage_id"]: r for r in cur.fetchall()}

    out = []
    for r in ranked:
        base = rows.get(r["passage_id"])
        if not base:
            continue
        out.append(
            {
                **base,
                "score": round(r["rrf"], 4),
                "via": "+".join(sorted(r["via"])) if r["via"] else "graph",
            }
        )
    return out
