import difflib
from typing import Dict, Any

async def verify_quote(pool, qdrant_client, collection_name: str, embedding_model, quote: str) -> Dict[str, Any]:
    """
    Cascades through exact, fuzzy, and semantic matching to verify a quote.
    Returns a strict verdict: 'verified', 'paraphrase', or 'not_found'.
    """

    # 1. Exact Match (Postgres)
    async with pool.acquire() as conn:
        # Check if the exact string exists within any passage
        exact_query = """
            SELECT passage_id, text, speaker, date, page
            FROM passages
            WHERE text ILIKE $1
            LIMIT 1
        """
        exact_hit = await conn.fetchrow(exact_query, f"%{quote}%")

        if exact_hit:
            return {
                "verdict": "verified",
                "match_type": "exact",
                "passage": dict(exact_hit)
            }

    # 2. Fuzzy Match (Postgres + Python)
    # Pull candidate passages using a fast full-text search, then check edit distance
    async with pool.acquire() as conn:
        candidates_query = """
            SELECT passage_id, text, speaker, date, page
            FROM passages
            WHERE search_vector @@ plainto_tsquery('english', $1)
            LIMIT 10
        """
        candidates = await conn.fetch(candidates_query, quote)

        for row in candidates:
            # difflib uses a SequenceMatcher algorithm to calculate string similarity
            similarity = difflib.SequenceMatcher(None, quote.lower(), row['text'].lower()).ratio()

            if similarity > 0.85: # High threshold to prevent false positives
                return {
                    "verdict": "verified",
                    "match_type": "fuzzy",
                    "passage": dict(row),
                    "similarity_score": round(similarity, 2)
                }

    # 3. Semantic Match (Qdrant Vectors)
    # Generate the vector for the user's quote
    quote_vector = embedding_model.encode(quote).tolist()

    semantic_hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=quote_vector,
        limit=1
    )

    if semantic_hits and semantic_hits[0].score > 0.75:
        best_hit = semantic_hits[0]
        return {
            "verdict": "paraphrase",
            "match_type": "semantic",
            "passage": {
                "passage_id": best_hit.payload.get("passage_id"),
                "text": best_hit.payload.get("text"),
                "speaker": best_hit.payload.get("speaker"),
                "date": best_hit.payload.get("date"),
                "page": best_hit.payload.get("page")
            },
            "similarity_score": round(best_hit.score, 2)
        }

    # 4. Fallback
    return {
        "verdict": "not_found",
        "match_type": "none",
        "passage": None
    }
