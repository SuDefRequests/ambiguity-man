"""
Two extraction paths, deliberately different:

1. Article/number references -> regex (app.ingest.extract_article_refs).
   Cheap, deterministic, ~100% accurate. No LLM needed for something a
   regular expression already solves perfectly.

2. Relations between people and articles (OPPOSED, SUPPORTED, ...) ->
   LLM, because these require reading comprehension a regex can't do.
   The LLM's output NEVER goes straight into Neo4j — it lands in
   edges_staging as 'pending' and a person (or a reviewer script) must
   approve it first. This is the safety mechanism against a wrong edge
   silently becoming "fact" in a government archive's graph.

Swap `call_llm()` for whatever you deploy (local vLLM/llama.cpp server,
or a hosted API) — the prompt and JSON contract stay the same either way.
"""

import json
import httpx

from app.db import get_conn
from app.config import LLM_API_BASE, LLM_API_KEY

ALLOWED_PREDICATES = {"OPPOSED", "SUPPORTED", "PROPOSED", "DISCUSSES"}

EXTRACTION_PROMPT = """You extract structured relations from a single \
passage of the Constituent Assembly Debates or a speech by Dr. Ambedkar.

Only extract relations between a NAMED PERSON and an Article NUMBER that \
is explicitly present in the text. If the passage contains no such \
relation, return an empty list. Never guess or infer beyond what the \
text states.

Allowed predicates: OPPOSED, SUPPORTED, PROPOSED, DISCUSSES

Return ONLY valid JSON, no prose, no markdown fences, in this shape:
{"edges": [{"person": "<exact name as written>", "predicate": "<one of the allowed predicates>", "article": "<number only, e.g. 17>", "confidence": <0.0-1.0>}]}

Passage:
---
{passage_text}
---
"""


def call_llm(passage_text: str) -> dict:
    """Minimal HTTP call to an OpenAI-compatible /chat/completions endpoint
    (works for a local vLLM server or most hosted APIs). Replace with
    your actual client if you use the Anthropic/OpenAI SDK directly."""
    if not LLM_API_BASE:
        raise RuntimeError("LLM_API_BASE not configured — set it in .env")

    resp = httpx.post(
        f"{LLM_API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {},
        json={
            "messages": [
                {"role": "user", "content": EXTRACTION_PROMPT.format(passage_text=passage_text)}
            ],
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_passage(passage_id: str, speaker_id: str | None, text: str) -> int:
    """Run LLM extraction on one passage, write results to edges_staging.
    Returns the number of edges written."""
    try:
        result = call_llm(text)
    except Exception as e:
        print(f"[extract] skipped {passage_id}: {e}")
        return 0

    edges = [e for e in result.get("edges", []) if e.get("predicate") in ALLOWED_PREDICATES]
    if not edges:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            for e in edges:
                cur.execute(
                    """
                    INSERT INTO edges_staging
                        (subj_id, subj_type, predicate, obj_id, obj_type,
                         passage_id, confidence, extraction_method, status)
                    VALUES (%s, 'Person', %s, %s, 'Article', %s, %s, 'llm', 'pending')
                    """,
                    (
                        speaker_id or _slug(e["person"]),
                        e["predicate"],
                        e["article"],
                        passage_id,
                        e.get("confidence", 0.5),
                    ),
                )
    return len(edges)


def extract_batch(limit: int = 100) -> None:
    """Run extraction on passages that mention at least one Article number
    and have not been processed yet. Keep `limit` small for a hackathon —
    a few hundred well-chosen passages beats extracting the whole corpus."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT passage_id, speaker_id, text
                FROM passages
                WHERE array_length(article_refs, 1) > 0
                  AND passage_id NOT IN (SELECT DISTINCT passage_id FROM edges_staging)
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    total = 0
    for row in rows:
        total += extract_passage(row["passage_id"], row["speaker_id"], row["text"])
    print(f"Extracted {total} candidate edges from {len(rows)} passages -> edges_staging (pending review)")


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


if __name__ == "__main__":
    extract_batch()
