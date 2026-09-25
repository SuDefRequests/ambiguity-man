import os
import time
import hashlib
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError
from app.core.indexer import search_archive

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

groq_client = Groq(api_key=groq_api_key)

# In-memory TTL cache with maximum size ceiling
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_MAX_CACHE_SIZE = 500


def _cache_key(query: str, top_k: int, archive: str, volume_filter: Optional[int]) -> str:
    raw = f"{query.strip().lower()}|{top_k}|{archive.lower()}|{volume_filter}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cached(key: str) -> Optional[Dict[str, Any]]:
    cached = _CACHE.get(key)
    if not cached:
        return None
    cached_at, response = cached
    if time.time() - cached_at < _CACHE_TTL_SECONDS:
        return response
    # Expired
    _CACHE.pop(key, None)
    return None


def _set_cached(key: str, response: Dict[str, Any]):
    if len(_CACHE) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_CACHE))
        _CACHE.pop(oldest_key, None)
    _CACHE[key] = (time.time(), response)


def generate_grounded_answer(
    query: str,
    top_k: int = 4,
    archive: str = "all",
    volume_filter: Optional[int] = None,
) -> Dict[str, Any]:
    # 0. Check cache
    cache_key = _cache_key(query, top_k, archive, volume_filter)
    cached_result = _get_cached(cache_key)
    if cached_result:
        return cached_result

    # 1. RETRIEVE from ChromaDB (routes across BAWS, CAD, or both)
    chunks = search_archive(
        query_text=query,
        n_results=top_k,
        archive=archive,
        volume_filter=volume_filter
    )

    if not chunks:
        return {
            "query": query,
            "answer": "No archival records were found regarding this specific inquiry in the database.",
            "archive_searched": archive,
            "citations": []
        }

    # 2. DEDUPLICATE & AUGMENT
    # Prevents feeding overlapping/duplicate chunks into context
    seen_entries = set()
    unique_chunks = []
    for c in chunks:
        # CAD uses (source, title, snippet[:50]), BAWS uses (source, page)
        entry_key = (c.get("source"), c.get("page"), c.get("title", ""))
        if entry_key not in seen_entries:
            seen_entries.add(entry_key)
            unique_chunks.append(c)

    context_blocks = []
    citations = []
    for c in unique_chunks:
        arc_type = c.get("archive_type", "baws").upper()
        
        if arc_type == "CAD":
            header = f"[Source: Constituent Assembly Debates, {c.get('source')} | Title: {c.get('title')} | URL: {c.get('url')}]"
        else:
            header = f"[Source: Writings & Speeches (BAWS), {c.get('source')}, Page: {c.get('page')}]"

        context_blocks.append(f"{header}\n\"{c['snippet']}\"")
        
        snippet = c["snippet"]
        citations.append({
            "archive": arc_type,
            "source": c.get("source"),
            "title": c.get("title", ""),
            "page": c.get("page"),
            "volume": c.get("volume"),
            "url": c.get("url", ""),
            "relevance_score": c.get("relevance_score"),
            "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
        })

    joined_context = "\n\n".join(context_blocks)

    system_instruction = (
        "You are the AI Research Assistant for the Dr. Ambedkar Heritage Archive & Constituent Assembly Research Portal. "
        "Synthesize a clear, authoritative, and fact-based answer relying strictly on the provided archival excerpts.\n\n"
        "Rules:\n"
        "1. Attribution: Clearly distinguish between Dr. Ambedkar's independent writings/speeches (BAWS) and debates on the floor of the Constituent Assembly (CAD).\n"
        "2. Citations: Explicitly mention the specific Volume and Page number for BAWS, and Volume/Session Title/URL for CAD in your text.\n"
        "3. Candor: If the excerpts do not contain enough facts to answer, explicitly state that the archive has no documented record on this inquiry."
    )

    user_prompt = f"""Archival Context:
{joined_context}

Visitor Question: {query}
"""

    # 3. GENERATE
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=700
        )
        answer_text = chat_completion.choices[0].message.content.strip()

    except RateLimitError:
        return {
            "query": query,
            "answer": "The archive assistant is currently experiencing high request volume. Please try again momentarily.",
            "archive_searched": archive,
            "citations": []
        }
    except APIError as e:
        return {
            "query": query,
            "answer": f"The archive assistant encountered an upstream communication issue: {str(e)}",
            "archive_searched": archive,
            "citations": []
        }

    response = {
        "query": query,
        "answer": answer_text,
        "archive_searched": archive,
        "citations": citations
    }

    # Store in TTL cache
    _set_cached(cache_key, response)
    return response