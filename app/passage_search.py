"""Adapt existing Chroma retrieval to canonical export-backed passage results."""
from fastapi import HTTPException

from app.archive import get_passages
from app.models import ArchiveSearchHit, ArchiveSearchResponse
from app.ocr.models import OCRSearchHit
from app.ocr.storage import OCRStorage


def retrieve_passages(collection, query: str, archive: str, volume: int | None,
                      limit: int) -> ArchiveSearchResponse:
    # Ingestion preserves uppercase export archive_type values in Chroma.
    archive_filter = {"archive_type": {"$in": ["BAWS", "CAD", "ocr"]}} if archive == "all" else {
        "archive_type": "ocr" if archive == "ocr" else archive.upper(),
    }
    where = archive_filter if volume is None else {
        "$and": [archive_filter, {"volume": volume}],
    }
    passages = get_passages()
    ocr_passages = OCRStorage().passages()
    try:
        retrieved = collection.query(query_texts=[query], n_results=limit, where=where)
    except Exception as exc:
        raise HTTPException(503, detail={
            "code": "search_unavailable", "message": "Archive search is unavailable",
        }) from exc

    results = []
    seen = set()
    for passage_id in (retrieved.get("ids") or [[]])[0]:
        passage = passages.get(passage_id) or ocr_passages.get(passage_id)
        # Never invent records for upload-only/stale IDs or trust indexed metadata.
        if passage is None or passage_id in seen:
            continue
        if archive != "all" and passage.archive_type != archive:
            continue
        if volume is not None and passage.volume != volume:
            continue
        seen.add(passage_id)
        hit_model = OCRSearchHit if passage.archive_type == "ocr" else ArchiveSearchHit
        results.append(hit_model(
            **passage.model_dump(), snippet=passage.text[:300], relevance_score=None,
        ))
        if len(results) == limit:
            break
    # Preserve Chroma rank. Distances are not presented as calibrated relevance.
    return ArchiveSearchResponse(query=query, archive_searched=archive,
                                 total_results=len(results), results=results)
