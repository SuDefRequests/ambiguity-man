"""Read-only passage lookup over the repository's original archive exports.

This is the export-backed source for this endpoint only; it does not reconcile
Chroma, PostgreSQL, or graph data. Restart the process after replacing exports.
"""
import json
from functools import lru_cache
from pathlib import Path

from app.models import ArchiveAdjacencyResponse, ArchiveBrowseResponse, ArchivePassage

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_passages() -> dict[str, ArchivePassage]:
    passages = {}
    for filename in ("baws_data_for_graph.json", "cad_data_for_graph.json"):
        with (ARCHIVE_ROOT / filename).open(encoding="utf-8") as source:
            records = json.load(source)
        for record in records:
            passage = ArchivePassage(
                passage_id=record["chunk_id"],
                archive_type=record["archive_type"].lower(),
                source=record["source"],
                page=record["page"],
                volume=record["volume"],
                title=record.get("title"),
                url=record.get("url"),
                text=record["text"],
            )
            if passage.passage_id in passages:
                raise ValueError(f"Duplicate archive passage ID: {passage.passage_id}")
            passages[passage.passage_id] = passage
    return passages


def browse_passages(archive: str, volume: int | None, offset: int, limit: int) -> ArchiveBrowseResponse:
    """Preserve export array order: BAWS first, then CAD for archive=all.

    get_passages preserves insertion order. Filtering is applied before slicing
    and retains that order, including when a volume spans both collections.
    Counts come from matching records, never a configured corpus size.
    """
    matches = [
        passage for passage in get_passages().values()
        if (archive == "all" or passage.archive_type == archive)
        and (volume is None or passage.volume == volume)
    ]
    total = len(matches)
    return ArchiveBrowseResponse(
        archive=archive, offset=offset, limit=limit, total=total,
        has_more=offset + limit < total,
        results=matches[offset:offset + limit],
    )


def passage_adjacency(passage_id: str) -> ArchiveAdjacencyResponse | None:
    """Immediate export neighbors only; never skip over a source boundary.

    BAWS is bounded by collection and volume. CAD additionally requires the
    same session title and URL (the session metadata present in the exports).
    No sorting, passage-ID parsing, or search-result ordering is involved.
    """
    passages = get_passages()
    current = passages.get(passage_id)
    if current is None:
        return None
    ordered = list(passages.values())
    index = next(i for i, passage in enumerate(ordered) if passage.passage_id == passage_id)

    def neighbor(position: int) -> ArchivePassage | None:
        if not 0 <= position < len(ordered):
            return None
        candidate = ordered[position]
        if (candidate.archive_type, candidate.volume) != (current.archive_type, current.volume):
            return None
        if current.archive_type == "cad" and (candidate.title, candidate.url) != (current.title, current.url):
            return None
        return candidate

    return ArchiveAdjacencyResponse(current=current, previous=neighbor(index - 1), next=neighbor(index + 1))
