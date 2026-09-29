from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Optional, Literal


class Passage(BaseModel):
    passage_id: str
    doc_id: str
    speaker: Optional[str] = None
    date: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    language: str = "en"
    ocr_confidence: Optional[float] = None
    article_refs: list[str] = []
    bbox: Optional[list] = None


class ArchivePassage(BaseModel):
    """Export-backed passage; no inferred document identity or metadata."""

    passage_id: str
    archive_type: Literal["baws", "cad"]
    source: str
    page: int
    volume: int
    title: Optional[str]
    url: Optional[str]
    text: str


class ArchiveAdjacencyResponse(BaseModel):
    current: ArchivePassage
    previous: Optional[ArchivePassage]
    next: Optional[ArchivePassage]


class ArchiveBrowseResponse(BaseModel):
    archive: Literal["all", "baws", "cad"]
    offset: int
    limit: int
    total: int
    has_more: bool
    results: list[ArchivePassage]


class ArchiveSearchHit(ArchivePassage):
    snippet: str
    relevance_score: Optional[float] = None


class ArchiveSearchResponse(BaseModel):
    query: str
    archive_searched: Literal["all", "baws", "cad"]
    total_results: int
    results: list[ArchiveSearchHit]


class SearchRequest(BaseModel):
    query: str
    lang: str = "en"
    top_k: int = 8
    use_graph: bool = True


class SearchHit(BaseModel):
    passage_id: str
    text: str
    score: float
    page: Optional[int] = None
    speaker: Optional[str] = None
    date: Optional[str] = None
    via: Literal["vector", "bm25", "graph", "fusion"]


class SearchResponse(BaseModel):
    passages: list[SearchHit]


class QuoteRequest(BaseModel):
    quote: str


class QuoteMatch(BaseModel):
    passage_id: str
    similarity: float
    text: str
    page: Optional[int] = None
    doc_id: Optional[str] = None


class QuoteResponse(BaseModel):
    verdict: Literal["verified", "paraphrase", "not_found"]
    matches: list[QuoteMatch]


class GraphNode(BaseModel):
    id: str
    type: str
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    passage_id: Optional[str] = None


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    archive: Literal["all", "baws", "cad"] = "all"
    language: Literal["en", "hi", "mr"] | None = None
    volume: Optional[int] = Field(None, ge=1, le=5)
    top_k: int = Field(6, ge=1, le=8)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[ArchivePassage]




# Bound each kiosk narration to a short passage (4,000 characters after trimming).
MAX_NARRATION_TEXT_LENGTH = 4000


class SpeakRequest(BaseModel):
    text: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_NARRATION_TEXT_LENGTH,
    )]
    voice: str = "alloy"
    language: str = "en"


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None