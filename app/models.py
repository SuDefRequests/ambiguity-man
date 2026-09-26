from pydantic import BaseModel
from typing import Optional, Literal


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
