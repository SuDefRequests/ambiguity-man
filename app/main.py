from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.core.indexer import search_archive
from app.core.rag import generate_grounded_answer

app = FastAPI(
    title="Digital Heritage Archive & CAD RAG Engine",
    description="Institutional Knowledge Platform for Dr. B. R. Ambedkar Archives and Constituent Assembly Debates",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResultItem(BaseModel):
    source: str
    archive_type: Optional[str] = "baws"
    page: Optional[int] = 0
    volume: Optional[int] = 0
    title: Optional[str] = ""
    url: Optional[str] = ""
    snippet: str
    relevance_score: float

class SearchResponse(BaseModel):
    query: str
    archive_searched: str
    total_results: int
    results: List[SearchResultItem]

class AskResponse(BaseModel):
    query: str
    answer: str
    archive_searched: str
    citations: List[Dict[str, Any]]


@app.get("/api/v1/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Query phrase across the archives"),
    archive: str = Query("all", description="Target archive: 'baws', 'cad', or 'all'"),
    volume: Optional[int] = Query(None, description="Filter by volume number"),
    limit: int = Query(4, description="Maximum number of passages to retrieve")
):
    results = search_archive(
        query_text=q,
        n_results=limit,
        archive=archive,
        volume_filter=volume
    )
    return {
        "query": q,
        "archive_searched": archive,
        "total_results": len(results),
        "results": results
    }


@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(
    q: str = Query(..., description="Ask questions about Dr. Ambedkar's works, speeches, or assembly debates"),
    archive: str = Query("all", description="Target archive: 'baws' (Writings), 'cad' (Debates), or 'all' (Cross-Archive)"),
    top_k: int = Query(4, description="Number of source excerpts to retrieve"),
    volume: Optional[int] = Query(None, description="Optional volume filter")
):
    return generate_grounded_answer(
        query=q,
        top_k=top_k,
        archive=archive,
        volume_filter=volume
    )


@app.post("/api/v1/ask/cad", response_model=AskResponse)
def ask_cad(
    q: str = Query(..., description="Dedicated query endpoint for the Constituent Assembly Debates (Vols 1-5)"),
    top_k: int = Query(4, description="Number of debate passages to retrieve"),
    volume: Optional[int] = Query(None, description="Optional CAD volume filter (1-5)")
):
    return generate_grounded_answer(
        query=q,
        top_k=top_k,
        archive="cad",
        volume_filter=volume
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "heritage-archive-rag"}