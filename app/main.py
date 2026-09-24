from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from app.core.indexer import search_archive
from app.core.rag import generate_grounded_answer

app = FastAPI(
    title="Digital Heritage Archive - Search & RAG Engine",
    description="Institutional Knowledge Platform for Dr. B. R. Ambedkar Archives",
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
    page: int
    volume: int
    snippet: str
    relevance_score: float

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResultItem]

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]

@app.get("/api/v1/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Query phrase"), limit: int = 4):
    results = search_archive(query_text=q, n_results=limit)
    return {
        "query": q,
        "total_results": len(results),
        "results": results
    }

@app.post("/api/v1/ask", response_model=AskResponse)
def ask_question(q: str = Query(..., description="Ask a question grounded in the archives")):
    return generate_grounded_answer(query=q)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "heritage-archive-rag"}