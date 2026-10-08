"""
Main FastAPI Server for AuriSearch Gold Mining RAG Search Engine
Equipped with Defensive Shield Middleware, ChromaDB, and Multi-LLM Orchestration.
"""

import os
import time
from fastapi import FastAPI, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.config import settings
from app.defensive import (
    DefensiveMiddleware,
    RateLimiter,
    validate_user_query,
    DefensiveSecurityException
)
from app.rag_engine import rag_engine

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Placer Gold Mining Search & Voice Assistant with RAG"
)

# Attach Defensive Security & Rate Limiting Middleware
rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS
)
app.add_middleware(DefensiveMiddleware, rate_limiter=rate_limiter)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount Static Assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Pydantic Schemas
class SearchQueryRequest(BaseModel):
    query: str = Field(..., description="User query or voice transcript", min_length=1, max_length=1000)

class SourceCitation(BaseModel):
    source: str
    similarity: float
    distance: float
    content: str

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    engine: str
    latency_ms: float
    status: str = "success"

class HealthResponse(BaseModel):
    app_name: str
    version: str
    status: str
    documents_indexed: int
    gemini_configured: bool
    gemini_model: str
    ollama_host: str
    ollama_model: str


@app.get("/", summary="Serve Gold Mining Web Interface")
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AuriSearch API running. Static frontend not yet compiled."}


@app.post("/api/query", response_model=SearchResponse, summary="Query Vector DB & LLM")
async def handle_search_query(payload: SearchQueryRequest):
    """
    Processes incoming text or voice-transcribed queries:
    1. Defensive input validation and prompt injection neutralization
    2. Semantic vector retrieval from ChromaDB
    3. Multi-LLM generation (Gemini -> Ollama -> Extractive)
    """
    # Defensive validation
    sanitized_query = validate_user_query(
        raw_query=payload.query,
        min_len=settings.MIN_QUERY_LENGTH,
        max_len=settings.MAX_QUERY_LENGTH
    )

    # Execute RAG
    result = rag_engine.answer_query(sanitized_query)
    
    return SearchResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"],
        engine=result["engine"],
        latency_ms=result["latency_ms"],
        status="success"
    )


@app.post("/api/ingest", summary="Re-index Knowledge Base")
async def trigger_ingestion():
    """Defensively re-ingests all documents in the data folder into ChromaDB."""
    try:
        count = rag_engine.ingest_documents()
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed {count} chunks in ChromaDB.",
            "total_chunks": count
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": f"Ingestion error handled: {str(e)}"}
        )


@app.get("/api/health", response_model=HealthResponse, summary="System Health & Diagnostic Telemetry")
async def health_check():
    """Returns vector database status, active LLMs, and defensive safeguards."""
    return HealthResponse(
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        status="healthy",
        documents_indexed=rag_engine.collection.count(),
        gemini_configured=bool(settings.GEMINI_API_KEY),
        gemini_model=settings.GEMINI_MODEL,
        ollama_host=settings.OLLAMA_HOST,
        ollama_model=settings.OLLAMA_MODEL
    )
