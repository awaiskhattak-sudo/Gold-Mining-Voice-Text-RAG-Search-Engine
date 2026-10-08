import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    # App Settings
    APP_NAME: str = "AuriSearch Gold Mining RAG Engine"
    APP_VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

    # API Keys & LLM Configuration
    # Set your Google AI Studio API key in .env or environment: GEMINI_API_KEY=your_key_here
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # Ollama Settings (Version 0.40.0 installed locally)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:latest")

    # Vector Database Settings
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"))
    COLLECTION_NAME: str = "gold_mining_knowledge"
    
    # RAG Retrieval Parameters
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.25  # Cosine distance or relevance threshold

    # Defensive Security & Rate Limiting
    MAX_QUERY_LENGTH: int = 800
    MIN_QUERY_LENGTH: int = 2
    RATE_LIMIT_REQUESTS: int = 30  # Max requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # per 60 seconds per IP

settings = Settings()
