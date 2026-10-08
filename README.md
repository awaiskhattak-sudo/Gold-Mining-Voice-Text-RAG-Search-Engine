# 🏆 AuriSearch: Placer Gold Mining RAG Search Engine & Voice Assistant

An enterprise-grade, defensive Retrieval-Augmented Generation (RAG) search engine for placer gold mining operations. Built with **FastAPI**, **ChromaDB**, **Google Gemini 2.0 / Ollama (v0.40.0)**, and an obsidian & gold voice-enabled web interface.

---

## 🌟 Key Features

1. **Multi-Modal Voice & Text Search**:
   - 🎙️ **Voice Assistant**: Interactive microphone button utilizing the browser's Web Speech API for real-time speech-to-text.
   - 🔊 **Voice Synthesis (TTS)**: Reads the grounded answers aloud with custom speech rates.
   - ⌨️ **Text Search & Quick Chips**: Preset prompt chips covering finance, machinery, sites, and metallurgy.

2. **Grounded RAG Pipeline**:
   - **Vector Database**: Embedded in persistent **ChromaDB** with cosine similarity scoring.
   - **Multi-Document Ingestion**:
     - `doc1_operation_workflow.txt`: Excavation, carpet washing, tup wash tubs, amalgamation & retorting.
     - `doc2_operation_requirements.txt`: Excavators, tractors, generators, fuel consumption, Indus water pipeline.
     - `doc3_operation_finance.txt`: Shakardara lease sites (Jungle Side, Khanano Chanda, Dowaa Khwla, Narla Khwla, Andakyy, Bajano Chanda), daily output, breakeven, and labor rates.
     - `doc4_global_gold_knowledge.txt`: Global production leaders, placer physics (specific gravity), 24K/22K purity, AISC metrics, Minamata convention.

3. **Dual LLM Orchestration with Fallback**:
   - **Primary**: Google Gemini API (`gemini-2.0-flash` from Google AI Studio).
   - **Secondary**: Local Ollama (v0.40.0 at `http://localhost:11434` with `llama3`).
   - **Defensive Tertiary**: Offline Extractive Fallback (zero crashes if APIs are unavailable).

4. **Defensive Shield Layer (`app/defensive.py`)**:
   - 🛡️ **Anti-Crash Exception Middleware**: Catches unhandled exceptions, sanitizes response headers.
   - 🔒 **Prompt Injection Guard**: Detects adversarial injection & jailbreak attempts.
   - 🧹 **Input Sanitizer**: Cleans control characters, script tags, and excessive repetition.
   - ⏱️ **Sliding-Window IP Rate Limiter**: Prevents DoS/flooding attacks (30 req / 60 sec).
   - ⚡ **Circuit Breaker**: Prevents cascading downtime when external LLM endpoints fail.

---

## 🚀 Quick Start (Local Run)

### 1. Configure API Key
Create a `.env` file in the project root:
```env
# Get your free key at: https://aistudio.google.com/
GEMINI_API_KEY=your_google_api_studio_key_here
GEMINI_MODEL=gemini-2.0-flash

# Ollama local settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run FastAPI Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at: **`http://localhost:8000`**

---

## 🐳 Docker Deployment

Docker is installed and active on your system. You can build and run this containerized application using:

### Method A: Docker Compose (Recommended)
```bash
# 1. Build and run the container
docker compose up -d --build

# 2. View live logs
docker compose logs -f

# 3. Stop the container
docker compose down
```

### Method B: Standard Docker Commands
```bash
# 1. Build Docker image
docker build -t aurisearch-gold-rag:1.0 .

# 2. Run Docker container
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY="your_google_api_studio_key" \
  -v ${PWD}/chroma_db:/app/chroma_db \
  --name aurisearch-container \
  aurisearch-gold-rag:1.0

# 3. Stop container
docker stop aurisearch-container
```

---

## 📡 API Endpoints

- **`GET /`**: Web UI with Voice Assistant and luxury gold styling.
- **`POST /api/query`**: JSON query endpoint (`{"query": "your question"}`).
- **`POST /api/ingest`**: Re-indexes documents in the `data/` folder into ChromaDB.
- **`GET /api/health`**: Health status, document count, and active LLMs.
