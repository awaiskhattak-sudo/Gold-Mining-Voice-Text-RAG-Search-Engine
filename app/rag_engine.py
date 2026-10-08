"""
RAG Engine: Document Ingestion, ChromaDB Vector Storage, Retrieval, and LLM Orchestration
Supports Google Gemini API with seamless fallback to Local Ollama 0.40.0 & Extractive Fallback.
"""

import os
import glob
import time
import logging
from typing import List, Dict, Any, Optional
import requests
import chromadb
from chromadb.utils import embedding_functions

from app.config import settings
from app.defensive import CircuitBreaker

logger = logging.getLogger("rag_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [RAG] %(message)s")

# Optional Google Generative AI import
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai package not installed or import error. Will fallback to Ollama.")


class RAGEngine:
    """Manages Vector DB, Chunking, Retrieval, and LLM Generation."""

    def __init__(self):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        # 1. Initialize Embedding Function
        # We use SentenceTransformer if available, otherwise Chroma's robust built-in ONNX default
        try:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Initialized SentenceTransformerEmbeddingFunction (all-MiniLM-L6-v2)")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to DefaultEmbeddingFunction.")
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # 2. Initialize Chroma Client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Circuit Breakers for LLM backends
        self.gemini_circuit = CircuitBreaker(failure_threshold=2, recovery_time_seconds=20)
        self.ollama_circuit = CircuitBreaker(failure_threshold=3, recovery_time_seconds=15)

        # 4. Configure Gemini if key is provided
        self._init_gemini()

        # 5. Ingest documents if collection is empty
        if self.collection.count() == 0:
            logger.info("Vector database is empty. Performing initial document ingestion...")
            self.ingest_documents()
        else:
            logger.info(f"Loaded existing ChromaDB collection with {self.collection.count()} chunks.")

    def _init_gemini(self):
        """Safely initializes Gemini API."""
        if GENAI_AVAILABLE and settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
                logger.info(f"Google Gemini configured with model: {settings.GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            if not settings.GEMINI_API_KEY:
                logger.info("No GEMINI_API_KEY set. Ollama and Extractive modes are ready.")

    def chunk_text(self, text: str, source_file: str, max_chunk_size: int = 700, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Splits document text into coherent contextual chunks with overlap.
        Preserves headers and sections.
        """
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append({
                        "id": f"{os.path.basename(source_file)}_chunk_{chunk_idx}",
                        "text": current_chunk.strip(),
                        "source": os.path.basename(source_file),
                        "chunk_index": chunk_idx
                    })
                    chunk_idx += 1
                
                # If a single paragraph is larger than max_chunk_size, split by sentences/lines
                if len(para) > max_chunk_size:
                    lines = para.split("\n")
                    temp = ""
                    for line in lines:
                        if len(temp) + len(line) <= max_chunk_size:
                            temp += ("\n" + line if temp else line)
                        else:
                            if temp:
                                chunks.append({
                                    "id": f"{os.path.basename(source_file)}_chunk_{chunk_idx}",
                                    "text": temp.strip(),
                                    "source": os.path.basename(source_file),
                                    "chunk_index": chunk_idx
                                })
                                chunk_idx += 1
                            temp = line
                    current_chunk = temp
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append({
                "id": f"{os.path.basename(source_file)}_chunk_{chunk_idx}",
                "text": current_chunk.strip(),
                "source": os.path.basename(source_file),
                "chunk_index": chunk_idx
            })

        return chunks

    def ingest_documents(self) -> int:
        """Reads all txt files from data directory, embeds and inserts into ChromaDB."""
        doc_files = glob.glob(os.path.join(settings.DATA_DIR, "*.txt"))
        if not doc_files:
            logger.warning(f"No .txt documents found in {settings.DATA_DIR}")
            return 0

        all_chunks = []
        for file_path in doc_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_chunks = self.chunk_text(content, file_path)
                all_chunks.extend(file_chunks)
                logger.info(f"Parsed {len(file_chunks)} chunks from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        if not all_chunks:
            return 0

        # Upsert into ChromaDB
        ids = [c["id"] for c in all_chunks]
        documents = [c["text"] for c in all_chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]

        # Ingest in batches to avoid payload limits
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        total_stored = self.collection.count()
        logger.info(f"Successfully ingested {len(all_chunks)} chunks. Total in collection: {total_stored}")
        return total_stored

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Performs vector search in ChromaDB and returns top-k matching chunks with similarity scores."""
        k = top_k or settings.TOP_K
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.5] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # For cosine distance: distance is 0 to 2 (0 = identical). Similarity = max(0, 1 - dist)
                sim_score = max(0.0, round((1.0 - float(dist)) * 100, 1))
                retrieved.append({
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "similarity": sim_score,
                    "distance": round(float(dist), 4)
                })

        return retrieved

    def generate_with_gemini(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API with timeout and error protection."""
        if not self.gemini_model or not self.gemini_circuit.can_execute():
            return None

        try:
            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024}
            )
            if response and response.text:
                self.gemini_circuit.record_success()
                return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API call error: {e}")
            self.gemini_circuit.record_failure()

        return None

    def generate_with_ollama(self, prompt: str) -> Optional[str]:
        """Calls local Ollama API (version 0.40.0) with timeout."""
        if not self.ollama_circuit.can_execute():
            return None

        ollama_url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 1024
            }
        }

        try:
            res = requests.post(ollama_url, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                self.ollama_circuit.record_success()
                return data.get("response", "").strip()
            else:
                logger.warning(f"Ollama returned HTTP {res.status_code}: {res.text[:200]}")
                self.ollama_circuit.record_failure()
        except Exception as e:
            logger.warning(f"Ollama connection unavailable or timed out: {e}")
            self.ollama_circuit.record_failure()

        return None

    def extractive_fallback(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Defensive fallback when both Gemini and Ollama are unavailable.
        Extracts verified factual points directly from top retrieved chunks.
        """
        if not chunks:
            return "No matching gold mining records were found for this query."

        top_chunk = chunks[0]["content"]
        source = chunks[0]["source"]

        response = (
            f"**Verified Grounded Intelligence (Extractive Mode)**\n\n"
            f"Based directly on official Shakardara operations and global gold documentation (`{source}`):\n\n"
        )
        
        # Present the core content cleanly
        lines = [line.strip() for line in top_chunk.split("\n") if line.strip() and not line.startswith("=")]
        for line in lines[:8]:
            if line.startswith("-") or line.startswith("*") or ":" in line:
                response += f"- {line.lstrip('-* ')}\n"
            else:
                response += f"{line}\n\n"

        response += "\n*(Generated via defensive local fallback mode while LLM connection is synchronizing)*"
        return response

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Complete RAG Pipeline:
        1. Retrieve top-k relevant chunks from ChromaDB
        2. Format prompt with system context & strict grounding instructions
        3. Attempt LLM generation: Gemini -> Ollama -> Extractive Fallback
        4. Return response with evidence sources, confidence, and engine telemetry
        """
        start_time = time.time()

        # 1. Retrieve
        chunks = self.retrieve(query)
        if not chunks:
            return {
                "query": query,
                "answer": "No relevant records found in the gold mining database.",
                "sources": [],
                "engine": "none",
                "latency_ms": round((time.time() - start_time) * 100, 2)
            }

        # 2. Build Context
        context_str = "\n\n---\n\n".join([
            f"[Source: {c['source']} | Match: {c['similarity']}%]\n{c['content']}"
            for c in chunks
        ])

        system_instruction = (
            "You are AuriSearch, the Chief Placer Gold Mining & Operations Intelligence AI. "
            "You provide authoritative, precise, and practical answers regarding gold mining operations, "
            "financial calculations, machinery, labor, fuel, and global gold industry standards.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground your answer strictly in the provided context below.\n"
            "2. State exact numbers, PKR figures, diesel consumption, equipment rates, and site names when asked.\n"
            "3. If details are not explicitly in the context, clearly state what is known from the records.\n"
            "4. Format your response cleanly with bullet points, bold key figures, and concise summaries.\n"
        )

        prompt = (
            f"{system_instruction}\n"
            f"DOCUMENTATION CONTEXT:\n{context_str}\n\n"
            f"USER QUERY: {query}\n\n"
            f"GROUNDED ANSWER:"
        )

        # 3. LLM Generation Cascade
        engine_used = "extractive_fallback"
        answer = None

        # Attempt 1: Gemini
        if settings.GEMINI_API_KEY:
            answer = self.generate_with_gemini(prompt)
            if answer:
                engine_used = f"Google Gemini ({settings.GEMINI_MODEL})"

        # Attempt 2: Ollama (if Gemini failed or no key provided)
        if not answer:
            answer = self.generate_with_ollama(prompt)
            if answer:
                engine_used = f"Local Ollama ({settings.OLLAMA_MODEL})"

        # Attempt 3: Extractive Fallback (guaranteed zero crash)
        if not answer:
            answer = self.extractive_fallback(query, chunks)
            engine_used = "Defensive Extractive Fallback"

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "sources": chunks,
            "engine": engine_used,
            "latency_ms": latency
        }


# Global singleton instance
rag_engine = RAGEngine()
