import os
import json
from typing import List, Dict, Any
import hashlib
import math
from collections import Counter

try:
    import fitz  # PyMuPDF (optional)
except ImportError:
    fitz = None
try:
    from pypdf import PdfReader  # pure-Python fallback
except ImportError:
    PdfReader = None

FAISS = None
HuggingFaceEmbeddings = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

class PDFRAGAgent:
    """Handles PDF ingestion, lightweight chunk store, and naive similarity retrieval.

    Design Notes:
    - Avoids heavy native dependencies for portability (Python 3.13 compatible).
    - Chunks stored in memory; consider persistence for production scale.
    - Similarity: simple TF cosine (fast, approximate vs embeddings).
    """

    def __init__(self, storage_dir: str = "vector_store", pdf_dir: str = "sample_pdfs"):
        self.storage_dir = storage_dir
        self.pdf_dir = pdf_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)
        # In-memory store of chunks: list of dicts {text, source, chunk}
        self.chunks = []
        # Optional embedding / backend configuration
        self.use_embeddings = bool(os.getenv("RAG_EMBEDDINGS"))
        self.backend = os.getenv("RAG_BACKEND", "tf").lower()
        self._chroma_client = None
        self._chroma_collection = None
        self._emb_model = None
        self._emb_vectors = []  # parallel to self.chunks when embeddings enabled
        if self.use_embeddings or self.backend in ("chroma",):
            self._try_init_embeddings()
        if self.backend == "chroma":
            self._try_init_chroma()

    def _try_init_embeddings(self):
        """Attempt to load a sentence-transformers model if available.

        Controlled by env var RAG_EMBEDDINGS=1. Falls back silently if unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model_name = os.getenv("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
            self._emb_model = SentenceTransformer(model_name)
        except Exception:
            self.use_embeddings = False  # disable gracefully
            self._emb_model = None

    def _init_vectorstore(self):
        pass

    def ingest_pdf(self, file_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"file": os.path.basename(file_path)}
        if not os.path.isfile(file_path):
            result.update({"status": "error", "message": "File not found"})
            return result
        try:
            text = self._extract_text(file_path)
            if not text.strip():
                # Heuristic: attempt naive bytes decode as last resort
                try:
                    with open(file_path, 'rb') as fh:
                        raw = fh.read().decode(errors='ignore')
                    # Pull any parentheses content (very naive PDF text extract)
                    import re
                    candidate = " ".join(re.findall(r'\((.*?)\)', raw))
                    text = candidate if candidate.strip() else text
                except Exception:
                    pass
            if not text.strip():
                result.update({"status": "warning", "message": "No extractable text"})
                return result
            chunks = self._chunk_text(text)
            new_records = []
            for i, c in enumerate(chunks):
                rec = {"text": c, "source": os.path.basename(file_path), "chunk": i}
                self.chunks.append(rec)
                new_records.append(rec)
            # If embeddings enabled and model loaded, build vectors for new chunks
            if (self.use_embeddings or self.backend == "chroma") and self._emb_model:
                try:
                    texts = [r["text"] for r in new_records]
                    embeds = self._emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                    for vec, rec in zip(embeds, new_records):
                        self._emb_vectors.append(vec)
                        if self.backend == "chroma" and self._chroma_collection is not None:
                            try:
                                self._chroma_collection.add(
                                    ids=[f"{rec['source']}::{rec['chunk']}"],
                                    embeddings=[vec.tolist()],
                                    metadatas=[{"source": rec['source'], "chunk": rec['chunk']}],
                                    documents=[rec['text']]
                                )
                            except Exception:
                                pass
                except Exception:
                    # Disable embeddings if something goes wrong mid-way (do not break ingestion)
                    self.use_embeddings = False
                    self._emb_model = None
                    self._emb_vectors = []
            result.update({"status": "ok", "chunks": len(chunks)})
            return result
        except Exception as e:
            result.update({"status": "error", "message": f"{type(e).__name__}: {e}"})
            return result

    def ingest_all_in_dir(self):
        pdfs = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        for p in pdfs:
            self.ingest_pdf(os.path.join(self.pdf_dir, p))

    def _extract_text(self, file_path: str) -> str:
        # Prefer fitz if available for better layout extraction
        if fitz:
            try:
                doc = fitz.open(file_path)
                texts = [page.get_text() for page in doc]
                return "\n".join(texts)
            except Exception:
                pass
        # Fallback to pypdf
        if PdfReader:
            try:
                reader = PdfReader(file_path)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages_text)
            except Exception:
                pass
        return ""  # last resort

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        if not self.chunks:
            return {"answer": "PDF knowledge base empty.", "sources": [], "debug": {"chunks_total": 0, "mode": self._mode()}}
        if not query or not query.strip():
            return {"answer": "Empty query.", "sources": [], "debug": {"chunks_total": len(self.chunks), "mode": self._mode()}}

        if self.backend == "chroma" and self._chroma_collection is not None:
            top = self._retrieve_with_chroma(query, k)
        elif self.use_embeddings and self._emb_model and self._emb_vectors:
            top = self._retrieve_with_embeddings(query, k)
        else:
            scored: List[tuple] = []
            for c in self.chunks:
                try:
                    score = self._similarity(query, c["text"])
                except Exception:
                    score = 0.0
                scored.append((score, c))
            top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        context = "\n".join(c["text"] for _, c in top)
        from agents.controller_agent import call_llm_api
        prompt = (
            "You are a PDF RAG assistant. Given the following retrieved chunks, answer the user query succinctly.\n"\
            f"Query: {query}\n\nChunks:\n{context[:4000]}\n\nAnswer:"
        )
        answer = call_llm_api(prompt)
        # Detect LLM failure / echo fallback patterns and apply local extractive summary
        if any(tag in answer for tag in ["[Custom LLM Error", "[LLM Fallback]", "[Echo]"]):
            local_summary = self._local_summarize(context, query)
            answer = f"(Local Summary Fallback)\n{local_summary}" if local_summary else answer
        sources = [{"source": c["source"], "chunk": c["chunk"], "preview": c["text"][:160], "score": float(s)} for s, c in top]
        return {"answer": answer, "sources": sources, "debug": {"chunks_total": len(self.chunks), "mode": self._mode(), "embeddings": bool(self.use_embeddings and self._emb_model)}}

    def _mode(self) -> str:
        if self.backend == "chroma" and self._chroma_collection is not None:
            return "chroma"
        return "embeddings" if (self.use_embeddings and self._emb_model) else "tf"

    def _retrieve_with_embeddings(self, query: str, k: int) -> List[tuple]:
        try:
            import numpy as np  # lazy import
        except Exception:
            return []  # fallback will produce empty context -> LLM fallback
        try:
            q_vec = self._emb_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
            # Normalize and compute cosine similarities
            def _norm(v):
                denom = (np.linalg.norm(v) + 1e-9)
                return v / denom
            qn = _norm(q_vec)
            sims = []
            for vec, chunk in zip(self._emb_vectors, self.chunks):
                try:
                    sims.append((float((_norm(vec) @ qn)), chunk))
                except Exception:
                    sims.append((0.0, chunk))
            top = sorted(sims, key=lambda x: x[0], reverse=True)[:k]
            return top
        except Exception:
            return []

    def _try_init_chroma(self):
        try:
            import chromadb  # type: ignore
            from chromadb.utils import embedding_functions  # noqa
            self._chroma_client = chromadb.Client()  # in-memory default
            self._chroma_collection = self._chroma_client.get_or_create_collection("pdf_chunks")
        except Exception:
            self.backend = "tf"
            self._chroma_client = None
            self._chroma_collection = None

    def _retrieve_with_chroma(self, query: str, k: int) -> List[tuple]:
        if not (self._chroma_collection and self._emb_model):
            return []
        try:
            import numpy as np
            q_vec = self._emb_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
            # Chroma similarity via manual cosine over stored embeddings we added
            sims = []
            for vec, chunk in zip(self._emb_vectors, self.chunks):
                try:
                    sims.append((float((vec @ q_vec) / ((np.linalg.norm(vec) * np.linalg.norm(q_vec)) + 1e-9)), chunk))
                except Exception:
                    sims.append((0.0, chunk))
            return sorted(sims, key=lambda x: x[0], reverse=True)[:k]
        except Exception:
            return []

    # Management helpers
    def stats(self) -> Dict[str, Any]:
        return {
            "chunks": len(self.chunks),
            "mode": self._mode(),
            "backend": self.backend,
            "embeddings_active": bool(self._emb_model),
        }

    def reset(self):
        self.chunks.clear()
        self._emb_vectors = []
        if self.backend == "chroma" and self._chroma_collection is not None:
            try:
                self._chroma_client.delete_collection("pdf_chunks")
            except Exception:
                pass
            self._try_init_chroma()

    def _similarity(self, a: str, b: str) -> float:
        # Simple cosine over term frequency
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        ca, cb = Counter(ta), Counter(tb)
        common = set(ca.keys()) & set(cb.keys())
        num = sum(ca[w] * cb[w] for w in common)
        da = math.sqrt(sum(v*v for v in ca.values()))
        db = math.sqrt(sum(v*v for v in cb.values()))
        return num / (da * db + 1e-9)

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t.strip()]

    def _local_summarize(self, context: str, query: str, max_sentences: int = 5) -> str:
        """Naive extractive summarizer as a fallback when LLM is unavailable.

        Strategy:
        - Split into sentences (rudimentary punctuation-based split)
        - Score sentences by term frequency overlap with query + overall token frequency (exclude stopwords)
        - Return top N sentences in original order
        """
        if not context.strip():
            return ""
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
        if not sentences:
            return context[:400]
        stop = {"the","is","a","an","of","to","and","in","for","on","with","by","this","that","it","as","at","be","are"}
        query_terms = {w for w in self._tokenize(query) if w not in stop}
        # Global term frequencies
        all_tokens = self._tokenize(context)
        from collections import Counter
        tf = Counter(t for t in all_tokens if t not in stop)
        scored = []
        for idx, sent in enumerate(sentences):
            tokens = [t for t in self._tokenize(sent) if t not in stop]
            if not tokens:
                continue
            overlap = sum(1 for t in tokens if t in query_terms)
            weight = sum(tf.get(t,0) for t in tokens) / (len(tokens) + 1e-9)
            score = overlap * 2 + weight  # simple linear combo
            scored.append((score, idx, sent))
        if not scored:
            return context[:400]
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
        # Preserve original order among selected
        ordered = [s for _,_,s in sorted(top, key=lambda x: x[1])]
        return " ".join(ordered)
