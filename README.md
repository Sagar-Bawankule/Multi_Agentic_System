# Multi-Agentic System

Dynamic multi-agent question answering platform (FastAPI backend + minimal frontend) that decides—per query—whether to route to PDF RAG, Web Search, ArXiv, an LLM, or a combination. Controller logs full reasoning traces. Supports Groq, Google Gemini (AI Studio), OpenAI, local (Ollama/custom) or echo fallback.

## Core Capabilities
1. Controller Agent
  - Rule-based + optional LLM-assisted reasoning
  - Supports providers: `groq`, `gemini`, `openai`, `custom` (Ollama), `echo` fallback
  - Logs: timestamp, query, decision rationale, agents invoked, per-agent run info, retrieved documents, final answer
2. PDF RAG Agent
  - PDF upload, extraction (PyMuPDF or pypdf, heuristic fallback)
  - Chunking (size 800, overlap 120) with metadata
  - Retrieval modes: TF cosine (default), Embeddings (sentence-transformers, env `RAG_EMBEDDINGS=1`), optional Chroma backend (`RAG_BACKEND=chroma`)
  - Reset & stats endpoints; synthetic PDF generation endpoint for quick demos
3. Web Search Agent
  - DuckDuckGo HTML parsing (no key) or SerpAPI (`SERPAPI_KEY`)
  - Summarization upgrade-ready via LLM (prompting happens in controller answer synthesis path)
4. ArXiv Agent
  - Direct ArXiv API queries with retry logic
  - Summarization via LLM if provider present

## Extended Features
- Endpoints for inspection: `/controller/decide`, `/agents/status`, `/pdf/stats`, `/pdf/reset`, `/pdf/generate_from_text`
- Structured JSON trace logs in `./logs`
- Minimal HTML UI: query box, PDF upload, answer display, agents used, rationale, logs panel, extended health
- Secure upload constraints (type+size) and environment-based secret handling

## Architecture Overview
```
User → FastAPI (/ask) → Controller
  Controller decides agents (rule/LLM) → invokes PDF RAG / Web Search / ArXiv / LLM
  Aggregates agent outputs → Final synthesized answer → Log trace → Response
```

### Decision Rules (Baseline)
- PDF flagged (checkbox or 'pdf' keyword) → include PDF RAG
- Query contains ('recent papers','arxiv','research') → include ArXiv
- Query contains ('latest news','recent developments','breaking news','current events') → include Web Search
- Else fallback: LLM only
Multiple matches yield multi-agent invocation; order preserved & deduplicated.

## Folder Structure

## Folder Structure
```
app.py                  # FastAPI entrypoint
agents/
  controller_agent.py
  pdf_rag_agent.py
  web_search_agent.py
  arxiv_agent.py
frontend/
  index.html
logs/                   # JSON trace logs
sample_pdfs/            # Place sample PDFs here
requirements.txt
README.md
REPORT.pdf (architecture report – placeholder / to be expanded)
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables (.env)
```
# LLM Provider Choice: groq | gemini | openai | custom | echo
LLM_PROVIDER=groq

# Groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-1.5-flash

# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

# Local (Ollama/custom) generic chat endpoint
LLM_PROVIDER=custom
LLM_API_BASE=http://localhost:11434
LLM_MODEL=llama3

# Web search enhancement (optional)
SERPAPI_KEY=your_serpapi_key

# RAG tunables
RAG_EMBEDDINGS=1            # enable sentence-transformers
RAG_EMBED_MODEL=all-MiniLM-L6-v2
RAG_BACKEND=chroma          # or tf (default)

# Misc future options
# RATE_LIMIT_QPM=60
```

### 3. Run Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# or
python run_app.py
```
Visit: http://localhost:8000

## Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/ask` | POST (form) | Fields: `query`, `use_pdf` (bool) |
| `/upload_pdf` | POST (multipart) | Upload & ingest PDF |
| `/logs` | GET | Retrieve recent decision logs |
| `/health` | GET | Healthcheck |
| `/health/extended` | GET | Extended metrics |
| `/controller/decide` | GET | Dry-run routing decision |
| `/agents/status` | GET | Agent diagnostics |
| `/pdf/stats` | GET | RAG stats (chunks, backend) |
| `/pdf/reset` | POST | Clear RAG store |
| `/pdf/generate_from_text` | POST (form) | Create + ingest synthetic PDF |

## PDF RAG
Extraction priority: PyMuPDF → pypdf → heuristic fallback.
Chunking: sliding window (size 800, overlap 120).
Retrieval:
- Default TF cosine token frequency
- Embeddings (if `RAG_EMBEDDINGS=1`) using sentence-transformers
- Optional Chroma backend (in-memory) if `RAG_BACKEND=chroma` + embeddings enabled

Sources list each chunk: source filename, chunk index, preview, score.

## LLM Integration
`call_llm_api` supports: Groq, Gemini, OpenAI, custom (Ollama), echo fallback. Errors degrade gracefully to echo. Provide only needed keys. Custom expects an Ollama-compatible `/api/chat` JSON interface.

## Deployment (Render / Hugging Face Spaces)
1. Set environment variables in dashboard (never commit secrets).
2. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Expose port 8000.
4. (Optional) Pre-ingest PDFs at startup (already auto-ingests `sample_pdfs/`).
5. Add a persistent volume if long-term PDF retention required (currently ephemeral).

## Security & Privacy
- PDF size limited to 5MB.
- Only PDFs accepted.
- Files stored locally in `sample_pdfs/`; clear directory for sensitive data.
- Add auth (e.g., API key header) for production.

## Security & Privacy
- PDF validation: extension + max 5MB.
- Local storage only; delete `sample_pdfs/` to purge.
- No PII detection yet—recommend external sanitizer for production.
- Add auth middleware (e.g., header token) for multi-tenant use.

## Limitations
- Without real LLM keys, synthesis quality is limited (echo/custom fallback).
- DuckDuckGo HTML parsing brittle if markup changes.
- ArXiv summarization concise but shallow; could add embedding rerank pipeline.
- Chroma not persisted across restarts (in-memory client).

## Roadmap Ideas
- Streaming token responses.
- FAISS index persistence layer or lightweight sqlite-backed Chroma.
- Rate limiting & API key auth.
- Hybrid retrieval (BM25 + embeddings).

## Roadmap Ideas
- Add conversation history & memory.
- Integrate better summarization models.
- Support streaming responses.
- Add user authentication & rate limiting.

## License
MIT (adjust as desired).
