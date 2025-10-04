# Multi-Agentic System Architecture

## 1. High-Level Overview
User query -> FastAPI `/ask` -> ControllerAgent decides -> Invokes one or more specialized agents (PDF RAG, Web Search, ArXiv, or LLM fallback) -> Aggregated answer + trace log stored.

Sequence (simplified):
1. User uploads PDFs (`/upload_pdf`) -> PDFRAGAgent ingests (extract, chunk, embed, index)
2. User submits query (`/ask`)
3. Controller applies rules:
   - PDF keyword or uploaded flag -> PDFRAGAgent
   - Research keywords -> ArxivAgent
   - News keywords -> WebSearchAgent
   - Else -> LLM fallback
4. Each invoked agent returns structured data.
5. Controller merges responses, writes JSON log.

## 2. Agent Interfaces
### ControllerAgent
Input: query:str, has_pdf:bool -> Output: log record dict
### PDFRAGAgent
Methods: ingest_pdf(path), ingest_all_in_dir(), answer(query)
### WebSearchAgent
Method: search_and_summarize(query)
### ArxivAgent
Method: search_and_summarize(query)

## 3. Decision Logic
Rules (priority additive; can invoke multiple agents):
- if has_pdf or 'pdf' in query -> add pdf_rag
- if any of ['recent papers','arxiv','research'] -> add arxiv
- if any of ['latest news','recent developments'] -> add web_search
- if none added -> add llm

LLM Prompt Example (for fallback):
System: "You are a concise assistant."
User: <original query>

## 4. Logging & Tracing
Each interaction saved as JSON in `logs/trace_<uuid>.json` containing: timestamp, query, rationale, agents_invoked, documents, final_answer.

## 5. Security & Privacy
- PDF size limit (5MB) & extension check
- Local ephemeral storage (delete or rotate logs in production)
- Environment variables for API keys
- Recommend adding auth & rate limiting for production

## 6. Limitations
- Simple summarization heuristics for some agents
- No streaming responses
- Minimal error recovery for network failures

## 7. Future Work
- Add vector store persistence options (Chroma)
- Advanced ranking & answer synthesis
- Multi-turn conversation memory
- Access control & usage metrics

## 8. Deployment
Dockerfile + docker-compose provided; run `docker compose up --build`.
