# Load environment variables FIRST before importing any agents
try:
    from dotenv import load_dotenv
    import os
    # Force reload environment variables
    load_dotenv(override=True)
    print(f"Environment loaded - LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")
except Exception as e:
    print(f"Failed to load .env: {e}")
    # dotenv is optional; proceed if unavailable
    pass

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from typing import Optional
import uuid

# Import agents AFTER environment is loaded
from agents.controller_agent import ControllerAgent
from agents.pdf_rag_agent import PDFRAGAgent
from agents.web_search_agent import WebSearchAgent
from agents.arxiv_agent import ArxivAgent

app = FastAPI(title="Multi-Agentic System", version="0.1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate agents
pdf_agent = PDFRAGAgent()
# Auto-ingest any PDFs placed in sample_pdfs at startup (silent best-effort)
try:
    pdf_agent.ingest_all_in_dir()
except Exception:
    pass
web_agent = WebSearchAgent(serpapi_key=os.getenv("SERPAPI_KEY"))
arxiv_agent = ArxivAgent()
controller = ControllerAgent(pdf_agent=pdf_agent, web_agent=web_agent, arxiv_agent=arxiv_agent)

# Simple middleware for request logging / error capture
@app.middleware("http")
async def add_observability(request: Request, call_next):
    from time import time
    start = time()
    try:
        response = await call_next(request)
    except Exception as e:
        duration = (time() - start) * 1000
        return JSONResponse({
            "error": f"{type(e).__name__}: {e}",
            "path": request.url.path,
            "duration_ms": round(duration, 2)
        }, status_code=500)
    return response

@app.get("/")
async def root():
    if os.path.exists("frontend/index.html"):
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return {"message": "Multi-Agentic System Running"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Basic security checks
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse({"error": "Only PDF files allowed"}, status_code=400)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB limit
        return JSONResponse({"error": "File too large (limit 5MB)"}, status_code=400)
    save_path = os.path.join("sample_pdfs", file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)
    ingest_res = pdf_agent.ingest_pdf(save_path)
    return {"status": "uploaded", "ingest": ingest_res}

@app.post("/ask")
async def ask(query: str = Form(...), use_pdf: Optional[bool] = Form(False)):
    result = controller.handle(query, has_pdf=use_pdf)
    return result

@app.get("/controller/decide")
async def controller_decide(query: str):
    return controller.decide(query, has_pdf=False)

@app.get("/logs")
async def get_logs(limit: int = 50):
    logs = controller.load_logs(limit=limit)
    return {"logs": logs}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/health/extended")
async def health_extended():
    return {
        "status": "ok",
        "pdf_chunks": len(getattr(pdf_agent, "chunks", []) or []),
        "llm_provider": os.getenv("LLM_PROVIDER", "unset"),
        "web_mode": "serpapi" if os.getenv("SERPAPI_KEY") else "duckduckgo",
        "arxiv_ready": True
    }

@app.get("/agents/status")
async def agents_status():
    pdf_mode = "none"
    pdf_chunks = 0
    embeddings = False
    try:
        if pdf_agent:
            pdf_chunks = len(getattr(pdf_agent, "chunks", []) or [])
            pdf_mode = getattr(pdf_agent, "_mode", lambda: "tf")()
            embeddings = bool(getattr(pdf_agent, "use_embeddings", False) and getattr(pdf_agent, "_emb_model", None))
    except Exception:
        pass
    return {
        "controller": {"status": "ok"},
        "pdf_rag": {"status": "ok" if pdf_chunks else "empty", "chunks": pdf_chunks, "mode": pdf_mode, "embeddings_active": embeddings},
        "web_search": {"status": "ok", "serpapi": bool(os.getenv("SERPAPI_KEY"))},
        "arxiv": {"status": "ok"}
    }

@app.get("/pdf/stats")
async def pdf_stats():
    return pdf_agent.stats()

@app.post("/pdf/reset")
async def pdf_reset():
    pdf_agent.reset()
    return {"status": "reset", "stats": pdf_agent.stats()}

@app.post("/pdf/generate_from_text")
async def pdf_generate_from_text(title: str = Form(...), body: str = Form(...)):
    """Generate a simple PDF from provided text and ingest it. Lightweight pure text PDF.
    """
    try:
        filename = f"generated_{uuid.uuid4().hex[:8]}.pdf"
        path = os.path.join("sample_pdfs", filename)
        # Minimal PDF writer using reportlab if available; else fallback to plain text wrapper
        try:
            from reportlab.pdfgen import canvas  # type: ignore
            from reportlab.lib.pagesizes import letter  # type: ignore
            c = canvas.Canvas(path, pagesize=letter)
            textobject = c.beginText(40, 750)
            textobject.setFont("Helvetica", 11)
            textobject.textLine(title)
            textobject.textLine("")
            for line in body.splitlines():
                textobject.textLine(line[:120])
            c.drawText(textobject)
            c.showPage()
            c.save()
        except Exception:
            # Fallback: create a trivial PDF structure (not robust, best-effort)
            with open(path, "wb") as f:
                f.write(b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 72 720 Td (" + title[:40].encode(errors='ignore') + b") Tj ET\nendstream endobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF")
        ingest_res = pdf_agent.ingest_pdf(path)
        return {"status": "generated", "file": filename, "ingest": ingest_res}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
