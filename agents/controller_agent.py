import os
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC

def call_llm_api(prompt: str) -> str:
    """Call a real LLM provider if configured, else fallback to echo.

    Supported providers (set env var LLM_PROVIDER):
      - openai  (requires OPENAI_API_KEY, optional LLM_MODEL)
      - groq    (requires GROQ_API_KEY, uses GROQ_MODEL or fallback)

    Fallback: echo placeholder to keep system functional without keys.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()  # Default to groq
    
    # Debug logging for production troubleshooting
    print(f"[DEBUG] LLM_PROVIDER: '{provider}'")
    print(f"[DEBUG] GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")
    print(f"[DEBUG] OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
    
    try:
        if provider == "openai":
            # OpenAI Python SDK v1 style
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise assistant summarizing or answering queries."},
                    {"role": "user", "content": prompt[:8000]}
                ],
                temperature=0.4,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        elif provider == "groq":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                print("[ERROR] GROQ_API_KEY not found in environment variables")
                return f"[Groq Config Error] Missing GROQ_API_KEY. Query: {prompt[:200]}"
            
            from groq import Groq
            client = Groq(api_key=groq_key)
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"[DEBUG] Using Groq model: {model}")
            
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a concise assistant summarizing or answering queries."},
                    {"role": "user", "content": prompt[:8000]}
                ],
                temperature=0.4,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        elif provider == "gemini":
            # Google AI Studio (Gemini) minimal integration
            import google.generativeai as genai  # type: ignore
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return f"[Gemini missing key] Fallback echo: {prompt[:400]}"
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt[:12000])
                if hasattr(resp, 'text') and resp.text:
                    return resp.text.strip()
                # Some responses might have candidates
                if getattr(resp, 'candidates', None):
                    for c in resp.candidates:
                        part_text = getattr(getattr(c, 'content', None), 'parts', [{}])[0].get('text') if hasattr(getattr(c,'content',None),'parts') else None
                        if part_text:
                            return part_text.strip()
                return f"[Gemini empty response] Fallback echo: {prompt[:200]}"
            except Exception as e:
                return f"[Gemini Error: {type(e).__name__}] Fallback echo: {prompt[:300]}"
        elif provider == "custom":
            # Generic JSON chat endpoint (e.g., Ollama). Expect LLM_API_BASE, LLM_MODEL
            import requests, json
            base = os.getenv("LLM_API_BASE", "http://localhost:11434")
            model = os.getenv("LLM_MODEL", "llama3")
            # Try Ollama chat endpoint first
            url = base.rstrip("/") + "/api/chat"
            payload = {"model": model, "messages": [{"role": "user", "content": prompt[:8000]}]}
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    # Ollama streams; final chunk may have 'message'. If aggregated, attempt extraction.
                    if isinstance(data, dict) and data.get("message"):
                        return data["message"].get("content", "").strip() or "[Empty response]"
                    # If server returns a sequence of JSON lines (already combined), try last message.
                    if isinstance(data, list):
                        for item in reversed(data):
                            if isinstance(item, dict) and item.get("message"):
                                return item["message"].get("content", "").strip()
                return f"[Custom LLM error HTTP {r.status_code}] Fallback echo: {prompt[:200]}"
            except Exception as e:
                return f"[Custom LLM Error: {type(e).__name__}] Fallback echo: {prompt[:200]}"
        elif provider == "echo":
            return f"[Echo]\n{prompt[:500]}"
    except Exception as e:
        error_msg = f"[LLM Provider Error: {type(e).__name__}: {str(e)}]"
        print(f"[ERROR] {error_msg}")
        return f"{error_msg} Fallback echo: {prompt[:400]}"
    
    # Fallback if provider unset or not recognized
    print(f"[WARNING] Unrecognized LLM provider: '{provider}'. Available: groq, openai, gemini, custom, echo")
    return f"[LLM Config Error] Unknown provider '{provider}'. Echo: {prompt[:500]}"

class ControllerAgent:
    """Controller agent that decides which specialized agents to invoke based on query and context."""

    def __init__(self, pdf_agent=None, web_agent=None, arxiv_agent=None, log_dir: str = "logs"):
        self.pdf_agent = pdf_agent
        self.web_agent = web_agent
        self.arxiv_agent = arxiv_agent
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def decide(self, query: str, has_pdf: bool = False) -> Dict[str, Any]:
        q_lower = (query or "").lower()
        rationale_parts: List[str] = []
        agents_to_call: List[str] = []

        # Rule-based routing (order matters for readability, we deduplicate later)
        if has_pdf or "pdf" in q_lower:
            rationale_parts.append("Detected PDF context (upload flag or keyword). Routing to PDF RAG agent.")
            agents_to_call.append("pdf_rag")
        if any(k in q_lower for k in ["recent papers", "arxiv", "research"]):
            rationale_parts.append("Detected scholarly research intent. Routing to ArXiv agent.")
            agents_to_call.append("arxiv")
        if any(k in q_lower for k in ["latest news", "recent developments", "breaking news", "current events", "sports news", "sports", "latest sports", "today's news", "news today", "current news", "business news", "latest business", "market news", "finance news", "economy news", "business updates", "stock market", "market updates", "market today", "stocks today"]):
            rationale_parts.append("Detected need for current events, sports, or business news. Routing to Web Search agent.")
            agents_to_call.append("web_search")

        if not agents_to_call:
            rationale_parts.append("No specific domain keywords detected. Falling back to generic LLM response.")
            agents_to_call.append("llm")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in agents_to_call:
            if a not in seen:
                seen.add(a)
                deduped.append(a)

        rationale = " ".join(rationale_parts)
        return {"agents": deduped, "rationale": rationale}

    def handle(self, query: str, has_pdf: bool = False) -> Dict[str, Any]:
        decision = self.decide(query, has_pdf)
        agents_used: List[str] = []
        retrieved_documents: Dict[str, Any] = {}
        answers: List[str] = []
        agent_run_info: List[Dict[str, Any]] = []

        for agent_name in decision["agents"]:
            record: Dict[str, Any] = {"agent": agent_name}
            try:
                if agent_name == "pdf_rag":
                    if not self.pdf_agent:
                        raise RuntimeError("PDF agent not configured")
                    result = self.pdf_agent.answer(query)
                    if result and result.get("answer"):
                        record["status"] = "ok"
                        record["meta"] = {"sources_count": len(result.get("sources", []))}
                        answers.append(result.get("answer", ""))
                        retrieved_documents["pdf_rag"] = result.get("sources", [])
                        agents_used.append("pdf_rag")
                    else:
                        record["status"] = "warning"
                        record["error"] = "No answer from PDF agent"
                elif agent_name == "web_search":
                    if not self.web_agent:
                        raise RuntimeError("Web search agent not configured")
                    result = self.web_agent.search_and_summarize(query)
                    if result and result.get("summary") and "error" not in result.get("summary", "").lower():
                        record["status"] = "ok"
                        record["meta"] = {"results_count": len(result.get("results", []))}
                        answers.append(result.get("summary", ""))
                        retrieved_documents["web_search"] = result.get("results", [])
                        agents_used.append("web_search")
                    else:
                        record["status"] = "warning"
                        record["error"] = "Web search failed or no results"
                elif agent_name == "arxiv":
                    if not self.arxiv_agent:
                        raise RuntimeError("Arxiv agent not configured")
                    result = self.arxiv_agent.search_and_summarize(query)
                    if result and result.get("summary") and "error" not in result.get("summary", "").lower():
                        record["status"] = "ok"
                        record["meta"] = {"papers_count": len(result.get("papers", []))}
                        answers.append(result.get("summary", ""))
                        retrieved_documents["arxiv"] = result.get("papers", [])
                        agents_used.append("arxiv")
                    else:
                        record["status"] = "warning"
                        record["error"] = "ArXiv search failed or no results"
                elif agent_name == "llm":
                    generic = call_llm_api(query)
                    record["status"] = "ok"
                    record["meta"] = {"chars": len(generic)}
                    answers.append(generic)
                    agents_used.append("llm")
                else:
                    record["status"] = "skipped"
                    record["error"] = "Unknown agent name"
            except Exception as e:  # capture per-agent error and continue
                record["status"] = "error"
                record["error"] = f"{type(e).__name__}: {e}"
            agent_run_info.append(record)

        # Synthesize final answer - use LLM to combine if multiple agents responded
        valid_answers = [a for a in answers if a and not any(marker in a for marker in ["[LLM Fallback]", "[Echo]", "[Custom LLM Error"])]
        
        if len(valid_answers) > 1:
            # Multiple agents responded - synthesize with LLM
            synthesis_prompt = f"""Synthesize a comprehensive answer from the following agent responses to the query: "{query}"

Agent Responses:
{chr(10).join(f"- {ans}" for ans in valid_answers)}

Provide a cohesive, well-structured final answer:"""
            final_answer = call_llm_api(synthesis_prompt)
        elif len(valid_answers) == 1:
            final_answer = valid_answers[0]
        elif answers:
            # Only fallback/error responses - join them
            final_answer = "\n---\n".join(a for a in answers if a)
        else:
            final_answer = "No answer generated."
            
        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "decision_rationale": decision["rationale"],
            "agents_invoked": agents_used,
            "agent_run_info": agent_run_info,
            "documents": retrieved_documents,
            "final_answer": final_answer
        }
        self._persist_log(log_record)
        return log_record

    def _persist_log(self, record: Dict[str, Any]):
        import json, uuid
        log_path = os.path.join(self.log_dir, f"trace_{uuid.uuid4().hex}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            import json
            json.dump(record, f, ensure_ascii=False, indent=2)

    def load_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        import json, glob
        files = sorted(glob.glob(os.path.join(self.log_dir, "trace_*.json")), reverse=True)[:limit]
        logs = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    logs.append(json.load(f))
            except Exception:
                continue
        return logs
