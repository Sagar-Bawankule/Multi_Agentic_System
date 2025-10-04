import requests
from typing import Dict, Any, List
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

class ArxivAgent:
    """Queries the ArXiv API and summarizes results."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_results
        }
        attempts = 0
        last_error: str | None = None
        while attempts < 2:
            attempts += 1
            try:
                resp = requests.get(ARXIV_API, params=params, timeout=20, headers={"User-Agent": "MultiAgentBot/0.1"})
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}"
                    continue
                root = ET.fromstring(resp.text)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                papers: List[Dict[str, Any]] = []
                for entry in root.findall('atom:entry', ns):
                    title_el = entry.find('atom:title', ns)
                    summary_el = entry.find('atom:summary', ns)
                    published_el = entry.find('atom:published', ns)
                    
                    title = title_el.text.strip() if title_el is not None and title_el.text else ''
                    summary = summary_el.text.strip() if summary_el is not None and summary_el.text else ''
                    published = published_el.text.strip()[:10] if published_el is not None and published_el.text else ''
                    
                    # Get ArXiv ID from the entry ID
                    id_el = entry.find('atom:id', ns)
                    arxiv_id = id_el.text.split('/')[-1] if id_el is not None and id_el.text else ''
                    
                    # Get authors
                    authors = []
                    for author_el in entry.findall('atom:author', ns):
                        name_el = author_el.find('atom:name', ns)
                        if name_el is not None and name_el.text:
                            authors.append(name_el.text.strip())
                    
                    link_el = entry.find("atom:link[@type='text/html']", ns)
                    link = link_el.attrib.get('href') if link_el is not None else f"https://arxiv.org/abs/{arxiv_id}"
                    
                    papers.append({
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "arxiv_id": arxiv_id,
                        "published": published,
                        "authors": authors[:3]  # Limit to first 3 authors
                    })
                return papers
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
        if last_error:
            return [{"error": last_error}]
        return []

    def search_and_summarize(self, query: str) -> Dict[str, Any]:
        papers = self.search(query)
        if papers and isinstance(papers[0], dict) and papers[0].get("error"):
            return {"papers": papers, "summary": f"ArXiv search failed: {papers[0]['error']}"}
            
        if not papers:
            return {"papers": [], "summary": "No relevant papers found on ArXiv."}
            
        # Create structured summary with LLM
        paper_summaries = []
        for paper in papers:
            if paper.get("title") and paper.get("summary"):
                paper_summaries.append(f"**{paper['title']}**: {paper['summary'][:200]}...")
                
        if paper_summaries:
            from agents.controller_agent import call_llm_api
            llm_summary_prompt = f"""Summarize the following ArXiv papers related to "{query}":

{chr(10).join(paper_summaries)}

Provide a concise overview of the main research themes and findings:"""
            
            llm_summary = call_llm_api(llm_summary_prompt)
            
            # Use LLM summary if available, otherwise create structured summary
            if not any(marker in llm_summary for marker in ["[LLM Fallback]", "[Echo]", "[Custom LLM Error"]):
                summary = f"Recent ArXiv Research:\n{llm_summary}"
            else:
                # Fallback to structured summary
                titles = [p.get("title", "Untitled") for p in papers if p.get("title")]
                summary = f"Found {len(papers)} papers: {'; '.join(titles[:3])}{'...' if len(titles) > 3 else ''}"
        else:
            summary = "Papers found but no usable content extracted."
            
        return {"papers": papers, "summary": summary}
