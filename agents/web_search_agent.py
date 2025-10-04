import requests
import urllib.parse
import datetime
from typing import Dict, Any, List, Optional

try:
    from ddgs import DDGS  # modern DuckDuckGo library (renamed from duckduckgo_search)
except ImportError:
    DDGS = None


class WebSearchAgent:
    """Web Search Agent using DuckDuckGo (free) with optional SerpAPI fallback."""

    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0 Safari/537.36'
        })

    # ✅ Free DuckDuckGo Search (using duckduckgo-search lib)
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = []
        try:
            if DDGS is None:
                raise ImportError("ddgs library not installed. Install with `pip install ddgs`.")

            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            results = [{"error": f"duckduckgo_error: {type(e).__name__}: {e}"}]
        return results

    # ✅ Optional SerpAPI (paid, only if key provided)
    def search_serpapi(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.serpapi_key:
            return []
        url = "https://serpapi.com/search"
        params = {"q": query, "engine": "google", "api_key": self.serpapi_key}
        try:
            resp = self.session.get(url, params=params, timeout=20)
            data = resp.json()
            organic = data.get("organic_results", [])[:max_results]
            return [
                {"title": r.get("title"), "link": r.get("link"), "snippet": r.get("snippet")}
                for r in organic if r
            ]
        except Exception as e:
            return [{"error": f"serpapi_error: {type(e).__name__}: {e}"}]

    def enhance_query_for_domain(self, query: str) -> str:
        """Enhance search query based on detected domain/topic"""
        query_lower = query.lower()
        
        # Business/Finance specific enhancements
        if any(keyword in query_lower for keyword in ['business', 'finance', 'economy', 'market', 'stock', 'trading']):
            if 'india' in query_lower:
                return f"{query} site:economictimes.indiatimes.com OR site:business-standard.com OR site:livemint.com"
            else:
                return f"{query} site:reuters.com/business OR site:bloomberg.com OR site:cnbc.com"
        
        # Technology news
        elif any(keyword in query_lower for keyword in ['tech', 'technology', 'startup', 'ai', 'software']):
            return f"{query} site:techcrunch.com OR site:theverge.com OR site:wired.com"
        
        # Sports news
        elif any(keyword in query_lower for keyword in ['sport', 'football', 'basketball', 'cricket', 'tennis']):
            return f"{query} site:espn.com OR site:bbc.com/sport OR site:si.com"
        
        # General news enhancement
        elif any(keyword in query_lower for keyword in ['news', 'latest', 'breaking', 'current']):
            return f"{query} site:reuters.com OR site:bbc.com OR site:cnn.com OR site:apnews.com"
        
        return query

    # ✅ Unified method with enhanced domain-specific search
    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        # Enhance query for better results
        enhanced_query = self.enhance_query_for_domain(query)
        
        # Use SerpAPI if available, otherwise DuckDuckGo
        if self.serpapi_key:
            results = self.search_serpapi(enhanced_query, max_results=max_results)
            source = "serpapi"
            if not results or "error" in str(results[0]):
                results = self.search_duckduckgo(enhanced_query, max_results=max_results)
                source = "duckduckgo_fallback"
        else:
            results = self.search_duckduckgo(enhanced_query, max_results=max_results)
            source = "duckduckgo"

        valid_results = [r for r in results if isinstance(r, dict) and "error" not in r]

        if not valid_results:
            return {"results": [], "summary": "No search results available.", "source": source}

        # Enhanced summarization with structured approach
        combined_content = " ".join(r.get("snippet", "") for r in valid_results)[:1500]
        
        if combined_content:
            try:
                # Check if we have actual news content vs generic website descriptions
                has_real_content = any(
                    keyword in combined_content.lower() 
                    for keyword in ['today', 'yesterday', 'this week', 'announced', 'reported', 'breaking', 'update', 'latest', 'new', 'recent']
                )
                
                if has_real_content:
                    # Try to use LLM for intelligent summarization
                    from agents.controller_agent import call_llm_api
                    llm_prompt = f"""Extract and summarize the KEY NEWS FACTS from these search results about "{query}":

{combined_content}

IMPORTANT: 
- Focus ONLY on actual news events, announcements, or recent developments
- Do NOT provide generic advice or suggestions about where to find news
- If the content lacks specific news facts, say "Limited current news details found"
- Present concrete information with dates, companies, numbers when available
- Keep it factual and news-focused

Summary:"""
                    
                    llm_summary = call_llm_api(llm_prompt)
                    
                    # Check if LLM gave a useful response vs generic advice
                    generic_indicators = [
                        "you can try", "you can visit", "some popular", 
                        "to get the latest", "you can also", "try searching",
                        "look for websites", "use news aggregators"
                    ]
                    
                    if (not any(marker in llm_summary for marker in ["[LLM Fallback]", "[Echo]", "[Custom LLM Error", "[LLM Provider Error"]) 
                        and not any(indicator in llm_summary.lower() for indicator in generic_indicators)):
                        summary = llm_summary
                    else:
                        # Use structured format when LLM is too generic
                        summary = self._create_structured_summary(query, valid_results, source)
                else:
                    # No real news content, use structured format
                    summary = self._create_structured_summary(query, valid_results, source)
                    
            except Exception:
                # Fallback to structured summary
                summary = self._create_structured_summary(query, valid_results, source)
        else:
            summary = f"Search completed for '{query}' but no detailed content available. (Source: {source})"

        return {"results": valid_results, "summary": summary, "source": source}
    
    def _create_structured_summary(self, query: str, results: List[Dict[str, Any]], source: str) -> str:
        """Create a structured summary from search results"""
        summary = f"**Search Results for '{query}':**\n\n"
        
        for i, r in enumerate(results[:5], 1):
            title = r.get('title', 'Untitled')
            snippet = r.get('snippet', '')[:200]
            link = r.get('link', '')
            
            # Extract domain for credibility
            domain = ""
            if link:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(link).netloc.replace('www.', '')
                except:
                    pass
            
            summary += f"**{i}. {title}**\n"
            if domain:
                summary += f"Source: {domain}\n"
            summary += f"{snippet}...\n\n"
        
        summary += f"*({len(results)} results from {source})*"
        return summary
