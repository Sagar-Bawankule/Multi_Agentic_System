import requests
import urllib.parse
import datetime
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, quote_plus
from datetime import datetime, timedelta

try:
    from ddgs import DDGS  # modern DuckDuckGo library (renamed from duckduckgo_search)
except ImportError:
    DDGS = None

try:
    import feedparser  # For RSS feeds
except ImportError:
    feedparser = None


class WebSearchAgent:
    """Enhanced Web Search Agent with multiple search strategies and intelligent result processing."""

    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
        })
        
        # Trusted news sources by category
        self.news_sources = {
            'general': ['reuters.com', 'apnews.com', 'bbc.com', 'cnn.com', 'npr.org'],
            'business': ['bloomberg.com', 'cnbc.com', 'wsj.com', 'ft.com', 'marketwatch.com'],
            'tech': ['techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com', 'venturebeat.com'],
            'sports': ['espn.com', 'si.com', 'bleacherreport.com', 'cbssports.com'],
            'science': ['nature.com', 'sciencemag.org', 'newscientist.com', 'scientificamerican.com']
        }
        
        # Time-sensitive keywords that indicate need for recent results
        self.time_keywords = [
            'latest', 'recent', 'today', 'yesterday', 'this week', 'this month',
            'breaking', 'current', 'now', 'update', 'new', 'just', 'announced',
            'happening', 'live', '2024', '2025','recent','news','trending','breking'
        ]

    # ✅ Enhanced DuckDuckGo Search with better filtering and error handling
    def search_duckduckgo(self, query: str, max_results: int = 8, time_filter: str = None) -> List[Dict[str, Any]]:
        results = []
        try:
            if DDGS is None:
                raise ImportError("ddgs library not installed. Install with `pip install ddgs`.")

            # Use time filter for recent searches
            search_params = {'max_results': max_results * 2}  # Get more results to filter
            if time_filter:
                search_params['timelimit'] = time_filter  # 'd' for day, 'w' for week, 'm' for month

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, **search_params))
                
                # Filter and enhance results
                for r in raw_results[:max_results]:
                    title = r.get("title", "").strip()
                    link = r.get("href", "").strip()
                    snippet = r.get("body", "").strip()
                    
                    # Skip low-quality results
                    if self._is_quality_result(title, snippet, link):
                        # Extract domain for source credibility
                        domain = self._extract_domain(link)
                        
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet,
                            "domain": domain,
                            "source_type": self._classify_source(domain),
                            "relevance_score": self._calculate_relevance(query, title, snippet)
                        })
                        
        except Exception as e:
            print(f"[DEBUG] DuckDuckGo search error: {type(e).__name__}: {e}")
            results = [{"error": f"duckduckgo_error: {type(e).__name__}: {e}"}]
        
        # Sort by relevance score
        if results and "error" not in results[0]:
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
        return results[:max_results]

    def _is_quality_result(self, title: str, snippet: str, link: str) -> bool:
        """Filter out low-quality search results"""
        if not title or not snippet or not link:
            return False
            
        # Skip spam/low-quality indicators
        spam_indicators = [
            'click here', 'buy now', 'download now', 'free download',
            'win money', 'get rich', 'lose weight', 'miracle cure'
        ]
        
        content = (title + " " + snippet).lower()
        if any(spam in content for spam in spam_indicators):
            return False
            
        # Skip very short or repetitive content
        if len(snippet) < 50 or title.count(title.split()[0] if title.split() else '') > 3:
            return False
            
        return True
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            domain = urlparse(url).netloc.lower()
            return domain.replace('www.', '') if domain.startswith('www.') else domain
        except:
            return ''
    
    def _classify_source(self, domain: str) -> str:
        """Classify the source type for credibility assessment"""
        for category, sources in self.news_sources.items():
            if any(source in domain for source in sources):
                return f'trusted_{category}'
        
        # Check for other known source types
        if any(indicator in domain for indicator in ['edu', 'gov', 'org']):
            return 'institutional'
        elif any(indicator in domain for indicator in ['wikipedia', 'wiki']):
            return 'wiki'
        else:
            return 'general'
    
    def _calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """Calculate relevance score for ranking results"""
        query_words = set(query.lower().split())
        content = (title + " " + snippet).lower()
        
        # Base score from keyword matches
        matches = sum(1 for word in query_words if word in content)
        score = matches / len(query_words) if query_words else 0
        
        # Boost for time-sensitive content if query is time-sensitive
        if any(keyword in query.lower() for keyword in self.time_keywords):
            if any(keyword in content for keyword in self.time_keywords):
                score += 0.3
        
        # Boost for trusted sources
        domain = self._extract_domain('')
        if 'trusted' in self._classify_source(domain):
            score += 0.2
            
        return min(score, 1.0)

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

    def enhance_query_for_domain(self, query: str) -> Tuple[str, str]:
        """Enhance search query based on detected domain/topic and time sensitivity"""
        query_lower = query.lower()
        enhanced_query = query
        time_filter = None
        
        # Detect time sensitivity
        is_time_sensitive = any(keyword in query_lower for keyword in self.time_keywords)
        if is_time_sensitive:
            if any(word in query_lower for word in ['today', 'latest', 'breaking', 'now', 'current']):
                time_filter = 'd'  # Last day
            elif any(word in query_lower for word in ['this week', 'recent', 'new']):
                time_filter = 'w'  # Last week
            elif 'this month' in query_lower:
                time_filter = 'm'  # Last month
        
        # Domain-specific enhancements
        domain_detected = None
        
        # Business/Finance
        business_keywords = ['business', 'finance', 'economy', 'market', 'stock', 'trading', 'earnings', 'revenue', 'ipo', 'merger', 'acquisition']
        if any(keyword in query_lower for keyword in business_keywords):
            domain_detected = 'business'
            if is_time_sensitive:
                enhanced_query = f"{query} (site:bloomberg.com OR site:cnbc.com OR site:reuters.com/business OR site:wsj.com)"
            else:
                enhanced_query = f"{query} business news finance"
        
        # Technology
        tech_keywords = ['tech', 'technology', 'startup', 'ai', 'artificial intelligence', 'software', 'hardware', 'app', 'platform', 'github', 'openai', 'google', 'microsoft', 'apple']
        if any(keyword in query_lower for keyword in tech_keywords):
            domain_detected = 'tech'
            if is_time_sensitive:
                enhanced_query = f"{query} (site:techcrunch.com OR site:theverge.com OR site:wired.com OR site:arstechnica.com)"
            else:
                enhanced_query = f"{query} technology news tech"
        
        # Sports
        elif any(keyword in query_lower for keyword in ['sport', 'sports', 'football', 'basketball', 'cricket', 'tennis', 'soccer', 'baseball', 'hockey', 'olympics', 'fifa', 'nfl', 'nba']):
            domain_detected = 'sports'
            if is_time_sensitive:
                enhanced_query = f"{query} (site:espn.com OR site:si.com OR site:bleacherreport.com OR site:bbc.com/sport)"
            else:
                enhanced_query = f"{query} sports news"
        
        # Science/Research
        elif any(keyword in query_lower for keyword in ['science', 'research', 'study', 'climate', 'space', 'nasa', 'medicine', 'health', 'vaccine', 'covid']):
            domain_detected = 'science'
            if is_time_sensitive:
                enhanced_query = f"{query} (site:nature.com OR site:sciencemag.org OR site:newscientist.com)"
            else:
                enhanced_query = f"{query} science research news"
        
        # General news
        elif any(keyword in query_lower for keyword in ['news', 'breaking', 'update', 'report', 'announcement', 'event']) or is_time_sensitive:
            domain_detected = 'general_news'
            if is_time_sensitive:
                enhanced_query = f"{query} (site:reuters.com OR site:bbc.com OR site:cnn.com OR site:apnews.com)"
            else:
                enhanced_query = f"{query} news"
        
        # Add time-specific terms for better results
        if is_time_sensitive and not any(time_word in enhanced_query.lower() for time_word in ['today', 'latest', 'recent', '2024', '2025']):
            current_year = datetime.now().year
            enhanced_query = f"{enhanced_query} {current_year}"
        
        return enhanced_query, time_filter

    # ✅ Advanced unified search with multiple strategies and intelligent processing
    def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        print(f"[DEBUG] WebSearch starting for query: '{query}'")
        
        # Enhanced query processing
        enhanced_query, time_filter = self.enhance_query_for_domain(query)
        print(f"[DEBUG] Enhanced query: '{enhanced_query}', time_filter: {time_filter}")
        
        # Multi-strategy search approach
        all_results = []
        search_sources = []
        
        # Strategy 1: Enhanced DuckDuckGo search
        ddg_results = self.search_duckduckgo(enhanced_query, max_results=max_results, time_filter=time_filter)
        if ddg_results and "error" not in str(ddg_results[0]):
            all_results.extend(ddg_results)
            search_sources.append("duckduckgo")
        
        # Strategy 2: SerpAPI if available (as fallback or primary)
        if self.serpapi_key and (not all_results or len(all_results) < max_results // 2):
            serp_results = self.search_serpapi(enhanced_query, max_results=max_results)
            if serp_results and "error" not in str(serp_results[0]):
                # Merge results, avoiding duplicates
                existing_links = {r.get('link', '') for r in all_results}
                for result in serp_results:
                    if result.get('link', '') not in existing_links:
                        all_results.append(result)
                search_sources.append("serpapi")
        
        # Strategy 3: Fallback search with original query if enhanced query failed
        if not all_results and enhanced_query != query:
            print(f"[DEBUG] Trying fallback search with original query")
            fallback_results = self.search_duckduckgo(query, max_results=max_results)
            if fallback_results and "error" not in str(fallback_results[0]):
                all_results.extend(fallback_results)
                search_sources.append("duckduckgo_fallback")
        
        # Filter and rank results
        valid_results = [r for r in all_results if isinstance(r, dict) and "error" not in r]
        
        if not valid_results:
            return {
                "results": [], 
                "summary": f"No valid search results found for '{query}'. Please try rephrasing your query or check your internet connection.", 
                "source": "none",
                "query_enhanced": enhanced_query,
                "search_strategies": search_sources
            }
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(valid_results)
        sorted_results = sorted(unique_results, key=lambda x: x.get('relevance_score', 0), reverse=True)[:max_results]
        
        # Advanced summarization
        summary = self._generate_advanced_summary(query, sorted_results, search_sources)
        
        return {
            "results": sorted_results, 
            "summary": summary, 
            "source": "+".join(search_sources),
            "query_enhanced": enhanced_query,
            "total_found": len(valid_results),
            "search_strategies": search_sources
        }
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL and title similarity"""
        seen_links = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            link = result.get('link', '')
            title = result.get('title', '').lower().strip()
            
            # Skip if we've seen this exact link
            if link in seen_links:
                continue
                
            # Skip if we've seen a very similar title
            title_words = set(title.split())
            is_similar = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words & seen_words) / max(len(title_words), len(seen_words), 1) > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_results.append(result)
                seen_links.add(link)
                seen_titles.add(title)
                
        return unique_results
    
    def _generate_advanced_summary(self, query: str, results: List[Dict[str, Any]], sources: List[str]) -> str:
        """Generate an advanced summary using multiple approaches"""
        if not results:
            return "No results to summarize."
        
        # Combine content from top results
        combined_content = ""
        source_info = []
        
        for i, result in enumerate(results[:5]):
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            domain = result.get('domain', '')
            source_type = result.get('source_type', 'general')
            
            combined_content += f"Result {i+1}: {title}. {snippet} "
            source_info.append({
                'title': title,
                'domain': domain,
                'source_type': source_type,
                'snippet': snippet[:150]
            })
        
        # Try LLM summarization first
        try:
            from agents.controller_agent import call_llm_api
            
            # Determine the type of summary needed
            query_lower = query.lower()
            is_news_query = any(keyword in query_lower for keyword in self.time_keywords + ['news', 'update', 'report'])
            
            if is_news_query:
                llm_prompt = f"""Analyze these search results for recent news about "{query}" and provide a concise news summary:

{combined_content[:2000]}

Provide a factual news summary that includes:
1. Key events or developments
2. Important dates, numbers, or details mentioned
3. Main parties/organizations involved
4. Current status or implications

Keep it concise but informative. Focus on facts, not speculation.

News Summary:"""
            else:
                llm_prompt = f"""Summarize these search results about "{query}" in a helpful and informative way:

{combined_content[:2000]}

Provide a comprehensive summary that:
1. Answers the user's query directly
2. Includes key facts and details
3. Mentions important sources or references
4. Is well-structured and easy to read

Summary:"""
            
            llm_summary = call_llm_api(llm_prompt)
            
            # Validate LLM response quality
            if (llm_summary and 
                not any(marker in llm_summary for marker in ["[LLM Fallback]", "[Echo]", "[Error"]) and
                len(llm_summary.strip()) > 50 and
                not any(generic in llm_summary.lower() for generic in ["you can try", "visit websites", "search for more"])):
                
                # Add source information
                trusted_sources = [info for info in source_info if 'trusted' in info.get('source_type', '')]
                if trusted_sources:
                    source_names = list(set(info['domain'] for info in trusted_sources[:3]))
                    llm_summary += f"\n\n*Sources include: {', '.join(source_names)}*"
                
                return llm_summary
        
        except Exception as e:
            print(f"[DEBUG] LLM summarization failed: {e}")
        
        # Fallback to structured summary
        return self._create_structured_summary(query, results, "+".join(sources))
    
    def _create_structured_summary(self, query: str, results: List[Dict[str, Any]], source: str) -> str:
        """Create a structured summary from search results"""
        summary = f"**Search Results for '{query}':**\n\n"
        
        for i, r in enumerate(results[:5], 1):
            title = r.get('title', 'Untitled')
            snippet = r.get('snippet', '')[:200]
            domain = r.get('domain', '')
            source_type = r.get('source_type', 'general')
            
            # Add credibility indicator
            credibility = "📰" if 'trusted' in source_type else "🌐" if source_type == 'institutional' else "📄"
            
            summary += f"**{credibility} {i}. {title}**\n"
            if domain:
                summary += f"*Source: {domain}*\n"
            summary += f"{snippet}{'...' if len(r.get('snippet', '')) > 200 else ''}\n\n"
        
        # Add search metadata
        summary += f"\n*Found {len(results)} results from {source}*"
        
        # Add tips for better results if query seems problematic
        if len(results) < 3:
            summary += "\n\n💡 *Try rephrasing your query or adding more specific terms for better results.*"
        
        return summary
    
    def get_news_feed(self, category: str = 'general', max_items: int = 5) -> Dict[str, Any]:
        """Get news from RSS feeds as an additional source (experimental)"""
        if not feedparser:
            return {"error": "feedparser library not available"}
        
        rss_feeds = {
            'general': ['http://feeds.reuters.com/reuters/topNews', 'http://rss.cnn.com/rss/edition.rss'],
            'tech': ['http://feeds.arstechnica.com/arstechnica/index', 'https://techcrunch.com/feed/'],
            'business': ['http://feeds.reuters.com/reuters/businessNews', 'https://feeds.bloomberg.com/markets/news.rss']
        }
        
        feed_urls = rss_feeds.get(category, rss_feeds['general'])
        items = []
        
        for feed_url in feed_urls[:2]:  # Limit to 2 feeds per category
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_items//2]:
                    items.append({
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', '')[:300]
                    })
            except Exception as e:
                print(f"[DEBUG] RSS feed error for {feed_url}: {e}")
                continue
        
        return {"items": items[:max_items], "source": "rss_feeds", "category": category}
    
    def validate_search_quality(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Validate the quality of search results and provide improvement suggestions"""
        if not results:
            return {
                "quality_score": 0,
                "issues": ["No results found"],
                "suggestions": ["Try rephrasing your query", "Use more specific terms", "Check spelling"]
            }
        
        issues = []
        suggestions = []
        quality_score = 0.5  # Base score
        
        # Check for result diversity
        domains = [r.get('domain', '') for r in results]
        unique_domains = len(set(domains))
        if unique_domains < len(results) * 0.7:
            issues.append("Low source diversity")
            suggestions.append("Try broader search terms for more diverse sources")
        else:
            quality_score += 0.2
        
        # Check for trusted sources
        trusted_count = sum(1 for r in results if 'trusted' in r.get('source_type', ''))
        if trusted_count > 0:
            quality_score += 0.2 * (trusted_count / len(results))
        else:
            issues.append("Few trusted news sources")
            suggestions.append("Add terms like 'news' or 'official' for more authoritative sources")
        
        # Check content quality
        avg_snippet_length = sum(len(r.get('snippet', '')) for r in results) / len(results)
        if avg_snippet_length < 100:
            issues.append("Short result snippets")
            suggestions.append("Try more specific queries for detailed results")
        else:
            quality_score += 0.1
        
        return {
            "quality_score": min(quality_score, 1.0),
            "issues": issues,
            "suggestions": suggestions,
            "result_count": len(results),
            "trusted_sources": trusted_count,
            "unique_domains": unique_domains
        }
