#!/usr/bin/env python3
"""
Enhanced Web Search Agent Test Suite
Tests all the new features and improvements made to the web search agent.
"""

from dotenv import load_dotenv
load_dotenv()

from agents.controller_agent import ControllerAgent
from agents.web_search_agent import WebSearchAgent
from agents.pdf_rag_agent import PDFRAGAgent
from agents.arxiv_agent import ArxivAgent
import json

def test_web_search_enhancements():
    """Test all the enhanced features of the web search agent."""
    
    print("🔍 Enhanced Web Search Agent Test Suite")
    print("=" * 50)
    
    # Initialize agents
    web_agent = WebSearchAgent()
    controller = ControllerAgent(
        web_agent=web_agent,
        pdf_agent=PDFRAGAgent(),
        arxiv_agent=ArxivAgent()
    )
    
    # Test cases covering different domains and query types
    test_queries = [
        {
            "query": "latest AI news today",
            "expected_domain": "tech",
            "expected_agent": "web_search"
        },
        {
            "query": "recent business news stock market",
            "expected_domain": "business", 
            "expected_agent": "web_search"
        },
        {
            "query": "sports news today NBA",
            "expected_domain": "sports",
            "expected_agent": "web_search"
        },
        {
            "query": "breaking tech news OpenAI",
            "expected_domain": "tech",
            "expected_agent": "web_search"
        },
        {
            "query": "what is machine learning",
            "expected_domain": "general",
            "expected_agent": "llm"
        }
    ]
    
    print("\n🧪 Testing Controller Routing...")
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{test['query']}'")
        
        # Test routing decision
        decision = controller.decide(test['query'])
        print(f"   Expected Agent: {test['expected_agent']}")
        print(f"   Routed to: {decision['agents']}")
        print(f"   Rationale: {decision['rationale']}")
        
        # Test query enhancement
        enhanced_query, time_filter = web_agent.enhance_query_for_domain(test['query'])
        print(f"   Enhanced Query: {enhanced_query}")
        if time_filter:
            print(f"   Time Filter: {time_filter}")
    
    print("\n🎯 Testing Advanced Web Search...")
    
    # Test advanced search with a tech query
    test_query = "latest OpenAI news today"
    print(f"\nSearching for: '{test_query}'")
    
    result = web_agent.search_and_summarize(test_query, max_results=5)
    
    print(f"✅ Search completed!")
    print(f"   Total results found: {result.get('total_found', 0)}")
    print(f"   Search strategies used: {result.get('search_strategies', [])}")
    print(f"   Enhanced query: {result.get('query_enhanced', '')}")
    
    # Test result quality validation
    if result.get('results'):
        quality = web_agent.validate_search_quality(result['results'], test_query)
        print(f"\n📊 Result Quality Assessment:")
        print(f"   Quality Score: {quality['quality_score']:.2f}")
        print(f"   Result Count: {quality['result_count']}")
        print(f"   Trusted Sources: {quality['trusted_sources']}")
        print(f"   Unique Domains: {quality['unique_domains']}")
        if quality.get('issues'):
            print(f"   Issues: {quality['issues']}")
        if quality.get('suggestions'):
            print(f"   Suggestions: {quality['suggestions']}")
    
    # Show summary sample
    print(f"\n📰 Summary Sample:")
    print(f"   {result['summary'][:300]}...")
    
    print("\n🔄 Testing Full Controller Integration...")
    
    # Test full controller execution with web search
    response = controller.handle("latest tech news about artificial intelligence")
    print(f"   Decision: {response['decision_rationale']}")
    print(f"   Agents Invoked: {response['agents_invoked']}")
    print(f"   Answer Sample: {response['final_answer'][:200]}...")
    
    # Test different query types
    print(f"\n📋 Testing Query Type Detection:")
    
    query_types = [
        "latest news",          # Time-sensitive
        "tech news today",      # Tech + time-sensitive  
        "business updates",     # Business
        "sports scores",        # Sports
        "recent research",      # Academic
        "what is Python"        # General knowledge
    ]
    
    for query in query_types:
        decision = controller.decide(query)
        print(f"   '{query}' → {decision['agents'][0] if decision['agents'] else 'none'}")
    
    print("\n✅ Enhanced Web Search Agent Test Complete!")
    print("\nKey Improvements Demonstrated:")
    print("  • Intelligent query enhancement with domain detection")
    print("  • Time-sensitive search filtering")
    print("  • Multi-strategy search approach")
    print("  • Advanced result filtering and deduplication")
    print("  • Source credibility assessment")
    print("  • Quality validation and suggestions")
    print("  • Improved controller routing")
    print("  • Better error handling and debugging")

if __name__ == "__main__":
    test_web_search_enhancements()