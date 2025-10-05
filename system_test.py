#!/usr/bin/env python3
"""
Quick self-test to verify the Multi-Agentic System is working properly.
Run this before deployment to catch configuration issues.
"""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test environment variable loading"""
    print("🔧 Testing environment variables...")
    load_dotenv()
    
    llm_provider = os.getenv("LLM_PROVIDER", "not_set")
    groq_key = bool(os.getenv("GROQ_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    
    print(f"   LLM_PROVIDER: {llm_provider}")
    print(f"   GROQ_API_KEY: {'✅' if groq_key else '❌'}")
    print(f"   OPENAI_API_KEY: {'✅' if openai_key else '❌'}")
    
    if not groq_key and not openai_key:
        print("   ⚠️  WARNING: No API keys found - will fallback to echo mode")
        return False
    return True

def test_llm():
    """Test LLM functionality"""
    print("\n🤖 Testing LLM integration...")
    try:
        from agents.controller_agent import call_llm_api
        result = call_llm_api("Say 'Test successful' if you can respond.")
        
        if "[LLM" in result or "Error" in result:
            print(f"   ❌ LLM Error: {result[:100]}")
            return False
        else:
            print(f"   ✅ LLM Response: {result[:60]}...")
            return True
    except Exception as e:
        print(f"   ❌ LLM Exception: {e}")
        return False

def test_agents():
    """Test agent system"""
    print("\n🕷️ Testing agent system...")
    try:
        from agents.controller_agent import ControllerAgent
        agent = ControllerAgent()
        result = agent.handle("Hello, are you working?")
        
        if result.get("final_answer"):
            print(f"   ✅ Agent Response: {result['final_answer'][:60]}...")
            print(f"   📊 Agents invoked: {result['agents_invoked']}")
            return True
        else:
            print("   ❌ No response from agent system")
            return False
    except Exception as e:
        print(f"   ❌ Agent Exception: {e}")
        return False

def test_web_search():
    """Test web search agent"""
    print("\n🌐 Testing web search...")
    try:
        from agents.web_search_agent import WebSearchAgent
        web_agent = WebSearchAgent()
        result = web_agent.search_and_summarize("latest tech news", max_results=2)
        
        if result and result.get("summary") and "error" not in result.get("summary", "").lower():
            print(f"   ✅ Web Search: {result['summary'][:60]}...")
            return True
        else:
            print(f"   ⚠️  Web Search: Limited results or error")
            return False
    except Exception as e:
        print(f"   ❌ Web Search Exception: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Multi-Agentic System Self-Test")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("LLM", test_llm),
        ("Agents", test_agents),
        ("Web Search", test_web_search)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ {name} Test Failed: {e}")
            results.append((name, False))
    
    print("\n📊 Test Summary:")
    print("-" * 20)
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All systems operational! Ready for deployment.")
    elif passed >= len(results) - 1:
        print("⚠️  System mostly functional. Minor issues detected.")
    else:
        print("🚨 Critical issues detected. Please fix before deployment.")
        
    return passed >= len(results) - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)