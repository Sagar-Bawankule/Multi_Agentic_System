#!/usr/bin/env python3
"""
Quick system test to verify the LLM fixes are working properly.
"""

import os
from dotenv import load_dotenv

def test_environment_loading():
    """Test that environment variables load correctly"""
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "")
    groq_key = bool(os.getenv("GROQ_API_KEY"))
    
    print(f"✓ Environment test:")
    print(f"  LLM_PROVIDER: {provider}")
    print(f"  GROQ_API_KEY present: {groq_key}")
    return provider and groq_key

def test_llm_api():
    """Test the LLM API function directly"""
    try:
        from agents.controller_agent import call_llm_api
        result = call_llm_api("Say 'Hello from LLM test'")
        
        print(f"✓ LLM API test:")
        print(f"  Result: {result[:100]}...")
        
        # Check if it's a real LLM response (not fallback)
        is_real_llm = not any(marker in result for marker in ["[LLM Fallback]", "[Groq Config Error]", "[Echo]"])
        print(f"  Real LLM response: {is_real_llm}")
        return is_real_llm
    except Exception as e:
        print(f"✗ LLM API test failed: {e}")
        return False

def test_controller_agent():
    """Test the controller agent"""
    try:
        from agents.controller_agent import ControllerAgent
        agent = ControllerAgent()
        response = agent.handle("What is AI?")
        
        print(f"✓ Controller agent test:")
        print(f"  Decision: {response.get('decision_rationale', 'N/A')}")
        print(f"  Agents invoked: {response.get('agents_invoked', [])}")
        print(f"  Final answer: {response.get('final_answer', '')[:100]}...")
        
        # Check if we got a real response
        final_answer = response.get('final_answer', '')
        is_good_response = final_answer and not any(marker in final_answer for marker in ["[LLM Fallback]", "[Groq Config Error]"])
        print(f"  Good response: {is_good_response}")
        return is_good_response
    except Exception as e:
        print(f"✗ Controller agent test failed: {e}")
        return False

def main():
    print("=== Multi-Agentic System Test ===\n")
    
    # Run tests
    env_ok = test_environment_loading()
    print()
    
    llm_ok = test_llm_api()
    print()
    
    controller_ok = test_controller_agent()
    print()
    
    # Summary
    all_good = env_ok and llm_ok and controller_ok
    print("=== Test Summary ===")
    print(f"Environment: {'✓' if env_ok else '✗'}")
    print(f"LLM API: {'✓' if llm_ok else '✗'}")
    print(f"Controller: {'✓' if controller_ok else '✗'}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_good else '❌ SOME TESTS FAILED'}")
    
    if all_good:
        print("\n🎉 System is ready for deployment!")
    else:
        print("\n⚠️  Please fix the failing tests before deploying.")

if __name__ == "__main__":
    main()