# Render Environment Variables Setup Guide

## Critical Issue: Missing Environment Variables

If your deployed app returns "[LLM Fallback]" responses, it means the environment variables are not properly configured in Render.

## Required Environment Variables for Render

### 1. Go to your Render Dashboard
- Navigate to your deployed service
- Click on "Environment" tab in the left sidebar

### 2. Add These Required Variables

**GROQ API (Primary LLM Provider):**
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

**Alternative: OpenAI API:**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk_your_actual_openai_api_key_here
LLM_MODEL=gpt-4o-mini
```

### 3. How to Get API Keys

**For Groq (Recommended - Free Tier Available):**
1. Go to https://console.groq.com/
2. Sign up/Login
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_`)

**For OpenAI:**
1. Go to https://platform.openai.com/
2. Sign up/Login  
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

### 4. Setting Variables in Render

1. In your Render service dashboard:
   - Click "Environment" in the left menu
   - Click "Add Environment Variable"
   - Enter the key name (e.g., `GROQ_API_KEY`)
   - Enter the value (your actual API key)
   - Click "Save"

2. **IMPORTANT:** After adding environment variables, you must:
   - Click "Manual Deploy" or trigger a new deployment
   - Wait for the deployment to complete

### 5. Verification

After deployment, test these endpoints:

**Check Environment Variables:**
```
GET https://your-app.onrender.com/debug/env
```

**Test LLM Directly:**
```
GET https://your-app.onrender.com/debug/llm
```

**Test Full System:**
```
POST https://your-app.onrender.com/ask
Content-Type: application/x-www-form-urlencoded

query=latest news&use_pdf=false
```

### 6. Common Issues

**Issue:** Still getting "[LLM Fallback]" after setting variables
**Solution:** 
- Double-check the variable names (case-sensitive)
- Ensure no extra spaces in keys or values
- Redeploy the service after setting variables
- Check debug endpoints to verify variables are loaded

**Issue:** "GROQ_API_KEY not found" error
**Solution:**
- Verify the API key is correctly copied (no missing characters)
- Ensure the variable name is exactly `GROQ_API_KEY`
- Test the API key works by calling Groq API directly

**Issue:** API rate limits or quota exceeded
**Solution:**
- Check your Groq/OpenAI account usage
- Consider upgrading to paid tier if needed
- Switch to alternative provider temporarily

### 7. Optional Variables

```
# Web search enhancement
SERPAPI_KEY=your_serpapi_key_here

# Server configuration  
PORT=8000

# RAG configuration
RAG_EMBEDDINGS=1
RAG_EMBED_MODEL=all-MiniLM-L6-v2
```

## Security Notes

- Never commit real API keys to Git
- Use Render's environment variable system, not .env files in production
- Regularly rotate your API keys
- Monitor API usage to prevent unexpected charges

## Support

If you're still experiencing issues:
1. Check the debug endpoints first
2. Verify your API keys work outside of the application
3. Ensure you've redeployed after setting environment variables
4. Check Render logs for specific error messages