# Deployment Guide

## Deploy to Render

### Option 1: Automatic Deployment (Recommended)

1. **Fork/Clone Repository**
   ```bash
   git clone https://github.com/Sagar-Bawankule/Multi_Agentic_System.git
   cd Multi_Agentic_System
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select `Multi_Agentic_System` repository

3. **Configure Service**
   - **Name**: `multi-agentic-system` (or your preferred name)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or higher for production)

4. **Set Environment Variables**
   Add these environment variables in Render dashboard:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here (optional)
   GOOGLE_API_KEY=your_google_api_key_here (optional)
   SERPAPI_KEY=your_serpapi_key_here (optional)
   LLM_PROVIDER=groq
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be available at `https://your-service-name.onrender.com`

### Option 2: Using render.yaml (Infrastructure as Code)

1. The repository includes a `render.yaml` file for automatic configuration
2. Simply connect your repository to Render
3. Render will automatically detect and use the configuration
4. Add your environment variables in the Render dashboard

### Option 3: Manual Configuration

1. **Create New Web Service**
   - Runtime: Python 3.11
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

2. **Advanced Settings**
   - Health Check Path: `/health`
   - Auto-Deploy: Yes (for automatic deployments on git push)

## Deploy to HuggingFace Spaces

### Requirements
- HuggingFace account
- Repository with all files

### Steps

1. **Create New Space**
   - Go to [HuggingFace Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" or "Streamlit" SDK (we'll use custom Docker)

2. **Configure Space**
   - **Name**: `multi-agentic-system`
   - **SDK**: `Docker`
   - **Visibility**: Public or Private

3. **Upload Files**
   - Upload all your project files to the Space
   - Ensure `Dockerfile` is in the root directory

4. **Set Environment Variables**
   In Space settings, add:
   ```
   GROQ_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

5. **Custom Dockerfile for HuggingFace**
   Create a `Dockerfile.huggingface`:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 7860

   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

## Environment Variables Required

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Primary LLM provider API key |
| `OPENAI_API_KEY` | ❌ | Fallback LLM provider |
| `GOOGLE_API_KEY` | ❌ | Google/Gemini API key |
| `SERPAPI_KEY` | ❌ | Enhanced web search |
| `LLM_PROVIDER` | ❌ | Defaults to "groq" |

## Testing Deployment

After deployment, test these endpoints:
- Health check: `GET /health`
- Extended health: `GET /health/extended`
- Query endpoint: `POST /query` with JSON body `{"query": "test"}`
- Frontend: `GET /` (serves the web interface)

## Common Issues

### Build Failures
- Ensure `requirements.txt` has all dependencies
- Check Python version compatibility (3.11 recommended)
- Verify no missing imports in your code

### Runtime Errors
- Set all required environment variables
- Check API key validity
- Monitor logs for detailed error messages

### Performance
- Free tier has limitations (sleep after inactivity)
- Consider upgrading to paid plans for production use
- Optimize for cold start times

## Monitoring

- **Render**: Check logs in dashboard
- **HuggingFace**: Monitor Space logs and metrics
- **Health Endpoints**: Use `/health/extended` for detailed status

## Production Recommendations

1. **Use Environment Variables**: Never hardcode API keys
2. **Enable Auto-Deploy**: Connect to GitHub for automatic deployments
3. **Monitor Usage**: Set up alerts for API usage and errors
4. **Backup**: Keep your environment configuration documented
5. **Security**: Use HTTPS (automatically provided by Render/HF)

## Support

- **Render**: [Render Documentation](https://render.com/docs)
- **HuggingFace**: [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- **Repository Issues**: Open issues on GitHub for bug reports