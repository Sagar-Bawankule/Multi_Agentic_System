#!/bin/bash
# Render deployment script to ensure proper FastAPI startup

echo "🚀 Starting Multi-Agentic System deployment..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Start FastAPI application
echo "🌟 Starting FastAPI application..."
exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1