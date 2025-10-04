#!/bin/bash
# EXPLICIT FastAPI startup script - NOT Django

echo "🚀 Starting FastAPI Multi-Agentic System..."
echo "⚠️  This is NOT a Django application"
echo "📦 Framework: FastAPI + Uvicorn"

# Prevent any Django detection
export DJANGO_SETTINGS_MODULE=""
export DISABLE_DJANGO_DETECTION=true
export FASTAPI_APP=true

# Install dependencies
echo "📦 Installing dependencies..."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Dependencies installed successfully"

# Start FastAPI application
echo "🌟 Starting FastAPI application with Uvicorn..."
exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT