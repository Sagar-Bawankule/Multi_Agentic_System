#!/bin/bash
# RENDER OVERRIDE: This is FastAPI, not Django
# DO NOT run: gunicorn your_application.wsgi  
# CORRECT command: python -m uvicorn app:app --host 0.0.0.0 --port $PORT

export PYTHONUNBUFFERED=1
export FASTAPI_APP=true  
export NOT_DJANGO=true

# Force disable Django detection
unset DJANGO_SETTINGS_MODULE
export DISABLE_DJANGO_DETECTION=true

echo "🚀 RENDER OVERRIDE: Starting FastAPI application"
echo "❌ NOT Django (no wsgi, no gunicorn needed)"
echo "✅ FastAPI + Uvicorn"

exec python -m uvicorn app:app --host 0.0.0.0 --port $PORT