# This file explicitly tells Render this is NOT a Django app
# Framework: FastAPI
# Server: Uvicorn
# Entry point: app.py

export PYTHONPATH=/opt/render/project/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}

# Prevent Django detection
export DJANGO_SETTINGS_MODULE=""