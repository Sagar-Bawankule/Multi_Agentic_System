FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs vector_store test_vector_store sample_pdfs

# Use PORT environment variable from Render
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health', timeout=5)" || exit 1

# Start with dynamic port
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT