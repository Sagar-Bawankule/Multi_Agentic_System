import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Explicit log level and no reload to keep process stable
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
