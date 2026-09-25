FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies including FFmpeg and libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Pre-download models during build phase so startup is instantaneous
RUN python download_models.py

# Render dynamically assigns $PORT (defaults to 8000 locally or if unset)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI server binding to 0.0.0.0 and dynamically using Render's $PORT
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
