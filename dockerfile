# Updated dockerfile that works for both training and serving
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src

# Set Python path
ENV PYTHONPATH=/app

# Expose port (for serving mode)
EXPOSE 8080
ENV PORT=8080

# Health check (for serving mode)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - can be overridden in Cloud Build
# For training: python src/cloud_orchestrator.py
# For serving: uvicorn src.main:app --host 0.0.0.0 --port $PORT
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]