# Use a Python base image - updated to newer version
FROM python:3.12-slim

# Set working directory to /app (project root)
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory to /app/src
COPY ./src ./src

# Set Python path to include the project root
ENV PYTHONPATH=/app

# Expose the port your FastAPI application will listen on
EXPOSE 8080

# Use environment variable for port to make it configurable
ENV PORT=8080

# Run the application using Uvicorn with full module path
CMD uvicorn src.main:app --host 0.0.0.0 --port $PORT