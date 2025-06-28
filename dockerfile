  # Use a Python base image
  FROM python:3.9-slim-buster

  # Set working directory
  WORKDIR /app

  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

  # Install dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

  # Copy the entire src directory (including its subdirectories and __init__.py files)
  # This is a more robust way to ensure all your modules are copied correctly.
  COPY src/ ./src/

  # Expose the port your FastAPI application will listen on
  EXPOSE 8080

  # Run the application using Uvicorn (ASGI server for FastAPI)
  # Use 0.0.0.0 to make it accessible from outside the container
  # --host 0.0.0.0 --port 8*** tells uvicorn to listen on all interfaces on port 8***
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]