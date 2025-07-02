  # Use a Python base image - updated to newer version
  FROM python:3.12-slim

  # Set working directory
  WORKDIR /app/src

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
  COPY ./src ./src
  ENV PYTHONPATH=/app
  
  # Expose the port your FastAPI application will listen on
  EXPOSE 8080

  # Run the application using Uvicorn (ASGI server for FastAPI)
  # Use 0.0.0.0 to make it accessible from outside the container
  # --host 0.0.0.0 --port 8080 tells uvicorn to listen on all interfaces on port 8080
  CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]