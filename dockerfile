  # Use a Python base image
  FROM python:3.9-slim-buster

  # Set working directory
  WORKDIR /application

  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

  # Install dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

  # Copy the rest of the application code
  COPY . /app

  # Expose the port your FastAPI application will listen on
  EXPOSE 8080

  # Run the application using Uvicorn (ASGI server for FastAPI)
  # Use 0.0.0.0 to make it accessible from outside the container
  # --host 0.0.0.0 --port 8000 tells uvicorn to listen on all interfaces on port 8000
  CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]