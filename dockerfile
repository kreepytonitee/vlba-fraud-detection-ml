  # Use a Python base image
  FROM python:3.9-slim-buster

  # Set working directory
  WORKDIR /app

  # Install dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # Copy your application code
  COPY src/app/main.py src/app/
  COPY src/features/feature_engineering.py src/features/
  COPY src/models/predict.py src/models/
  COPY src/utils/config.py src/utils/

  # Create directories for MLflow artifacts and data if they don't exist
  # RUN mkdir -p /app/mlruns /app/data/processed /app/data/simulated

  # Command to run the application (e.g., Flask/FastAPI server)
  # The actual model path will be loaded via MLflow
  CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]