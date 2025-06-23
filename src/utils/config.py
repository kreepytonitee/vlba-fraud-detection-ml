import os

class Config:
    # GCS Bucket where the raw dataset is stored
    GCS_DATA_BUCKET = os.getenv("GCS_DATA_BUCKET", "vlba-fd-data")
    GCS_DATA_FILE = os.getenv("GCS_DATA_FILE", "transactions.csv") # Example filename
    GCS_PRODUCTION_DATA_FILE = os.getenv("GCS_DATA_FILE", "transactions_production.csv") # Example filename

    # GCS Bucket where the model artifact is stored
    GCS_MODEL_BUCKET = os.getenv("GCS_MODEL_BUCKET", "vlba-fd-model")
    GCS_MODEL_FILE = os.getenv("GCS_MODEL_FILE", "best_fraud_detection_lgbm_model.joblib")

    # Cloud Run/App Engine service name (for frontend to connect)
    BACKEND_SERVICE_URL = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8000") # Default for local testing

    # Monitoring related variables
    PREDICTION_COUNT_KEY = "prediction_count"
    PREDICTION_DISTRIBUTION_KEY = "prediction_distribution"
    PREDICTION_LATENCY_KEY = "prediction_latency"