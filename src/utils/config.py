import os
from logger import logger

class Config:
    # GCS Bucket where the raw dataset is stored
    GCS_DATA_BUCKET = os.getenv("GCS_DATA_BUCKET", "vlba-fd-data")
    GCS_DATA_FILE = os.getenv("GCS_DATA_FILE", "transactions.csv")
    GCS_PRODUCTION_DATA_FILE = os.getenv("GCS_PRODUCTION_DATA_FILE", "transactions_production.csv")
    GCS_FEATURE_FILE = os.getenv("GCS_FEATURE_FILE", "feature_columns.json")

    # GCS Bucket where the model artifact is stored
    GCS_MODEL_BUCKET = os.getenv("GCS_MODEL_BUCKET", "vlba-fd-model")
    GCS_MODEL_FILE = os.getenv("GCS_MODEL_FILE", "fraud_detection_model.joblib")

    # Cloud Run/App Engine service name (for frontend to connect)
    BACKEND_SERVICE_URL = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8000") # Default for local testing

    # Monitoring related variables
    PREDICTION_COUNT_KEY = "prediction_count"
    PREDICTION_DISTRIBUTION_KEY = "prediction_distribution"
    PREDICTION_LATENCY_KEY = "prediction_latency"


logger.info(f"Config loaded: Model Bucket={Config.GCS_MODEL_BUCKET}, Data Bucket={Config.GCS_DATA_BUCKET}")

