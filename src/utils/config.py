import os
from src.utils.logger import logger

class Config:
    # # GCS Bucket where the raw dataset is stored
    # GCS_DATA_BUCKET = os.getenv("GCS_DATA_BUCKET", "vlba-fd-data")
    # GCS_DATA_FILE = os.getenv("GCS_DATA_FILE", "transactions.csv")
    # GCS_PRODUCTION_DATA_FILE = os.getenv("GCS_PRODUCTION_DATA_FILE", "transactions_production.csv")
    # GCS_FEATURE_FILE = os.getenv("GCS_FEATURE_FILE", "feature_columns.json")

    # # GCS Bucket where the model artifact is stored
    # GCS_MODEL_BUCKET = os.getenv("GCS_MODEL_BUCKET", "vlba-fd-model")
    # GCS_MODEL_FILE = os.getenv("GCS_MODEL_FILE", "fraud_detection_model.joblib")
   
    # GCS Bucket where the raw dataset and processed data are stored
    GCS_DATA_BUCKET = os.getenv("GCS_DATA_BUCKET", "vlba-fd-data-bucket")
    GCS_RAW_DATA_FILE = os.getenv("GCS_RAW_DATA_FILE", "transactions.csv")
    GCS_TRAIN_DATA_FILE = os.getenv("GCS_TRAIN_DATA_FILE", "transactions_train.csv")
    GCS_PRODUCTION_DATA_FILE = os.getenv("GCS_PRODUCTION_DATA_FILE", "transactions_production.csv")
    GCS_FEATURE_ENGINEERED_TRAIN_FILE = os.getenv("GCS_FEATURE_ENGINEERED_TRAIN_FILE", "feature_engineered.csv")
    GCS_FEATURE_ENGINEERED_PROD_FILE = os.getenv("GCS_FEATURE_ENGINEERED_PROD_FILE", "production_feature_engineered.csv")
    GCS_PREDICTIONS_FILE = os.getenv("GCS_PREDICTIONS_FILE", "production_predictions.csv")

    # GCS Bucket for lookup tables (e.g., target encodings, fraud rates)
    GCS_LOOKUPS_BUCKET = os.getenv("GCS_LOOKUPS_BUCKET", "vlba-fd-lookups-bucket")

    # Add other lookup files here if they were accessed by direct paths in scripts
    # e.g., GCS_DAY_LOOKUP_FILE = "Fraud_Rate_By_Day_lookup.csv"

    # GCS Bucket where the model artifact is stored
    GCS_MODEL_BUCKET = os.getenv("GCS_MODEL_BUCKET", "vlba-fd-model-bucket")
    GCS_MODEL_FILE = os.getenv("GCS_MODEL_FILE", "model.pkl") # Saved model
    GCS_FEATURE_COLUMNS_FILE = os.getenv("GCS_FEATURE_COLUMNS_FILE", "feature_columns.txt") # Saved input columns for model

    # Cloud Run/App Engine service name (for frontend to connect)
    BACKEND_SERVICE_URL = os.getenv("BACKEND_SERVICE_URL", "http://localhost:8000") # Default for local testing

    # Monitoring related variables
    PREDICTION_COUNT_KEY = "prediction_count"
    PREDICTION_DISTRIBUTION_KEY = "prediction_distribution"
    PREDICTION_LATENCY_KEY = "prediction_latency"


logger.info(f"Config loaded: Model Bucket={Config.GCS_MODEL_BUCKET}, Data Bucket={Config.GCS_DATA_BUCKET}")

