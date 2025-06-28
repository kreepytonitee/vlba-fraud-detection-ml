from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import os
import time
import io
from typing import List, Optional, Dict, Any
from datetime import datetime
from google.cloud import storage
from collections import Counter
from src.utils.config import Config
from src.utils.logger import logger
from src.features.feature_engineering import apply_feature_engineering # Import the function

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent bank transactions."
)

# --- CORS Configuration ---
# IMPORTANT: Adjust 'allow_origins' in production for security.
# For development/showcase, '*' is often used, but specify your frontend URL
# (e.g., "https://your-website-bucket-name.storage.googleapis.com") for production.
origins = [
    "https://storage.googleapis.com/vlba-fd-frontend/index.html" # Allows all origins for development/testing.
    # Replace "*" with your actual static website URL for production security:
    # "https://your-website-bucket-name.storage.googleapis.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, etc.
    allow_headers=["*"], # Allows all headers
)
# --- End CORS Configuration ---

# Global variables for model and monitoring
model = None
production_dataset: pd.DataFrame = pd.DataFrame()
monitoring_data = {
    Config.PREDICTION_COUNT_KEY: 0,
    Config.PREDICTION_DISTRIBUTION_KEY: Counter(),
    Config.PREDICTION_LATENCY_KEY: [] # Store latencies to calculate average
}
MODEL_FEATURE_ORDER = [] # Initialize as an empty list

# Define your transaction features here based on your dataset
class Transaction(BaseModel):
    # Field(alias="Unnamed: int64") can be used if you absolutely need to keep original name
    Timestamp: str
    From_Bank: int
    Account: str
    To_Bank: int
    Account_To: str = Field(alias="Account.1") # Renamed from Account.1 for valid Python variable name
    Amount_Received: float
    Receiving_Currency: str
    Amount_Paid: float
    Payment_Currency: str
    Payment_Format: str
    # Is_Laundering: float # Removed: Assuming this is the target variable, not an input feature.



# Helper to load model from GCS
def load_model_from_gcs():
    """Loads the machine learning model from Google Cloud Storage."""
    global model
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_MODEL_BUCKET)
    blob = bucket.blob(Config.GCS_MODEL_FILE)

    try:
        model_bytes = blob.download_as_bytes()
        model = joblib.load(io.BytesIO(model_bytes))
        logger.info(f"Successfully loaded model from gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error loading model from GCS: {e}")
        raise

# Helper to load feature columns for model
def load_feature_columns_for_model_from_gcs():
    """Loads the list of feature columns from a JSON file in Google Cloud Storage."""
    global MODEL_FEATURE_ORDER
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_MODEL_BUCKET) # Assuming feature file is in the model bucket
    blob = bucket.blob(Config.GCS_FEATURE_FILE)

    try:
        feature_bytes = blob.download_as_bytes()
        # Decode bytes to string and then parse JSON
        MODEL_FEATURE_ORDER = json.loads(feature_bytes.decode('utf-8'))
        logger.info(f"Successfully loaded feature columns from gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_FEATURE_FILE}")
        logger.info(f"Loaded feature order: {MODEL_FEATURE_ORDER}")
    except Exception as e:
        logger.error(f"Error loading feature columns from GCS: {e}")
        raise

# Helper to load production dataset
def load_production_data_from_gcs():
    """Loads data from Google Cloud Storage."""
    global production_dataset
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_DATA_BUCKET)
    blob = bucket.blob(Config.GCS_PRODUCTION_DATA_FILE)

    try:
        production_data_bytes = blob.download_as_bytes()
        production_dataset = pd.read_csv(io.BytesIO(production_data_bytes))
        logger.info(f"Successfully loaded production data from gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}")
    except Exception as e:
        logger.error(f"Error loading production data from GCS: {e}")
        raise

# --- Application Startup Event ---
# This function runs once when the FastAPI application starts.
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    load_model_from_gcs()
    load_feature_columns_for_model_from_gcs()
    load_production_data_from_gcs() # Load production data when app starts

# --- Health Check Endpoint (Recommended for Cloud Deployments) ---
# This endpoint can be used by deployment platforms (like Cloud Run) to check
# if the application is healthy and ready to receive requests.
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None, "production_data_loaded": not production_dataset.empty}

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Convert Pydantic model to a Python dictionary, then to a Pandas DataFrame row.
        # Use .model_dump(by_alias=True) to handle field aliases (like "Account.1")
        transaction_dict = transaction.model_dump(by_alias=True)
        # Wrap the dict in a list to create a DataFrame with a single row
        input_df = pd.DataFrame([transaction_dict])

        logger.info(f"Received input DataFrame for prediction: \n{input_df.to_string()}")

        # Apply the same feature engineering logic used during training.
        # This function will handle categorical encoding and feature alignment.
        processed_feature_df = apply_feature_engineering(input_df)
        
        logger.info(f"Processed DataFrame for model prediction: \n{processed_feature_df.to_string()}")
        logger.info(f"Processed DataFrame columns: {processed_feature_df.columns.tolist()}")

        # Validate that the processed features match the expected shape for the model
        if processed_feature_df.shape[1] != len(MODEL_FEATURE_ORDER):
            error_detail = (f"Feature count mismatch after processing. Expected {len(MODEL_FEATURE_ORDER)} features, "
                            f"but got {processed_feature_df.shape[1]}. "
                            f"Expected: {MODEL_FEATURE_ORDER}, Got: {processed_feature_df.columns.tolist()}")
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)

        # Ensure the column order is exactly as expected by the model.
        # This is handled by `apply_feature_engineering` by selecting `MODEL_FEATURE_ORDER`.
        processed_feature_df = processed_feature_df[MODEL_FEATURE_ORDER] # Redundant if apply_feature_engineering does it

        # Perform prediction
        start_time = time.perf_counter()
        
        # `predict_proba` returns probabilities for each class (e.g., [prob_not_fraud, prob_fraud])
        # We assume binary classification here. `[0]` takes the first (and only) prediction.
        prediction_proba = model.predict_proba(processed_feature_df)[0].tolist() 
        
        # `predict` returns the predicted class (e.g., 0 for not fraud, 1 for fraud)
        prediction_class = int(model.predict(processed_feature_df)[0])
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000 # Convert to milliseconds

        logger.info(f"Prediction successful: Class={prediction_class}, Probabilities={prediction_proba}, Latency={latency:.2f} ms")

        return {
            "prediction_class": prediction_class,
            "prediction_probabilities": prediction_proba,
            "latency_ms": latency
        }

    except KeyError as e:
        logger.error(f"Missing expected feature in the input transaction or during processing: {e}")
        raise HTTPException(status_code=400, detail=f"Missing expected data for feature: {e}. Please ensure all required transaction fields are provided.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during prediction. Details: {e}")


@app.get("/get-next-transaction")
async def get_next_transaction():
    """Returns the next row from the production dataset for frontend display."""
    global production_dataset

    if production_dataset.empty:
        raise HTTPException(status_code=404, detail="No more transactions in the production dataset.")

    # Get the first row, convert to dictionary, then remove it from the dataset
    next_row = production_dataset.iloc[0].to_dict()
    production_dataset = production_dataset.iloc[1:].reset_index(drop=True)
    
    return {"transaction": next_row}

@app.get("/monitoring")
async def get_monitoring_data():
    """Returns current model monitoring metrics."""
    # Calculate average latency
    avg_latency = sum(monitoring_data[Config.PREDICTION_LATENCY_KEY]) / len(monitoring_data[Config.PREDICTION_LATENCY_KEY]) \
                  if monitoring_data[Config.PREDICTION_LATENCY_KEY] else 0

    return {
        "total_predictions": monitoring_data[Config.PREDICTION_COUNT_KEY],
        "prediction_distribution": dict(monitoring_data[Config.PREDICTION_DISTRIBUTION_KEY]),
        "average_latency_ms": avg_latency,
        "current_production_data_remaining": len(production_dataset)
    }