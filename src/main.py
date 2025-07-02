from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import os
import time
import io
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from google.cloud import storage
from collections import Counter
from src.utils.config import Config
from src.utils.logger import logger
from src.utils.gcs_utils import load_model_from_gcs, load_feature_columns_for_model_from_gcs, load_production_data_from_gcs
from src.feature_engineering.production_features import generate_production_features # Import the function

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent bank transactions."
)

# --- CORS Configuration ---
# IMPORTANT: Adjust 'allow_origins' in production for security.
# For development/showcase, '*' is often used, but specify your frontend URL
# (e.g., "https://your-website-bucket-name.storage.googleapis.com") for production.
origins = [
    "http://localhost:3000", # For Create React App
    "http://localhost:5173", # For Vite
    "http://localhost:8000",
    "https://storage.googleapis.com",
    "https://storage.cloud.google.com" # Allows all origins for development/testing.
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
    transaction_id: int = Field(alias="Unnamed: 0")
    timestamp: str = Field(alias="Timestamp") 
    from_bank: int = Field(alias="From Bank")
    account: str = Field(alias="Account")
    to_bank: int = Field(alias="To Bank")
    account_to: str = Field(alias="Account.1")
    amount_received: float = Field(alias="Amount Received")
    receiving_currency: str = Field(alias="Receiving Currency")
    amount_paid: float = Field(alias="Amount Paid")
    payment_currency: str = Field(alias="Payment Currency")
    payment_format: str = Field(alias="Payment Format")


# --- Application Startup Event ---
# This function runs once when the FastAPI application starts.
@app.on_event("startup")
async def startup_event():
    # DECLARE GLOBAL VARIABLES HERE SO YOU CAN MODIFY THEM
    global model, production_dataset, MODEL_FEATURE_ORDER
    logger.info("Starting up application...")

    # ASSIGN THE RETURN VALUES TO THE GLOBAL VARIABLES
    model = load_model_from_gcs()
    MODEL_FEATURE_ORDER = load_feature_columns_for_model_from_gcs()
    production_dataset = load_production_data_from_gcs() # Load production data when app starts

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

        # 1. Create a temporary file for input to generate_production_features
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_input.csv', encoding='utf-8') as tmp_input_file:
            input_temp_filepath = tmp_input_file.name
            input_df.to_csv(tmp_input_file, index=False)
            tmp_input_file.flush()
            # os.fsync(tmp_input_file.fileno())

        # 2. Create a temporary file path for the output of generate_production_features
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_output.csv', encoding='utf-8') as tmp_output_file:
            output_temp_filepath = tmp_output_file.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_output.csv', encoding='utf-8') as tmp_output_file:
            outputlabel_temp_filepath = tmp_output_file.name
            # Don't write anything to it yet, just get the path

        # 3. Call generate_production_features, passing both temp file paths
        # It will read from input_temp_filepath and save to output_temp_filepath and outputlabel_temp_filepath
        generate_production_features(
            input_path=input_temp_filepath,
            output_path_data=output_temp_filepath,
            output_path_label=outputlabel_temp_filepath,
            lookup_dir=f"gs://{Config.GCS_LOOKUPS_BUCKET}/"
        )

        # 4. Read the processed DataFrame from the output temporary file
        processed_feature_df = pd.read_csv(output_temp_filepath) # <--- Read from the output temp file
        true_label = pd.read_csv(outputlabel_temp_filepath)
        
        logger.info(f"Processed DataFrame for model prediction: \n{processed_feature_df.to_string()}")
        # logger.info(f"Processed DataFrame columns: {processed_feature_df.columns.tolist()}")

        for col in MODEL_FEATURE_ORDER:  # Fill encoded columns
            if col not in processed_feature_df.columns:
                processed_feature_df[col] = 0
        processed_feature_df = processed_feature_df[MODEL_FEATURE_ORDER] # Ensure the column order is exactly as expected by the model.

        # Validate that the processed features match the expected shape for the model
        if processed_feature_df.shape[1] != len(MODEL_FEATURE_ORDER):
            error_detail = (f"Feature count mismatch after processing. Expected {len(MODEL_FEATURE_ORDER)} features, "
                            f"but got {processed_feature_df.shape[1]}. "
                            f"Expected: {MODEL_FEATURE_ORDER}, Got: {processed_feature_df.columns.tolist()}")
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)

        # Perform prediction
        start_time = time.perf_counter()
        
        # print("Model expects:", MODEL_FEATURE_ORDER)
        # print("Data columns:", processed_feature_df.columns.tolist())
        # print("Columns not matching:", set(processed_feature_df.columns) - set(MODEL_FEATURE_ORDER))
        # print("Extra columns:", set(processed_feature_df.columns) - set(MODEL_FEATURE_ORDER))
        # print("Missing columns:", set(MODEL_FEATURE_ORDER) - set(processed_feature_df.columns))

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
            "latency_ms": latency,
            "true_label": true_label
        }

    except KeyError as e:
        logger.error(f"Missing expected feature in the input transaction or during processing: {e}")
        raise HTTPException(status_code=400, detail=f"Missing expected data for feature: {e}. Please ensure all required transaction fields are provided.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during prediction. Details: {e}")
    
    finally:
        # --- This cleanup logic is CRUCIAL for Cloud Run ---
        if input_temp_filepath and os.path.exists(input_temp_filepath):
            try:
                os.remove(input_temp_filepath)
                logger.info(f"Cleaned up temporary input file: {input_temp_filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up input temp file {input_temp_filepath}: {e}")
        if output_temp_filepath and os.path.exists(output_temp_filepath):
            try:
                os.remove(output_temp_filepath)
                logger.info(f"Cleaned up temporary output file: {output_temp_filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up output temp file {output_temp_filepath}: {e}")
        if outputlabel_temp_filepath and os.path.exists(outputlabel_temp_filepath):
            try:
                os.remove(outputlabel_temp_filepath)
                logger.info(f"Cleaned up temporary output file: {outputlabel_temp_filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up output temp file {outputlabel_temp_filepath}: {e}")

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
    global monitoring_data
    # Calculate average latency
    avg_latency = sum(monitoring_data[Config.PREDICTION_LATENCY_KEY]) / len(monitoring_data[Config.PREDICTION_LATENCY_KEY]) \
                  if monitoring_data[Config.PREDICTION_LATENCY_KEY] else 0

    return {
        "total_predictions": monitoring_data[Config.PREDICTION_COUNT_KEY],
        "prediction_distribution": dict(monitoring_data[Config.PREDICTION_DISTRIBUTION_KEY]),
        "average_latency_ms": avg_latency,
        "current_production_data_remaining": len(production_dataset)
    }