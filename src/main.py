from fastapi import FastAPI, HTTPException, Request
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
original_production_dataset: pd.DataFrame = pd.DataFrame()  # Keep original for restart
monitoring_data = {
    Config.PREDICTION_COUNT_KEY: 0,
    Config.PREDICTION_DISTRIBUTION_KEY: Counter(),
    Config.PREDICTION_LATENCY_KEY: [] # Store latencies to calculate average
}

# Server monitoring data
server_monitoring_data = {
    "total_requests": 0,
    "error_count": 0,
    "request_times": [],
    "last_request_duration": 0.0,
    "start_time": time.time()
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


# Middleware to track server metrics
@app.middleware("http")
async def track_requests(request: Request, call_next):
    global server_monitoring_data
    
    start_time = time.time()
    server_monitoring_data["total_requests"] += 1
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        server_monitoring_data["error_count"] += 1
        raise e
    finally:
        process_time = time.time() - start_time
        server_monitoring_data["request_times"].append(process_time)
        server_monitoring_data["last_request_duration"] = process_time
        
        # Keep only last 1000 request times to prevent memory issues
        if len(server_monitoring_data["request_times"]) > 1000:
            server_monitoring_data["request_times"] = server_monitoring_data["request_times"][-1000:]


# --- Application Startup Event ---
# This function runs once when the FastAPI application starts.
@app.on_event("startup")
async def startup_event():
    # DECLARE GLOBAL VARIABLES HERE SO YOU CAN MODIFY THEM
    global model, production_dataset, original_production_dataset, MODEL_FEATURE_ORDER, server_monitoring_data
    logger.info("Starting up application...")

    try:
        # Load model with retry logic for Cloud Run deployment
        model = load_model_from_gcs()
        if model is None:
            logger.error("Failed to load model during startup")
            # In production, you might want to fail fast or load a fallback model
            
        MODEL_FEATURE_ORDER = load_feature_columns_for_model_from_gcs()
        if not MODEL_FEATURE_ORDER:
            logger.error("Failed to load feature columns during startup")
            
        production_dataset = load_production_data_from_gcs()
        original_production_dataset = production_dataset.copy()
        
        # Initialize server monitoring start time
        server_monitoring_data["start_time"] = time.time()
        
        logger.info(f"Startup completed successfully. Model loaded: {model is not None}, "
                   f"Feature columns: {len(MODEL_FEATURE_ORDER)}, "
                   f"Production data rows: {len(production_dataset)}")
                   
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # In production, you might want to exit here
        # raise e

# --- Health Check Endpoint (Recommended for Cloud Deployments) ---
# This endpoint can be used by deployment platforms (like Cloud Run) to check
# if the application is healthy and ready to receive requests.
@app.get("/health")
async def health_check():
    """Enhanced health check for Cloud Run deployment"""
    health_status = {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "production_data_loaded": not production_dataset.empty,
        "feature_columns_loaded": len(MODEL_FEATURE_ORDER) > 0,
        "uptime_seconds": time.time() - server_monitoring_data["start_time"]
    }
    
    # Return 503 if critical components failed to load
    if not health_status["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return health_status

# --- Endpoint to check deployment info ---
@app.get("/deployment-info")
async def get_deployment_info():
    """Get information about the current deployment"""
    return {
        "service_name": "fraud-detection-api",
        "environment": "cloud-run" if os.getenv('K_SERVICE') else "local",
        "build_id": os.getenv('BUILD_ID', 'unknown'),
        "revision": os.getenv('K_REVISION', 'unknown'),
        "service": os.getenv('K_SERVICE', 'unknown'),
        "region": os.getenv('GOOGLE_CLOUD_REGION', 'unknown'),
        "project": os.getenv('GOOGLE_CLOUD_PROJECT', 'unknown')
    }

# --- Reset Dataset Endpoint ---
@app.post("/reset-dataset")
async def reset_dataset():
    """Reset the production dataset to its original state and clear monitoring data."""
    global production_dataset, monitoring_data, server_monitoring_data
    
    try:
        production_dataset = original_production_dataset.copy()
        
        # Reset monitoring data
        monitoring_data = {
            Config.PREDICTION_COUNT_KEY: 0,
            Config.PREDICTION_DISTRIBUTION_KEY: Counter(),
            Config.PREDICTION_LATENCY_KEY: []
        }
        
        # Reset server monitoring (except total requests and errors which are cumulative)
        server_monitoring_data["request_times"] = []
        server_monitoring_data["last_request_duration"] = 0.0
        
        logger.info("Dataset and monitoring data reset successfully")
        return {"message": "Dataset reset successfully", "total_transactions": len(production_dataset)}
    except Exception as e:
        logger.error(f"Error resetting dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting dataset: {e}")

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(transaction: Transaction):
    global monitoring_data
    
    input_temp_filepath = None
    output_temp_filepath = None
    outputlabel_temp_filepath = None
    
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

        # 2. Create a temporary file path for the output of generate_production_features
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_output.csv', encoding='utf-8') as tmp_output_file:
            output_temp_filepath = tmp_output_file.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_output.csv', encoding='utf-8') as tmp_output_file:
            outputlabel_temp_filepath = tmp_output_file.name

        # 3. Call generate_production_features, passing both temp file paths
        generate_production_features(
            input_path=input_temp_filepath,
            output_path_data=output_temp_filepath,
            output_path_label=outputlabel_temp_filepath,
            lookup_dir=f"gs://{Config.GCS_LOOKUPS_BUCKET}/"
        )

        # 4. Read the processed DataFrame from the output temporary file
        processed_feature_df = pd.read_csv(output_temp_filepath)
        true_label_df = pd.read_csv(outputlabel_temp_filepath)
        
        logger.info(f"Processed DataFrame for model prediction: \n{processed_feature_df.to_string()}")

        for col in MODEL_FEATURE_ORDER:  # Fill encoded columns
            if col not in processed_feature_df.columns:
                processed_feature_df[col] = 0
        processed_feature_df = processed_feature_df[MODEL_FEATURE_ORDER]

        # Validate that the processed features match the expected shape for the model
        if processed_feature_df.shape[1] != len(MODEL_FEATURE_ORDER):
            error_detail = (f"Feature count mismatch after processing. Expected {len(MODEL_FEATURE_ORDER)} features, "
                            f"but got {processed_feature_df.shape[1]}. "
                            f"Expected: {MODEL_FEATURE_ORDER}, Got: {processed_feature_df.columns.tolist()}")
            logger.error(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)

        # Perform prediction
        start_time = time.perf_counter()
        
        prediction_proba = model.predict_proba(processed_feature_df)[0].tolist() 
        prediction_class = int(model.predict(processed_feature_df)[0])
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000 # Convert to milliseconds

        # Update monitoring data
        monitoring_data[Config.PREDICTION_COUNT_KEY] += 1
        monitoring_data[Config.PREDICTION_DISTRIBUTION_KEY][str(prediction_class)] += 1
        monitoring_data[Config.PREDICTION_LATENCY_KEY].append(latency)
        
        # Keep only last 1000 latencies to prevent memory issues
        if len(monitoring_data[Config.PREDICTION_LATENCY_KEY]) > 1000:
            monitoring_data[Config.PREDICTION_LATENCY_KEY] = monitoring_data[Config.PREDICTION_LATENCY_KEY][-1000:]

        logger.info(f"Prediction successful: Class={prediction_class}, Probabilities={prediction_proba}, Latency={latency:.2f} ms")

        # Extract true label value
        true_label = None
        if not true_label_df.empty and 'Is Laundering' in true_label_df.columns:
            true_label = int(true_label_df['Is Laundering'].iloc[0])

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
        # Cleanup temporary files
        for temp_file in [input_temp_filepath, output_temp_filepath, outputlabel_temp_filepath]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {temp_file}: {e}")

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
    """Returns current model and server monitoring metrics."""
    global monitoring_data, server_monitoring_data
    
    # Calculate average latency
    avg_latency = sum(monitoring_data[Config.PREDICTION_LATENCY_KEY]) / len(monitoring_data[Config.PREDICTION_LATENCY_KEY]) \
                  if monitoring_data[Config.PREDICTION_LATENCY_KEY] else 0

    # Calculate server metrics
    avg_response_time = sum(server_monitoring_data["request_times"]) / len(server_monitoring_data["request_times"]) \
                       if server_monitoring_data["request_times"] else 0
    
    error_rate = (server_monitoring_data["error_count"] / server_monitoring_data["total_requests"]) * 100 \
                 if server_monitoring_data["total_requests"] > 0 else 0
    
    uptime = time.time() - server_monitoring_data["start_time"]

    return {
        # Model monitoring
        "total_predictions": monitoring_data[Config.PREDICTION_COUNT_KEY],
        "prediction_distribution": dict(monitoring_data[Config.PREDICTION_DISTRIBUTION_KEY]),
        "average_latency_ms": avg_latency,
        "current_production_data_remaining": len(production_dataset),
        
        # Server monitoring
        "total_requests": server_monitoring_data["total_requests"],
        "avg_response_time_ms": avg_response_time * 1000,  # Convert to ms
        "last_request_duration_ms": server_monitoring_data["last_request_duration"] * 1000,  # Convert to ms
        "error_count": server_monitoring_data["error_count"],
        "error_rate_percent": error_rate,
        "uptime_seconds": uptime
    }