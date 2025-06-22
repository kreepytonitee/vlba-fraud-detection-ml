from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import os
import pandas as pd # Make sure pandas is imported if you use it for model input

from contextlib import asynccontextmanager # Import the new helper

# Define your input schema
class TransactionFeatures(BaseModel):
    feature1: float
    feature2: float
    # ... add all your features your model expects

# Global variable to hold the loaded model
# This will be loaded once when the application starts
model = None

# MLflow model URI can be passed via environment variable for flexibility
# Use a default that makes sense for your setup during local development,
# but ensure it's overridden by your deployment environment (e.g., Cloud Run)
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "runs:/<YOUR_LAST_MLFLOW_RUN_ID>/xgboost_model")
# OR for a registered model:
# MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/XGBoostFraudDetector/Production")


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Loads the ML model when the application starts.
    """
    global model
    print(f"[{__name__}] Startup: Loading model from MLflow URI: {MLFLOW_MODEL_URI}")
    try:
        # MLflow's load_model can be synchronous, but it's good practice
        # to use await if any part of the model loading becomes async.
        # For typical MLflow model loading, direct call is fine.
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        print(f"[{__name__}] Startup: Model loaded successfully!")
    except Exception as e:
        print(f"[{__name__}] Startup: Error loading model: {e}")
        # Depending on criticality, you might want to exit here or log more severely
        # For production, an app failing to load its core component should usually not start.
        raise RuntimeError(f"Failed to load ML model at startup: {e}") from e

    yield # The application starts serving requests here

    # This code runs on shutdown
    print(f"[{__name__}] Shutdown: Application shutting down.")
    # You could add cleanup logic here if needed, e.g., closing database connections
# --- End Lifespan Context Manager ---


# Pass the lifespan context manager to the FastAPI application
app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "error", "model_loaded": False, "message": "Model not loaded. Check startup logs."}

@app.post("/predict")
async def predict_fraud(transaction: TransactionFeatures):
    """Endpoint for predicting fraud."""
    if model is None:
        # This error indicates a critical failure during startup
        return {"error": "Model not loaded. Please check server logs.", "status": "error"}, 500

    # Convert input features to a format the model expects (e.g., pandas DataFrame)
    # Ensure the order and names of features match what the model was trained on
    # You might need to add more robust feature ordering/validation here.
    features_dict = transaction.dict()
    features_df = pd.DataFrame([features_dict])

    # Make prediction
    try:
        # Assuming your MLflow pyfunc model has a predict_proba method for binary classification
        prediction_proba = model.predict_proba(features_df)[:, 1][0]
        prediction = (prediction_proba > 0.5).astype(int) # Example threshold
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed due to model error.", "details": str(e), "status": "error"}, 500


    return {"is_fraud_prediction": prediction, "fraud_probability": prediction_proba, "status": "success"}

# This block is for local development/testing only
if __name__ == "__main__":
    import uvicorn
    print(f"[{__name__}] Running FastAPI app directly for local development...")
    # For local testing, you might need to ensure MLFLOW_MODEL_URI points to a local path
    # or a local MLflow server with the model registered.
    # Example for local run artifact:
    # os.environ["MLFLOW_MODEL_URI"] = "mlruns/0/abcdef1234567890abcdef1234567890/artifacts/xgboost_model"
    # Example for a local registered model:
    # os.environ["MLFLOW_MODEL_URI"] = "models:/XGBoostFraudDetector/Production"
    # Ensure you have the necessary MLflow artifacts and folders accessible if using local paths.

    uvicorn.run(app, host="0.0.0.0", port=8000)