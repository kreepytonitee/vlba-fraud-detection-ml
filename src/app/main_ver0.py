from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import os

# Import the prediction logic
from src.models.predict import preprocess_and_predict, load_model_and_features

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent banking transactions.",
    version="1.0.0"
)

# Pydantic model for incoming transaction data
class Transaction(BaseModel):
    amount_sent: float
    amount_received: float
    date_time: datetime
    currency_sent: str
    currency_received: str
    bank_sender: str
    bank_receiver: str

@app.on_event("startup")
async def startup_event():
    """Load model and features when the application starts."""
    try:
        load_model_and_features()
        print("Model and features loaded successfully on startup.")
    except Exception as e:
        print(f"Error loading model or features on startup: {e}")
        # Optionally, re-raise to prevent startup if model is critical
        # raise RuntimeError("Failed to load ML model on startup.")

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """
    Receives a transaction and returns the probability of it being fraudulent.
    """
    try:
        # Convert Pydantic model to a dictionary suitable for preprocessing
        input_data_dict = transaction.model_dump()

        fraud_probability = preprocess_and_predict(input_data_dict)

        return {"fraud_probability": fraud_probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True if _model is not None else False}

if __name__ == "__main__":
    # For local testing, ensure MLFLOW_MODEL_URI is set, e.g.:
    # export MLFLOW_MODEL_URI="runs:/<YOUR_MLFLOW_RUN_ID>/fraud_detection_model"
    # or "models:/XGBoostFraudDetector/Production"

    # If running this directly without Docker, you might need to ensure
    # the feature_columns.json is accessible or provide a dummy.

    uvicorn.run(app, host="0.0.0.0", port=8000)