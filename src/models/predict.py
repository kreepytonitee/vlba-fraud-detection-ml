import pandas as pd
import mlflow.pyfunc
import json
import os

# Global variables for model and feature columns
_model = None
_feature_columns = None
_mlflow_model_uri = os.environ.get("MLFLOW_MODEL_URI", "models:/XGBoostFraudDetector/latest")
_feature_columns_path = "feature_columns.json" # Assumed to be available in the container

def load_model_and_features():
    """Loads the MLflow model and feature columns into global variables."""
    global _model, _feature_columns
    if _model is None:
        _model = mlflow.pyfunc.load_model(_mlflow_model_uri)
        print(f"Model loaded from: {_mlflow_model_uri}")
    if _feature_columns is None:
        # In a deployed setting, feature_columns.json should be part of the artifact
    # or retrieved from a known location. For local Flask/FastAPI, ensure it's copied.
        if not os.path.exists(_feature_columns_path):
            # Fallback for local testing, if not copied, assumes a basic set
            print(f"Warning: {_feature_columns_path} not found. Using a dummy feature set for testing.")
            _feature_columns = ['amount_sent', 'amount_received', 'amount_diff', 'amount_ratio',
                                'hour_of_day', 'day_of_week', 'month', 'is_weekend']
        else:
            with open(_feature_columns_path, "r") as f:
                _feature_columns = json.load(f)
        print(f"Feature columns loaded: {_feature_columns}")

def preprocess_and_predict(input_data: dict) -> float:
    """
    Preprocesses incoming single transaction data and makes a prediction.
    Assumes input_data keys match original CSV columns.
    """
    if _model is None or _feature_columns is None:
        load_model_and_features()

    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Ensure 'date_time' is datetime object
    if 'date_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'])

    # Apply the same feature engineering as during training
    from src.features.feature_engineering import apply_feature_engineering
    processed_df = apply_feature_engineering(df.copy())

    # Handle categorical columns (one-hot encode as done in training)
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        processed_df = pd.get_dummies(processed_df, columns=list(categorical_cols), drop_first=True)

    # Select and reindex columns to match the model's expected feature set
    # Fill missing columns (e.g., if a new bank_pair appears) with 0
    final_features_df = processed_df.reindex(columns=_feature_columns, fill_value=0)

    # Make prediction
    prediction_proba = _model.predict_proba(final_features_df)[:, 1][0]
    return float(prediction_proba)

if __name__ == "__main__":
    # Example usage for local testing
    load_model_and_features() # Ensure model and features are loaded

    sample_transaction = {
        'amount_sent': 120, 'amount_received': 115, 'date_time': '2025-06-19 14:30:00',
        'currency_sent': 'USD', 'currency_received': 'USD', 'bank_sender': 'BankA', 'bank_receiver': 'BankX'
    }

    # Assuming you have a trained model available via MLFLOW_MODEL_URI
    # For a truly local test, you might need to mock MLflow or ensure the model is downloaded.
    # For this example, if the model isn't found, it will print an error.

    try:
        fraud_probability = preprocess_and_predict(sample_transaction)
        print(f"Fraud probability for sample transaction: {fraud_probability:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Please ensure MLFLOW_MODEL_URI environment variable is set or model is available.")