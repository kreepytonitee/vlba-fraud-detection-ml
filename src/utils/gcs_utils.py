from google.cloud import storage
import pandas as pd
import joblib
import json
import io
# import os
from src.utils.config import Config
from src.utils.logger import logger

# Helper to load model from GCS
def load_model_from_gcs():
    """Loads the machine learning model from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_MODEL_BUCKET)
    blob = bucket.blob(Config.GCS_MODEL_FILE)

    try:
        model_bytes = blob.download_as_bytes()
        model = joblib.load(io.BytesIO(model_bytes))
        logger.info(f"Successfully loaded model from gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_MODEL_FILE}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from GCS: {e}")
        raise

# Helper to load feature columns for model
def load_feature_columns_for_model_from_gcs():
    """Loads the list of feature columns from a JSON file in Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_MODEL_BUCKET)
    blob = bucket.blob(Config.GCS_FEATURE_COLUMNS_FILE)

    try:
        feature_bytes = blob.download_as_bytes()
        # Decode bytes to string and then parse file
        MODEL_FEATURE_ORDER = feature_bytes.decode('utf-8').splitlines()
        logger.info(f"Successfully loaded feature columns from gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_FEATURE_COLUMNS_FILE}")
        logger.info(f"Loaded feature order: {MODEL_FEATURE_ORDER}")
        return MODEL_FEATURE_ORDER
    except Exception as e:
        logger.error(f"Error loading feature columns from GCS: {e}")
        raise

# Helper to load production dataset
def load_production_data_from_gcs():
    """Loads data from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_DATA_BUCKET)
    blob = bucket.blob(Config.GCS_PRODUCTION_DATA_FILE)

    try:
        production_data_bytes = blob.download_as_bytes()
        production_dataset = pd.read_csv(io.BytesIO(production_data_bytes))
        logger.info(f"Successfully loaded production data from gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}")
        return production_dataset
    except Exception as e:
        logger.error(f"Error loading production data from GCS: {e}")
        raise

# Helper to check if a file exists
def check_gcs_blob_exists(bucket_name, blob_name):
    """Checks if a blob exists in a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

# Helper to write json file
def upload_json_to_gcs(data: Union[Dict[str, Any], list], bucket_name: str, blob_name: str) -> bool:
    """
    Upload JSON data to Google Cloud Storage
    
    Args:
        data: Dictionary or list to be converted to JSON
        bucket_name: Name of the GCS bucket
        blob_name: Path/name of the blob in the bucket
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Convert data to JSON string
        json_string = json.dumps(data, indent=2, default=str)
        
        # Upload to GCS
        blob.upload_from_string(
            json_string,
            content_type='application/json'
        )
        
        logger.info(f"Successfully uploaded JSON to gs://{bucket_name}/{blob_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload JSON to gs://{bucket_name}/{blob_name}: {e}")
        return False
