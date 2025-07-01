from google.cloud import storage
import pandas as pd
import joblib
# import json
import io
# import os
from utils.config import Config
from utils.logger import logger

# def upload_dataframe_to_gcs(dataframe, bucket_name, destination_blob_name):
#     """Uploads a pandas DataFrame to GCS as a CSV file."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)

#     csv_buffer = io.StringIO()
#     dataframe.to_csv(csv_buffer, index=False)
#     blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
#     print(f"DataFrame uploaded to gs://{bucket_name}/{destination_blob_name}")

# def download_dataframe_from_gcs(bucket_name, source_blob_name):
#     """Downloads a CSV file from GCS and returns it as a pandas DataFrame."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     # Download to a string in memory
#     csv_string = blob.download_as_text()
#     dataframe = pd.read_csv(io.StringIO(csv_string))
#     print(f"DataFrame downloaded from gs://{bucket_name}/{source_blob_name}")
#     return dataframe

# def upload_model_to_gcs(model, bucket_name, destination_blob_name):
#     """Uploads a joblib-serialized model to GCS."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)

#     # Serialize model to bytes in memory
#     # model_bytes = joblib.dump(model)
#     blob.upload_from_string(model, content_type='application/octet-stream')
#     print(f"Model uploaded to gs://{bucket_name}/{destination_blob_name}")

# def download_model_from_gcs(bucket_name, source_blob_name):
#     """Downloads a joblib-serialized model from GCS and deserializes it."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     # Download model bytes to memory
#     model_bytes = blob.download_as_bytes()
#     model = joblib.load(model_bytes)
#     print(f"Model downloaded from gs://{bucket_name}/{source_blob_name}")
#     return model

# def upload_text_to_gcs(text_content, bucket_name, destination_blob_name):
#     """Uploads a string (e.g., feature list) to GCS as a text file."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_string(text_content, content_type='text/plain')
#     print(f"Text content uploaded to gs://{bucket_name}/{destination_blob_name}")

# def download_text_from_gcs(bucket_name, source_blob_name):
#     """Downloads text content from GCS."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     text_content = blob.download_as_text()
#     print(f"Text content downloaded from gs://{bucket_name}/{source_blob_name}")
#     return text_content

# def upload_series_to_gcs(series, bucket_name, destination_blob_name):
#     """Uploads a pandas Series to GCS as a CSV file."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)

#     csv_buffer = io.StringIO()
#     series.to_csv(csv_buffer, header=False) # Series to_csv doesn't need index=False, but header=False is good for lookup
#     blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
#     print(f"Series uploaded to gs://{bucket_name}/{destination_blob_name}")

# def download_series_from_gcs(bucket_name, source_blob_name):
#     """Downloads a CSV file from GCS and returns it as a pandas Series."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     csv_string = blob.download_as_text()
#     # Read as DataFrame first then squeeze to Series, handling potential header from training
#     series = pd.read_csv(io.StringIO(csv_string), index_col=0, header=None).squeeze("columns")
#     print(f"Series downloaded from gs://{bucket_name}/{source_blob_name}")
#     return series

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
    bucket = client.get_bucket(Config.GCS_MODEL_BUCKET)
    blob = bucket.blob(Config.GCS_FEATURE_COLUMNS_FILE)

    try:
        feature_bytes = blob.download_as_bytes()
        # Decode bytes to string and then parse file
        MODEL_FEATURE_ORDER = feature_bytes.decode('utf-8').splitlines()
        logger.info(f"Successfully loaded feature columns from gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_FEATURE_COLUMNS_FILE}")
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

# Helper to check if a file exists
def check_gcs_blob_exists(bucket_name, blob_name):
    """Checks if a blob exists in a GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()

