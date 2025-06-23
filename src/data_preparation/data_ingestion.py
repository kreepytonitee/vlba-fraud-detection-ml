import pandas as pd
from google.cloud import storage
import io
from src.utils.config import Config
from src.utils.logger import logger

def load_data_from_gcs():
    """Loads data from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(Config.GCS_DATA_BUCKET)
    blob = bucket.blob(Config.GCS_DATA_FILE)

    try:
        data_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data_bytes))
        logger.info(f"Successfully loaded data from gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_DATA_FILE}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from GCS: {e}")
        raise