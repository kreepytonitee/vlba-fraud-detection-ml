import os
import logging
from utils.logger import logger
from utils.config import Config
from utils.gcs_utils import check_gcs_blob_exists

def run_training_pipeline():
    """
    Run the ML training pipeline using your existing functions
    This is adapted from your current cloud_orchestrator.py
    """
    try:
        logger.info("--- Starting Cloud Orchestrated ML Training Pipeline ---")
        
        # Step 1: Data Preprocessing
        logger.info("\n--- Stage 1: Data Preprocessing ---")
        if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_RAW_DATA_FILE):
            logger.error(f"Raw data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_RAW_DATA_FILE}) not found.")
            logger.info("Please upload your 'transactions.csv' to this GCS location manually and try again.")
            return False

        from data_preprocessing.data_clean_split import clean_and_split_data
        clean_and_split_data(
            input_csv_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_RAW_DATA_FILE}",
            train_output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_TRAIN_DATA_FILE}",
            prod_output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}",
            target_col='Is Laundering',
            test_size=0.25,
            random_state=42
        )
        logger.info("Data preprocessing completed and results uploaded to GCS.")

        # Step 2: Feature Engineering (Training Data)
        logger.info("\n--- Stage 2: Feature Engineering (Training Data) ---")
        if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_TRAIN_DATA_FILE):
            logger.error(f"Training data not found.")
            return False

        from feature_engineering.train_features import generate_training_features
        generate_training_features(
            input_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_TRAIN_DATA_FILE}",
            output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE}",
            feature_cols_output_path=f"gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_FEATURE_COLUMNS_FILE}",
            lookup_dir=f"gs://{Config.GCS_LOOKUPS_BUCKET}/"
        )
        logger.info("Training feature engineering completed and artifacts uploaded to GCS.")

        # Step 3: Model Training
        logger.info("\n--- Stage 3: Model Training ---")
        if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE):
            logger.error(f"Feature engineered training data not found.")
            return False

        from models.train_model import train_and_save_model
        train_and_save_model(
            input_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE}",
            model_output_dir=f"gs://{Config.GCS_MODEL_BUCKET}/"
        )
        logger.info("Model training completed and model uploaded to GCS.")

        # Save training pipeline metadata
        pipeline_info = {
            "status": "success",
            "mode": "training",
            "build_id": os.getenv('BUILD_ID', 'local'),
            "model_version": os.getenv('MODEL_VERSION', 'latest'),
            "timestamp": os.getenv('BUILD_TIMESTAMP', 'unknown')
        }
        
        from utils.gcs_utils import upload_json_to_gcs
        upload_json_to_gcs(
            pipeline_info, 
            Config.GCS_MODEL_BUCKET, 
            f"pipeline_runs/training_run_{os.getenv('BUILD_ID', 'local')}.json"
        )

        logger.info("\n--- Cloud Orchestrated ML Training Pipeline Completed Successfully ---")
        return True

    except Exception as e:
        logger.error(f"Training Pipeline failed: {str(e)}", exc_info=True)
        
        # Save error info
        error_info = {
            "status": "failed",
            "mode": "training",
            "error": str(e),
            "build_id": os.getenv('BUILD_ID', 'local')
        }
        
        try:
            from utils.gcs_utils import upload_json_to_gcs
            upload_json_to_gcs(
                error_info, 
                Config.GCS_MODEL_BUCKET, 
                f"pipeline_runs/failed_training_run_{os.getenv('BUILD_ID', 'local')}.json"
            )
        except:
            logger.error("Failed to save error info to GCS")
            
        return False

def main():
    """
    Main function - runs training pipeline when called from Cloud Build
    """
    # Check if we're in training mode
    training_mode = os.getenv('TRAINING_MODE', '').lower() == 'true'
    is_cloud_build = os.getenv('BUILD_ID') is not None
    
    if training_mode or is_cloud_build:
        logger.info("Running in TRAINING mode")
        success = run_training_pipeline()
        if not success:
            exit(1)  # Exit with error code for Cloud Build
    else:
        logger.info("No training mode specified. Use TRAINING_MODE=true to run training.")

if __name__ == "__main__":
    run_training_pipeline()