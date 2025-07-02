import os
import sys

# # Add the project root to sys.path to enable imports of pipeline modules
# project_root = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, project_root)

# Import GCS utilities (still needed for checks)
from utils.gcs_utils import check_gcs_blob_exists

# Import core pipeline functions (now GCS-aware)
from data_preprocessing.data_clean_split import clean_and_split_data
from feature_engineering.train_features import generate_training_features
from feature_engineering.production_features import generate_production_features
from models.train_model import train_and_save_model
from inference.predict import predict_on_production_data

# Import Config for GCS paths
from utils.config import Config
from utils.logger import logger

def run_cloud_pipeline():
    """
    Orchestrates the ML pipeline, with all core ML pipeline functions.
    """
    print("--- Starting Cloud Orchestrated ML Pipeline (GCS Direct Processing) ---")

    # --- Step 1: Data Preprocessing ---
    print("\n--- Stage 1: Data Preprocessing ---")
    # Check if raw data exists in GCS
    if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_RAW_DATA_FILE):
        logger.error(f"Raw data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_RAW_DATA_FILE}) not found.")
        logger.info("Please upload your 'transactions.csv' to this GCS location manually and try again.")
        return

    # Call the GCS-aware data cleaning and splitting function
    try:
        clean_and_split_data(
            input_csv_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_RAW_DATA_FILE}",
            train_output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_TRAIN_DATA_FILE}",
            prod_output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}",
            target_col='Is Laundering',
            test_size=0.25,
            random_state=42
        )
        logger.info("Data preprocessing completed and results uploaded to GCS.")
    except Exception as e:
        logger.info(f"Pipeline failed during Data Preprocessing: {e}")
        return


    # --- Step 2: Feature Engineering (Training Data) ---
    print("\n--- Stage 2: Feature Engineering (Training Data) ---")
    # Check if training data exists in GCS (output from previous step)
    if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_TRAIN_DATA_FILE):
        logger.error(f"Training data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_TRAIN_DATA_FILE}) not found.")
        logger.info("Ensure Data Preprocessing stage completed successfully.")
        return

    # Call the GCS-aware feature engineering function for training data
    try:
        generate_training_features(
            input_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_TRAIN_DATA_FILE}",
            output_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE}",
            feature_cols_output_path=f"gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_FEATURE_COLUMNS_FILE}",
            lookup_dir=f"gs://{Config.GCS_LOOKUPS_BUCKET}/"
        )
        logger.info("Training feature engineering completed and artifacts uploaded to GCS.")
    except Exception as e:
        logger.info(f"Pipeline failed during Training Feature Engineering: {e}")
        return


    # --- Step 3: Model Training ---
    print("\n--- Stage 3: Model Training ---")
    # Check if feature engineered training data exists in GCS
    if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE):
        logger.error(f"Feature engineered training data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE}) not found.")
        logger.info("Ensure Training Feature Engineering stage completed successfully.")
        return

    # Call the GCS-aware model training function
    try:
        train_and_save_model(
            input_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_TRAIN_FILE}",
            model_output_dir=f"gs://{Config.GCS_MODEL_BUCKET}/"
        )
        logger.info("Model training completed and model uploaded to GCS.")
    except Exception as e:
        logger.info(f"Pipeline failed during Model Training: {e}")
        return


    # # --- Step 4: Feature Engineering (Production Data) --- will be handled by FastAPI
    # print("\n--- Stage 4: Feature Engineering (Production Data) ---")
    # # Check if raw production data and lookups exist in GCS
    # if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_PRODUCTION_DATA_FILE):
    #     logger.error(f"Error: Raw production data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}) not found.")
    #     logger.info("Ensure Data Preprocessing stage completed successfully.")
    #     return
    # if not check_gcs_blob_exists(Config.GCS_LOOKUPS_BUCKET, 'Fraud_Rate_By_Day_lookup.csv'): # Check a representative lookup
    #     logger.error(f"Error: Lookup files not found in gs://{Config.GCS_LOOKUPS_BUCKET}.")
    #     logger.info("Ensure Training Feature Engineering stage completed successfully to generate lookups.")
    #     return

    # # Call the GCS-aware feature engineering function for production data
    # try:
    #     generate_production_features(
    #         input_path=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_PRODUCTION_DATA_FILE}",
    #         output_path_data=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_PROD_FILE}",
    #         output_path_label=f"gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_LABEL_PROD_FILE}",
    #         lookup_dir=f"gs://{Config.GCS_LOOKUPS_BUCKET}/"
    #     )
    #     logger.info("Production feature engineering completed and results uploaded to GCS.")
    # except Exception as e:
    #     logger.info(f"Pipeline failed during Production Feature Engineering: {e}")
    #     return


    # # --- Step 5: Prediction --- will be handled by FastAPI
    # GCS Variable must update from BLOB to FILE, print replace by logger, input on prediction function must be adapted
    # print("\n--- Stage 5: Prediction ---")
    # # Check if feature engineered production data, model, and feature columns exist in GCS
    # if not check_gcs_blob_exists(Config.GCS_DATA_BUCKET, Config.GCS_FEATURE_ENGINEERED_PROD_FILE):
    #     print(f"Error: Feature engineered production data (gs://{Config.GCS_DATA_BUCKET}/{Config.GCS_FEATURE_ENGINEERED_PROD_FILE}) not found.")
    #     print("Ensure Production Feature Engineering stage completed successfully.")
    #     return
    # if not check_gcs_blob_exists(Config.GCS_MODEL_BUCKET, Config.GCS_MODEL_FILE):
    #     print(f"Error: Trained model (gs://{Config.GCS_MODEL_BUCKET}/{Config.GCS_MODEL_FILE}) not found.")
    #     print("Ensure Model Training stage completed successfully.")
    #     return
    # if not check_gcs_blob_exists(Config.GCS_LOOKUPS_BUCKET, Config.GCS_FEATURE_COLUMNS_FILE):
    #     print(f"Error: Feature columns file (gs://{Config.GCS_LOOKUPS_BUCKET}/{Config.GCS_FEATURE_COLUMNS_FILE}) not found.")
    #     print("Ensure Training Feature Engineering stage completed successfully.")
    #     return

    # # Call the GCS-aware prediction function
    # try:
    #     predict_on_production_data(
    #         feature_engineered_input_bucket=Config.GCS_DATA_BUCKET,
    #         feature_engineered_input_blob=Config.GCS_FEATURE_ENGINEERED_PROD_FILE,
    #         model_bucket=Config.GCS_MODEL_BUCKET,
    #         model_blob=Config.GCS_MODEL_FILE,
    #         feature_columns_bucket=Config.GCS_LOOKUPS_BUCKET,
    #         feature_columns_blob=Config.GCS_FEATURE_COLUMNS_FILE,
    #         predictions_output_bucket=Config.GCS_DATA_BUCKET,
    #         predictions_output_blob=Config.GCS_PREDICTIONS_FILE
    #     )
    #     print("Prediction completed and results uploaded to GCS.")
    # except Exception as e:
    #     print(f"Pipeline failed during Prediction: {e}")
    #     return


    logger.info("\n--- Cloud Orchestrated ML Pipeline Completed Successfully ---")


if __name__ == "__main__":
    # The --clear-local argument is removed as local directories are no longer used
    # for intermediate storage by the orchestrator.
    run_cloud_pipeline()

