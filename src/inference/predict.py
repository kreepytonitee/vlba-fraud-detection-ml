import pandas as pd
import joblib
import os
import sys

def predict_on_production_data(raw_input_path='data/transactions_production.csv',
                               feature_engineered_path='production_feature_engineered.csv',
                               model_path='trained_models/model.pkl',
                               feature_columns_path='feature_columns.txt',
                               predictions_output_path='production_predictions.csv',
                               lookup_dir='lookups/'):
    """
    Loads raw production data, ensures it's feature engineered,
    makes predictions with the trained model, and saves the results.

    Args:
        raw_input_path (str): Path to the raw production data CSV.
        feature_engineered_path (str): Expected path of the feature-engineered production data.
        model_path (str): Path to the trained model file.
        feature_columns_path (str): Path to the file containing list of feature columns used by the model.
        predictions_output_path (str): Path to save the final predictions.
        lookup_dir (str): Directory containing lookup tables for feature engineering.
    """
    print("Starting prediction pipeline...")

    # Step 1: Ensure raw production data is feature engineered.
    # In a real-world setting, this might be a separate scheduled job.
    # Here, for demonstration, we will call the production_features.py logic if the
    # feature_engineered_path doesn't exist.
    if not os.path.exists(feature_engineered_path):
        print(f"Feature engineered data not found at {feature_engineered_path}.")
        print("Attempting to run production feature engineering now...")
        # Dynamically import and run the feature engineering for production.
        # Add the parent directory of this script to sys.path to enable relative import
        current_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        sys.path.insert(0, project_root)
        try:
            from feature_engineering.production_features import generate_production_features
            generate_production_features(input_path=raw_input_path,
                                         output_path=feature_engineered_path,
                                         lookup_dir=lookup_dir)
            print("Production feature engineering completed.")
        except ImportError:
            print("Error: Could not import 'generate_production_features'. Ensure 'feature_engineering/production_features.py' is correctly defined.")
            return
        except Exception as e:
            print(f"Error during production feature engineering: {e}")
            return
        finally:
            # Remove the added path to keep sys.path clean for other imports
            if project_root in sys.path:
                sys.path.remove(project_root)
    else:
        print(f"Found existing feature engineered data at {feature_engineered_path}. Skipping re-engineering.")


    try:
        df = pd.read_csv(feature_engineered_path)
        print(f"Loaded feature engineered production data from: {feature_engineered_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Feature engineered production data still not found at {feature_engineered_path} after attempt to generate. Cannot proceed.")
        return

    # Store original identifier columns before dropping them for prediction
    original_data_for_output = df[['Account', 'Account.1', 'Timestamp', 'From Bank', 'To Bank', 'Unnamed: 0']].copy()

    # Step 2: Load feature columns used during model training
    try:
        with open(feature_columns_path, 'r') as f:
            feature_cols = [line.strip() for line in f if line.strip() != "Unnamed: 0"]
        print(f"Loaded {len(feature_cols)} feature columns from: {feature_columns_path}")
    except FileNotFoundError:
        print(f"Error: Feature columns file not found at {feature_columns_path}. Cannot proceed.")
        return

    # Step 3: Align columns with the trained model's expected features
    # This is crucial for one-hot encoded features where a category might be
    # present in training but not in a specific production batch, or vice-versa.
    missing_cols = set(feature_cols) - set(df.columns)
    for c in missing_cols:
        df[c] = 0 # Add missing columns and fill with zero (common for OHE)
        print(f"Adding missing feature column for prediction: {c}")

    # Drop any columns in df that are NOT in feature_cols, and ensure order
    # (Exclude 'Is Laundering' if it somehow ended up here, though it shouldn't for production data)
    cols_to_keep = [col for col in feature_cols if col in df.columns]
    X_prod = df[cols_to_keep]

    # Handle any columns in feature_cols that are numeric but became objects due to mixed types (e.g., during read_csv with missing values)
    for col in X_prod.columns:
        if X_prod[col].dtype == 'object':
            try:
                X_prod[col] = pd.to_numeric(X_prod[col], errors='coerce').fillna(0) # Convert to numeric, fill NaNs
                print(f"Warning: Converted object column '{col}' to numeric and filled NaNs with 0.")
            except ValueError:
                print(f"Warning: Column '{col}' is object type and cannot be converted to numeric. It might cause model errors.")

    # Reorder X_prod columns to match the exact order expected by the model
    # (based on feature_columns.txt)
    X_prod = X_prod[feature_cols] # This reorders and adds missing columns as 0, or drops extra ones
    print(f"Prepared X_prod for prediction with shape: {X_prod.shape}")


    # Step 4: Load the trained model
    try:
        model = joblib.load(model_path)
        print(f"Loaded trained model from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Trained model not found at {model_path}. Please run models/train_model.py first.")
        return
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}. Ensure the model file is valid.")
        return

    # Step 5: Make predictions
    print("Generating predictions...")
    df['Is_Laundering_Pred'] = model.predict(X_prod)
    df['Is_Laundering_Prob'] = model.predict_proba(X_prod)[:, 1]
    print("Predictions generated.")

    # Step 6: Combine with original identifiers and save predictions
    # We use original_data_for_output to ensure Account, Account.1, Timestamp are present.
    # Then merge with predictions from df.
    predictions_df = original_data_for_output.copy()
    predictions_df['Is_Laundering_Pred'] = df['Is_Laundering_Pred']
    predictions_df['Is_Laundering_Prob'] = df['Is_Laundering_Prob']

    # Ensure the output directory exists
    # os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    predictions_df.to_csv(predictions_output_path, index=False)
    print(f"Production predictions saved to: {predictions_output_path}")

if __name__ == "__main__":
    # This block allows the script to be run directly for inference.
    # It will attempt to run preceding steps if their outputs are missing.
    print("Running the end-to-end prediction pipeline...")

    # Set up necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('lookups', exist_ok=True)
    os.makedirs('trained_models', exist_ok=True)

    # Orchestration of the pipeline steps:
    # 1. Data Preprocessing (Clean and Split)
    if not os.path.exists('data/transactions_train.csv') or not os.path.exists('data/transactions_production.csv'):
        print("\n--- Running Data Preprocessing ---")
        try:
            from data_preprocessing.data_clean_split import clean_and_split_data
            clean_and_split_data()
        except ImportError:
            print("Error: Could not import 'clean_and_split_data'. Please ensure 'data_preprocessing/data_clean_split.py' is in your Python path.")
        except Exception as e:
            print(f"An error occurred during data preprocessing: {e}")

    # 2. Training Feature Engineering (to generate 'feature_engineered.csv' and 'lookups')
    if not os.path.exists('feature_engineered.csv') or not os.path.exists('lookups/Fraud_Rate_By_Day_lookup.csv'):
        print("\n--- Running Training Feature Engineering ---")
        try:
            from feature_engineering.train_features import generate_training_features
            generate_training_features()
        except ImportError:
            print("Error: Could not import 'generate_training_features'. Please ensure 'feature_engineering/train_features.py' is in your Python path.")
        except Exception as e:
            print(f"An error occurred during training feature engineering: {e}")

    # 3. Model Training (to generate 'model.pkl')
    if not os.path.exists('trained_models/model.pkl'):
        print("\n--- Running Model Training ---")
        try:
            from models.train_model import train_and_save_model
            train_and_save_model()
        except ImportError:
            print("Error: Could not import 'train_and_save_model'. Please ensure 'models/train_model.py' is in your Python path.")
        except Exception as e:
            print(f"An error occurred during model training: {e}")

    # 4. Prediction on Production Data
    print("\n--- Running Prediction ---")
    predict_on_production_data()

