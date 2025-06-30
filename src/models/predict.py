import pandas as pd
import joblib
import os
from src.features.feature_engineering import apply_feature_engineering

def predict_fraud(data_path: str, model_path: str) -> pd.DataFrame:
    """
    Loads a trained fraud classification model and makes predictions on new data.
    Args:
        data_path (str): Path to the new data CSV file for prediction.
        model_path (str): Path to the trained model (.joblib file).
    Returns:
        pd.DataFrame: DataFrame with original data and 'Predicted_Is_Laundering' column.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    # Load the trained model
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")

    # Load new data
    df_new = pd.read_csv(data_path)

    # Apply the same feature engineering steps
    df_processed = apply_feature_engineering(df_new.copy())

    # Prepare features for prediction
    X_predict = df_processed.drop(columns=['Unnamed: 0', 'Timestamp', 'Is Laundering', 'From Bank', 'Account', 'To Bank', 'Account.1'], errors='ignore')

    # Ensure all columns used during training are present, fill missing with 0 (or appropriate value)
    # This assumes that all features engineered during training are present in new data after feature engineering.
    # In a production system, a more robust feature alignment strategy would be needed.
    train_features = model.feature_names_in_
    for col in train_features:
        if col not in X_predict.columns:
            X_predict[col] = 0 # Or the mean/median from training data
    X_predict = X_predict[train_features] # Ensure column order is consistent

    # Handle potential NaN values introduced by mapping
    X_predict = X_predict.fillna(0)

    # Make predictions
    df_new['Predicted_Is_Laundering'] = model.predict(X_predict)

    return df_new

if __name__ == '__main__':
    training_data_file = 'transactions_train.csv'
    model_output_file = 'fraud_detection_model.joblib'

    # Example of how to use the predict_fraud function with the same data for demonstration
    # In a real scenario, you would pass new, unseen data to predict_fraud
    print("\n--- Running Prediction Example ---")
    predictions_df = predict_fraud(training_data_file, model_output_file)
    print(predictions_df[['Timestamp', 'Amount Paid', 'Is Laundering', 'Predicted_Is_Laundering']].head())