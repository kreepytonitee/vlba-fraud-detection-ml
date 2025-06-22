import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import mlflow.pyfunc
import json

def evaluate_model(model_uri: str, X_test: pd.DataFrame, y_test: pd.Series, feature_cols_path: str):
    """Evaluates a deployed model from MLflow on the test set."""

    # Load the model from MLflow
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded from MLflow URI: {model_uri}")

    # Load feature columns
    with open(feature_cols_path, "r") as f:
        feature_columns = json.load(f)

    # Ensure test data has the same columns as training data, including one-hot encoding
    categorical_cols = X_test.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        X_test = pd.get_dummies(X_test, columns=list(categorical_cols), drop_first=True)

    # Reindex X_test to match the order of feature_columns from training
    X_test = X_test.reindex(columns=feature_columns, fill_value=0) # Fill missing new cols with 0

    y_pred = model.predict(X_test)
    # For MLflow PyFuncModel, predict may return probabilities if the model was logged that way
    # Try to infer probabilities if possible
    y_proba = None
    try:
        # If the model returns a 2D array or DataFrame with probabilities, use the second column
        y_pred_proba = model.predict(X_test)
        if hasattr(y_pred_proba, "shape") and len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
            y_proba = y_pred_proba[:, 1]
        elif hasattr(y_pred_proba, "columns") and "1" in y_pred_proba.columns:
            y_proba = y_pred_proba["1"].values
    except Exception:
        y_proba = None

    print("\n--- Model Evaluation on Test Set ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC: {roc_auc:.4f}")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
    else:
        print("\nModel does not support predict_proba, ROC AUC and PR AUC skipped.")


if __name__ == "__main__":
    # Load processed test data
    test_df = pd.read_csv("../../data/processed/testing_data.csv")

    # Apply feature engineering to the loaded test data
    from src.features.feature_engineering import apply_feature_engineering
    test_df = apply_feature_engineering(test_df.copy())

    # Define features (X) and target (y)
    features = [col for col in test_df.columns if col not in ['is_laundering', 'date_time']]
    X_test_raw = test_df[features]
    y_test = test_df['is_laundering']

    # You need to replace this with the actual MLflow URI from your training run
    # Example URI: "runs:/<RUN_ID>/fraud_detection_model" or "models:/XGBoostFraudDetector/latest"
    # To get RUN_ID, check your mlflow_tracking/ directory or the MLflow UI
    mlflow_run_id = "YOUR_LAST_MLFLOW_RUN_ID" # <<< REPLACE THIS
    model_uri = f"runs:/{mlflow_run_id}/fraud_detection_model"
    feature_cols_path = "feature_columns.json" # This file should be an artifact of the MLflow run

    # To get this artifact locally, you might need to use mlflow.artifacts.download_artifacts
    # Or, during CI/CD, it would be available from the run.
    # For local testing, ensure 'feature_columns.json' is in the root of the project
    # or adjust path if you've already manually copied it.

    # For a practical demonstration, ensure you've run train_model.py at least once
    # to generate a `feature_columns.json` file in the current working directory.
    # If not, create a dummy one for testing:
    # with open("feature_columns.json", "w") as f:
    #     json.dump(['amount_sent', 'amount_received', 'amount_diff', 'amount_ratio', 'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'bank_pair_BankA_BankX'], f)

    evaluate_model(model_uri, X_test_raw, y_test, feature_cols_path)