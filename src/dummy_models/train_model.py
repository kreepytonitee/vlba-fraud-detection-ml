import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
from typing import Union
import json # For saving feature columns

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
    """Trains an XGBoost classifier with hyperparameter tuning and logs with MLflow."""

    # Define the target column (assuming 'is_laundering') and features
    # Filter out non-numeric or directly identifiable columns like original text ones
    # and ensure all categorical features are one-hot encoded or handled by XGBoost

    # For simplicity, let's assume one-hot encoding for categorical features
    # This should ideally be part of your feature engineering pipeline
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        X_train = pd.get_dummies(X_train, columns=list(categorical_cols), drop_first=True)
        X_val = pd.get_dummies(X_val, columns=list(categorical_cols), drop_first=True)

        # Align columns after one-hot encoding
        common_cols = list(set(X_train.columns) & set(X_val.columns))
        X_train = X_train[common_cols]
        X_val = X_val[common_cols]

    # Hyperparameter tuning (simplified for demonstration)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)

    best_model = grid_search.best_estimator_
    print(f"Best model parameters: {grid_search.best_params_}")

    # MLflow Tracking
    mlflow.set_tracking_uri("file:./mlruns") # Local MLflow tracking
    with mlflow.start_run(run_name="Fraud_Detection_Training"):
        mlflow.log_params(grid_search.best_params_)

        # Evaluate on validation set
        y_pred_val = best_model.predict(X_val)
        y_proba_val = best_model.predict_proba(X_val)[:, 1]

        # report_val = classification_report(y_val, y_pred_val, output_dict=True)

        roc_auc_val = roc_auc_score(y_val, y_proba_val)
        precision_val, recall_val, _ = precision_recall_curve(y_val, y_proba_val)
        pr_auc_val = auc(recall_val, precision_val)

        # mlflow.log_metrics({
        #     "val_precision": float(report_val['1']['precision']),
        #     "val_recall": float(report_val['1']['recall']),
        #     "val_f1_score": float(report_val['1']['f1-score']),
        #     "val_roc_auc": float(roc_auc_val),
        #     "val_pr_auc": float(pr_auc_val)
        # })
        # print(f"Validation F1-score: {report_val['1']['f1-score']:.4f}")

        raw_report = classification_report(y_val, y_pred_val, output_dict=True)

        if not isinstance(raw_report, dict):
            raise TypeError("Expected classification_report to return a dict")

        report_val: dict[str, Union[dict[str, float], float, str]] = raw_report

        metrics_1 = report_val.get('1')
        if not isinstance(metrics_1, dict):
            raise TypeError("Expected metrics for label '1' to be a dict")

        val_precision = float(metrics_1.get('precision', 0.0))
        val_recall = float(metrics_1.get('recall', 0.0))
        val_f1_score = float(metrics_1.get('f1-score', 0.0))

        mlflow.log_metrics({
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1_score,
            "val_roc_auc": float(roc_auc_val),
            "val_pr_auc": float(pr_auc_val)
        })

        print(f"Validation F1-score: {val_f1_score:.4f}")


        # Log the model
        mlflow.xgboost.log_model(best_model, "fraud_detection_model", # type: ignore
                                    registered_model_name="XGBoostFraudDetector")

        # Save feature columns for later inference
        feature_columns = list(X_train.columns)
        with open("feature_columns.json", "w") as f:
            json.dump(feature_columns, f)
        mlflow.log_artifact("feature_columns.json")

        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"Model logged to MLflow run {active_run.info.run_id}")
        else:
            print("Model logged to MLflow, but no active run found.")
    return best_model, feature_columns

if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv("../../data/processed/training_data.csv")
    val_df = pd.read_csv("../../data/processed/validation_data.csv")

    # Apply feature engineering to the loaded data (important for consistent features)
    from features.feature_engineering_H import apply_feature_engineering
    train_df = apply_feature_engineering(train_df.copy())
    val_df = apply_feature_engineering(val_df.copy())

    # Define features (X) and target (y)
    # Drop original identifier columns that are not features
    # Ensure 'date_time', 'currency_sent', 'currency_received', 'bank_sender', 'bank_receiver'
    # are handled. For `train_model`, we'll pass X with only numeric/encoded features.

    # Example of selecting relevant features and handling categorical for training
    features = [col for col in train_df.columns if col not in ['is_laundering', 'date_time']]
    X_train_raw = train_df[features]
    y_train = train_df['is_laundering']

    X_val_raw = val_df[features]
    y_val = val_df['is_laundering']

    # Call train_model with raw features; it will handle encoding internally for simplicity here
    # In a real pipeline, preprocessing would be a separate step before passing to train_model
    trained_model, feature_cols = train_model(X_train_raw, y_train, X_val_raw, y_val)
    print("Model training complete.")