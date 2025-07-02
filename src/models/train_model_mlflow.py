# # 1. Install MLflow and pyngrok - in requirements.txt
# !pip install mlflow pyngrok lightgbm xgboost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import os
import fsspec

# 2. Import required modules
import threading
import time
from pyngrok import ngrok

# 3. Set your ngrok authtoken (get it from https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_AUTH_TOKEN = "2zI5kxX0uEC3zQULiosBMa8kBW0_6Lr4nyktMNk9ujeY8M48n"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# 4. Start MLflow UI in background and tunnel via ngrok
def run_mlflow_ui():
    get_ipython().system_raw('mlflow ui --port 5000 &')

threading.Thread(target=run_mlflow_ui).start()
time.sleep(5)  # Give the server time to start

public_url = ngrok.connect(5000)
print('MLflow UI is live at:', public_url)


# 4. Training and logging code (update input_path as needed!)
def objective_score(precision, recall, f1):
    return 0.6 * recall + 0.3 * f1 + 0.1 * precision

def train_and_save_model(input_path='feature_engineered.csv',
                         model_output_dir='trained_models/',
                         experiment_name='FraudDetection_Experiments'):
    mlflow.set_experiment(experiment_name)

    try:
        df_model = pd.read_csv(input_path)
        print(f"Successfully loaded feature-engineered data from {input_path}. Shape: {df_model.shape}")
    except FileNotFoundError:
        print(f"Error: Feature engineered training data not found at {input_path}.")
        return

    # drop_cols_for_model = [
    #     'Account', 'Account.1', 'Timestamp', 'From Bank', 'To Bank', 'Day', 'Hour', 'Minute', 'Unnamed: 0'
    # ]
    # df_model = df_model.drop(columns=[col for col in drop_cols_for_model if col in df_model.columns], errors='ignore')

    X = df_model.drop(columns=['Is Laundering'])
    y = df_model['Is Laundering']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_scores = {}

    with mlflow.start_run(run_name="Model_Comparison") as parent_run:
        # Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        score_rf = objective_score(precision_rf, recall_rf, f1_rf)
        model_scores['RandomForest'] = (rf, score_rf)
        with mlflow.start_run(run_name="RandomForest", nested=True):
            mlflow.log_param('model', 'RandomForest')
            mlflow.log_metric('precision', precision_rf)
            mlflow.log_metric('recall', recall_rf)
            mlflow.log_metric('f1_score', f1_rf)
            mlflow.log_metric('objective_score', score_rf)
            mlflow.sklearn.log_model(rf, "model", registered_model_name="FraudDetectionRandomForest")

        # LightGBM
        lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
        lgb_clf.fit(X_train, y_train)
        y_pred_lgb = lgb_clf.predict(X_test)
        precision_lgb = precision_score(y_test, y_pred_lgb)
        recall_lgb = recall_score(y_test, y_pred_lgb)
        f1_lgb = f1_score(y_test, y_pred_lgb)
        score_lgb = objective_score(precision_lgb, recall_lgb, f1_lgb)
        model_scores['LightGBM'] = (lgb_clf, score_lgb)
        with mlflow.start_run(run_name="LightGBM", nested=True):
            mlflow.log_param('model', 'LightGBM')
            mlflow.log_metric('precision', precision_lgb)
            mlflow.log_metric('recall', recall_lgb)
            mlflow.log_metric('f1_score', f1_lgb)
            mlflow.log_metric('objective_score', score_lgb)
            mlflow.lightgbm.log_model(lgb_clf, "model", registered_model_name="FraudDetectionLightGBM")

        # XGBoost
        xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        precision_xgb = precision_score(y_test, y_pred_xgb)
        recall_xgb = recall_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        score_xgb = objective_score(precision_xgb, recall_xgb, f1_xgb)
        model_scores['XGBoost'] = (xgb_clf, score_xgb)
        with mlflow.start_run(run_name="XGBoost", nested=True):
            mlflow.log_param('model', 'XGBoost')
            mlflow.log_metric('precision', precision_xgb)
            mlflow.log_metric('recall', recall_xgb)
            mlflow.log_metric('f1_score', f1_xgb)
            mlflow.log_metric('objective_score', score_xgb)
            mlflow.xgboost.log_model(xgb_clf, "model", registered_model_name="FraudDetectionXGBoost")

        # Ensemble
        ensemble_clf = VotingClassifier(
            estimators=[('rf', rf), ('lgb', lgb_clf), ('xgb', xgb_clf)],
            voting='soft',
            n_jobs=-1
        )
        ensemble_clf.fit(X_train, y_train)
        y_pred_ensemble = ensemble_clf.predict(X_test)
        precision_ensemble = precision_score(y_test, y_pred_ensemble)
        recall_ensemble = recall_score(y_test, y_pred_ensemble)
        f1_ensemble = f1_score(y_test, y_pred_ensemble)
        score_ensemble = objective_score(precision_ensemble, recall_ensemble, f1_ensemble)
        model_scores['Ensemble'] = (ensemble_clf, score_ensemble)
        with mlflow.start_run(run_name="Ensemble", nested=True):
            mlflow.log_param('model', 'VotingClassifier')
            mlflow.log_metric('precision', precision_ensemble)
            mlflow.log_metric('recall', recall_ensemble)
            mlflow.log_metric('f1_score', f1_ensemble)
            mlflow.log_metric('objective_score', score_ensemble)
            mlflow.sklearn.log_model(ensemble_clf, "model", registered_model_name="FraudDetectionEnsemble")

        # Best model registration
        best_model_name = max(model_scores, key=lambda k: model_scores[k][1])
        best_model, best_score = model_scores[best_model_name]
        print(f"\nBest Model: {best_model_name} (Objective Score: {best_score:.4f})")

        if best_model_name == "RandomForest":
            mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="FraudDetectionBestModel")
        elif best_model_name == "LightGBM":
            mlflow.lightgbm.log_model(best_model, "best_model", registered_model_name="FraudDetectionBestModel")
        elif best_model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, "best_model", registered_model_name="FraudDetectionBestModel")
        else:
            mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="FraudDetectionBestModel")

    model_save_path = model_output_dir + 'model.pkl'
    try:
        with fsspec.open(model_save_path, 'wb') as f:
            joblib.dump(best_model, f)
        print(f"Model saved to: {model_save_path}")
    except Exception as e:
        print(f"Error saving model to {model_save_path}: {e}")

# 5. Local test
if __name__ == "__main__":
    # This block allows the script to be run directly for model training.
    # # It assumes 'feature_engineered.csv' exists.
    print("Running model training process...")
    if not os.path.exists('feature_engineered.csv'):
        print("Warning: 'feature_engineered.csv' not found. Please run 'feature_engineering/train_features.py' first.")
    os.makedirs('trained_models', exist_ok=True)
    train_and_save_model(input_path="C:/Users/anhng/Downloads/VSCode-Python/vlba-fd/anna/data/feature_engineered.csv")

