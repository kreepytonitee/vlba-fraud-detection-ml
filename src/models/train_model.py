import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import os
import fsspec

def objective_score(precision, recall, f1):
    """
    Define Objective Scoring Function
    """
    return 0.6 * recall + 0.3 * f1 + 0.1 * precision

def train_and_save_model(input_path='feature_engineered.csv',
                         model_output_dir='trained_models/'):
    """
    Loads feature-engineered data, trains baseline and ensemble models,
    and saves the best performing model (XGBoost based on original script's choice).

    Args:
        input_path (str): Path to the feature-engineered training data.
        model_output_dir (str): Directory to save the trained model.
    """
    try:
        df_model = pd.read_csv(input_path)
        print(f"Successfully loaded feature-engineered data from {input_path}. Shape: {df_model.shape}")
    except FileNotFoundError:
        print(f"Error: Feature engineered training data not found at {input_path}. Please run feature_engineering/train_features.py first.")
        return

    # # Ensure model output directory exists
    # os.makedirs(model_output_dir, exist_ok=True)

    # # Drop columns not intended for direct model input but present in the DataFrame.
    # # These include original identifiers and timestamps that have been processed into features.
    # drop_cols_for_model = [
    #     'Account', 'Account.1', 'Timestamp', 'From Bank', 'To Bank', 'Day', 'Hour', 'Minute', 'Unnamed: 0'
    # ]
    # # Filter to drop only columns that actually exist in the DataFrame
    # df_model = df_model.drop(columns=[col for col in drop_cols_for_model if col in df_model.columns], errors='ignore')
    # print("Dropped identifier/original columns not used for direct model training.")

    # Separating features (X) and target (y)
    X = df_model.drop(columns=['Is Laundering'])
    y = df_model['Is Laundering']
    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    # Splitting the data into training and testing sets (80% train, 20% test)
    # Stratify ensures that the proportion of the target variable is the same in both splits.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training (X_train: {X_train.shape}) and testing (X_test: {X_test.shape}) sets.")

    # --- Dictionary to Track Model Objects and Scores ---
    model_scores = {}

    # --- Train and Evaluate Random Forest ---
    print("\n--- Training Baseline Random Forest Classifier ---")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    score_rf = objective_score(precision_rf, recall_rf, f1_rf)
    model_scores['RandomForest'] = (rf, score_rf)

    print("Random Forest Baseline Performance:")
    print(classification_report(y_test, y_pred_rf, digits=4))
    print(f"Objective Score: {score_rf:.4f} (Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}, Precision: {precision_rf:.4f})")

    # --- Train and Evaluate LightGBM ---
    print("\n--- Training Baseline LightGBM ---")
    lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    lgb_clf.fit(X_train, y_train)
    y_pred_lgb = lgb_clf.predict(X_test)

    precision_lgb = precision_score(y_test, y_pred_lgb)
    recall_lgb = recall_score(y_test, y_pred_lgb)
    f1_lgb = f1_score(y_test, y_pred_lgb)
    score_lgb = objective_score(precision_lgb, recall_lgb, f1_lgb)
    model_scores['LightGBM'] = (lgb_clf, score_lgb)

    print("LightGBM Baseline Performance:")
    print(classification_report(y_test, y_pred_lgb, digits=4))
    print(f"Objective Score: {score_lgb:.4f} (Recall: {recall_lgb:.4f}, F1: {f1_lgb:.4f}, Precision: {precision_lgb:.4f})")

    # --- Train and Evaluate XGBoost ---
    print("\n--- Training Baseline XGBoost ---")
    # use_label_encoder=False and eval_metric='logloss' are set for compatibility with newer XGBoost versions
    xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)

    precision_xgb = precision_score(y_test, y_pred_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    score_xgb = objective_score(precision_xgb, recall_xgb, f1_xgb)
    model_scores['XGBoost'] = (xgb_clf, score_xgb)

    print("XGBoost Baseline Performance:")
    print(classification_report(y_test, y_pred_xgb, digits=4))
    print(f"Objective Score: {score_xgb:.4f} (Recall: {recall_xgb:.4f}, F1: {f1_xgb:.4f}, Precision: {precision_xgb:.4f})")

    # --- Train and Evaluate Ensemble Model ---
    print("\n--- Training Ensemble Model (VotingClassifier) ---")
    # Ensemble combines predictions from multiple models (soft voting uses probabilities)
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

    print("Ensemble Model Performance:")
    print(classification_report(y_test, y_pred_ensemble, digits=4))
    print(f"Objective Score: {score_ensemble:.4f} (Recall: {recall_ensemble:.4f}, F1: {f1_ensemble:.4f}, Precision: {precision_ensemble:.4f})")

    # --- Select and Save the Best Model ---                    
    best_model_name = max(model_scores, key=lambda k: model_scores[k][1])
    best_model, best_score = model_scores[best_model_name]

    print(f"\n Best Model: {best_model_name} with Objective Score: {best_score:.4f}")
    model_save_path = model_output_dir + 'model.pkl'
    try:
        with fsspec.open(model_save_path, 'wb') as f:
            joblib.dump(best_model, f)
        print(f"Model saved to: {model_save_path}")
    except Exception as e:
        print(f"Error saving model to {model_save_path}: {e}")

if __name__ == "__main__":
    # This block allows the script to be run directly for model training.
    # # It assumes 'feature_engineered.csv' exists.
    print("Running model training process...")
    if not os.path.exists('feature_engineered.csv'):
        print("Warning: 'feature_engineered.csv' not found. Please run 'feature_engineering/train_features.py' first.")
    os.makedirs('trained_models', exist_ok=True)
    train_and_save_model(input_path="C:/Users/anhng/Downloads/VSCode-Python/vlba-fd/anna/data/feature_engineered.csv")

