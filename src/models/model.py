import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.features.feature_engineering import apply_feature_engineering

def train_and_save_model(data_path: str, model_path: str):
    """
    Trains a fraud classification model and saves it to a joblib file.
    Args:
        data_path (str): Path to the training data CSV file.
        model_path (str): Path to save the trained model (.joblib file).
    """
    # Load the training data
    df = pd.read_csv(data_path)

    # Apply feature engineering
    df = apply_feature_engineering(df.copy())

    # Define features (X) and target (y)
    # Exclude non-numeric and target columns from features
    X = df.drop(columns=['Unnamed: 0', 'Timestamp', 'Is Laundering', 'From Bank', 'Account', 'To Bank', 'Account.1'], errors='ignore')
    y = df['Is Laundering']

    # Handle potential NaN values introduced by mapping in feature engineering
    # For simplicity, filling with 0 or a suitable constant.
    # In a real-world scenario, more sophisticated imputation might be needed.
    X = X.fillna(0)

    # Split data into training and testing sets (optional, but good practice for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Training Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == '__main__':
    training_data_file = 'transactions_train.csv'
    model_output_file = 'src/models/fraud_detection_model.joblib'

    # Train and save the model
    train_and_save_model(training_data_file, model_output_file)
