# fraud_detection_pipeline/data_preprocessing/data_clean_split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def clean_and_split_data(input_csv_path='data/transactions.csv',
                         train_output_path='data/transactions_train.csv',
                         prod_output_path='data/transactions_production.csv',
                         target_col='Is Laundering',
                         test_size=0.25,
                         random_state=42):
    """
    Loads raw transaction data, handles missing target values,
    and splits the data into training and production sets.

    Args:
        input_csv_path (str): Path to the raw transactions CSV file.
        train_output_path (str): Path to save the training data CSV.
        prod_output_path (str): Path to save the production data CSV (without target).
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the production split.
        random_state (int): Random seed for reproducibility.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded data from {input_csv_path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}. Please ensure it exists.")
        return

    # Check for missing values in target before the split and drop them
    missing_targets = df[target_col].isnull().sum()
    if missing_targets > 0:
        print(f"Missing values in '{target_col}': {missing_targets}. Dropping rows.")
        df = df.dropna(subset=[target_col])
        print(f"Data shape after dropping missing target rows: {df.shape}")
    else:
        print(f"No missing values in '{target_col}'.")

    # Performing stratified split: 75% train, 25% production
    # Stratified split ensures the proportion of target classes is maintained in both sets.
    train_df, prod_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # Remove target column from production data before saving
    # This simulates a real production scenario where the label is unknown.
    prod_df_no_label = prod_df.drop(columns=[target_col])

    # # Ensure output directory exists
    # os.makedirs(os.path.dirname(train_output_path), exist_ok=True)

    # Save train data WITH LABELS
    train_df.to_csv(train_output_path, index=False)
    print(f"Training data saved to: {train_output_path}")

    # Save production data WITHOUT LABELS
    prod_df_no_label.to_csv(prod_output_path, index=False)
    print(f"Production data (without labels) saved to: {prod_output_path}")

    # Print class distribution for confirmation of stratified split
    print("\nTraining data class distribution:")
    print(train_df[target_col].value_counts(normalize=True))

    print("\nProduction data class distribution (from original production split):")
    # We use the original prod_df (with labels) here to show distribution for verification
    print(prod_df[target_col].value_counts(normalize=True))

if __name__ == "__main__":
    # This block allows the script to be run directly for data cleaning and splitting.
    # It will create the 'data' directory if it doesn't exist.
    print("Running data cleaning and splitting process...")
    clean_and_split_data()

