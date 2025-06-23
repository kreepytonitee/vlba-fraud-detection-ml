import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logger import logger

def process_and_split_data(df: pd.DataFrame):
    # Let's check for missing values in target before the split and drop them
    df = df.dropna(subset=['Is Laundering'])
        
    # Let’s store the Target column in a variable
    target_col = 'Is Laundering'

    # For the "production" set, we'll take a small sample from the test set.
    # In a real scenario, this would come from a live stream or separate source.

    # Performing stratified split: 75% train, 25% production (Splitting it in 75-25 ratio)
    train_df, prod_df = train_test_split(
        df,
        test_size=0.25,
        stratify=df[target_col],
        random_state=42
    )
    
    # This line remove's target column from production data before saving
    prod_df_no_label = prod_df.drop(columns=[target_col])

    # Save train data WITH LABELS
    train_df.to_csv('data/transactions_train.csv', index=False)

    # Save production data WITHOUT LABELS
    prod_df_no_label.to_csv('data/transactions_production.csv', index=False)

    logger.info("Training data saved to 'transactions_train.csv'. Production data saved to 'transactions_production.csv'.")