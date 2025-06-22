import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------ Preprocessing ------------------

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Convert data types
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    numeric_columns = ['Amount Received', 'Amount Paid', 'Is Laundering']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    # # Drop high cardinality and redundant features
    # df = df.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)

    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42):
    # ------------------ Data Split ------------------

    # First, split into training + (validation + test)
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['is_laundering'])
    
    # Then, split the temp_df into validation and test sets
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['is_laundering']) # 0.5 of temp_df means 0.5 * 0.2 = 0.1 of original
    
    print(f"Data split: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Create a dummy dataframe for demonstration
    data = {
        'Is Laundering': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'Amount Paid': [100, 5000, 200, 150, 7000, 300, 250, 6000, 180, 400],
        'Amount Received': [90, 4800, 190, 140, 6800, 280, 240, 5900, 170, 390],
        'Timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 10:05:00', '2025-01-02 11:00:00', 
                                        '2025-01-02 11:15:00', '2025-01-03 12:00:00', '2025-01-03 12:30:00',
                                        '2025-01-04 09:00:00', '2025-01-04 09:10:00', '2025-01-05 13:00:00',
                                        '2025-01-05 13:20:00']),
        'Payment Currency': ['USD', 'USD', 'EUR', 'USD', 'GBP', 'EUR', 'USD', 'USD', 'EUR', 'USD'],
        'Receiving Currency': ['USD', 'USD', 'EUR', 'USD', 'GBP', 'EUR', 'USD', 'USD', 'EUR', 'USD'],
        'From Bank': ['BankA', 'BankB', 'BankC', 'BankA', 'BankD', 'BankC', 'BankA', 'BankB', 'BankC', 'BankA'],
        'To Bank': ['BankX', 'BankY', 'BankZ', 'BankX', 'BankW', 'BankZ', 'BankX', 'BankY', 'BankZ', 'BankX']
    }
    sample_df = pd.DataFrame(data)

    cleaned_df = clean_data(sample_df)
    train_df, val_df, test_df = split_data(cleaned_df)

    # Save processed data (optional, but good practice)
    train_df.to_csv("../../data/processed/training_data.csv", index=False)
    val_df.to_csv("../../data/processed/validation_data.csv", index=False)
    test_df.to_csv("../../data/processed/testing_data.csv", index=False)