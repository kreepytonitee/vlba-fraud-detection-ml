import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts time-based features."""
    df['hour_of_day'] = df['Timestamp'].dt.hour
    return df

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates ratio-based features."""
    # Avoid division by zero
    df['amount_diff'] = df['Amount Paid'] - df['Amount Received']
    df['amount_ratio'] = df.apply(lambda row: row['Amount Paid'] / row['Amount Received'] if row['Amount Received'] != 0 else 0, axis=1)
    return df

def create_bank_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Frequency of each From Bank
    from_bank_freq = df['From Bank'].value_counts().to_dict()
    df['from_bank_freq'] = df['From Bank'].map(from_bank_freq)

    # Frequency of each To Bank
    to_bank_freq = df['To Bank'].value_counts().to_dict()
    df['to_bank_freq'] = df['To Bank'].map(to_bank_freq)

    # Frequency of each Account (sender)
    account_freq = df['Account'].value_counts().to_dict()
    df['account_freq'] = df['Account'].map(account_freq)

    # Frequency of each Account.1 (receiver)
    account1_freq = df['Account.1'].value_counts().to_dict()
    df['account1_freq'] = df['Account.1'].map(account1_freq)

    # Mean amount received by Account
    account_mean_received = df.groupby('Account')['Amount Received'].mean().to_dict()
    df['account_mean_received'] = df['Account'].map(account_mean_received)

    # Mean amount paid to Account.1
    account1_mean_paid = df.groupby('Account.1')['Amount Paid'].mean().to_dict()
    df['account1_mean_paid'] = df['Account.1'].map(account1_mean_paid)

    return df

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Encode categorical columns
    categorical_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col])
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps."""
    df = create_time_features(df)
    df = create_ratio_features(df)
    df = create_bank_interaction_features(df)
    df = encode_categorical_columns(df)
    return df

if __name__ == "__main__":
    # Dummy data for demonstration
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
    
    engineered_df = apply_feature_engineering(sample_df.copy())
    print(engineered_df.head())