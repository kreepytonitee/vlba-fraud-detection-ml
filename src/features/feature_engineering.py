import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts time-based features."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
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

PYDANTIC_TO_ORIGINAL_COL_MAP = {
    "transaction_id": "Unnamed: 0",
    "timestamp": "Timestamp",
    "from_bank": "From Bank",
    "account": "Account",
    "to_bank": "To Bank",
    "account_1": "Account.1",
    "amount_received": "Amount Received",
    "receiving_currency": "Receiving Currency",
    "amount_paid": "Amount Paid",
    "payment_currency": "Payment Currency",
    "payment_format": "Payment Format"
    # Make sure all fields from your TransactionData model are covered here
}

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Create the reverse mapping: original_col_name -> pydantic_attr_name
    original_to_pydantic_map = {
        original_col_name: pydantic_attr_name
        for pydantic_attr_name, original_col_name in PYDANTIC_TO_ORIGINAL_COL_MAP.items()
    }

    # Filter the mapping to only include columns that exist in the DataFrame
    rename_mapping_for_df = {
        original_col_name: pydantic_attr_name
        for original_col_name, pydantic_attr_name in original_to_pydantic_map.items()
        if original_col_name in df.columns
    }

    df = df.rename(columns=rename_mapping_for_df)
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps."""
    df = create_time_features(df)
    df = create_ratio_features(df)
    df = create_bank_interaction_features(df)
    df = encode_categorical_columns(df)
    df = rename_columns(df)
    return df