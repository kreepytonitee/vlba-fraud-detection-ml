import pandas as pd
import numpy as np
import json
import sys
sys.path.append(r"C:/Users/anhng/Downloads/VSCode-Python/vlba-fd/git-ver1/src")

from utils.logger import logger

"""1. Converting Timestamp to datetime and arranging them in ascending order to prevent data leakage"""
def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Sort by Timestamp ascending
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    return df

"""2. Transaction Amount Features"""
def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform to reduce skewness (use log1p to handle zero values safely)
    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])

    # Absolute difference between received and paid amounts
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])

    # Ratio of received to paid amount (adding 1 to denominator to avoid division by zero)
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1)

    # Function to flag outliers based on IQR
    def flag_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).astype(int)

    # Create outlier flags for Amount Received and Amount Paid
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])
    return df

"""3. Account-Based Features"""
def create_account_features(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate stats per sender (Account)
    sender_agg = df.groupby('Account').agg(
        sender_total_txn=('Is Laundering', 'count'),
        sender_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    sender_agg['sender_fraud_rate'] = sender_agg['sender_fraud_txn'] / sender_agg['sender_total_txn']

    # Aggregate stats per receiver (Account.1)
    receiver_agg = df.groupby('Account.1').agg(
        receiver_total_txn=('Is Laundering', 'count'),
        receiver_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    receiver_agg['receiver_fraud_rate'] = receiver_agg['receiver_fraud_txn'] / receiver_agg['receiver_total_txn']

    # Merge these stats back to the main dataframe
    df = df.merge(sender_agg[['Account', 'sender_total_txn', 'sender_fraud_rate']], on='Account', how='left')
    df = df.merge(receiver_agg[['Account.1', 'receiver_total_txn', 'receiver_fraud_rate']], on='Account.1', how='left')

    # Unique receivers per sender
    unique_receivers = df.groupby('Account')['Account.1'].nunique().reset_index().rename(columns={'Account.1':'unique_receivers_per_sender'})
    df = df.merge(unique_receivers, on='Account', how='left')

    # Unique senders per receiver
    unique_senders = df.groupby('Account.1')['Account'].nunique().reset_index().rename(columns={'Account':'unique_senders_per_receiver'})
    df = df.merge(unique_senders, on='Account.1', how='left')

    # Time since last transaction per sender
    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds()
    # Fill NaN (first transaction) with some large value or -1
    df['Time_Since_Last_Txn_Sender'] = df['Time_Since_Last_Txn_Sender'].fillna(-1)

    # Time since last transaction per receiver
    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Receiver'] = df['Time_Since_Last_Txn_Receiver'].fillna(-1)
    return df

"""4. Time based features"""
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extract time components if not already extracted
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    # 1. Fraud rate by Day, Hour, Minute (from your EDA)
    fraud_rate_by_day = df.groupby('Day')['Is Laundering'].mean()
    fraud_rate_by_hour = df.groupby('Hour')['Is Laundering'].mean()
    fraud_rate_by_minute = df.groupby('Minute')['Is Laundering'].mean()

    # Map fraud rates back to dataframe as features
    df['Fraud_Rate_By_Day'] = df['Day'].map(fraud_rate_by_day)
    df['Fraud_Rate_By_Hour'] = df['Hour'].map(fraud_rate_by_hour)
    df['Fraud_Rate_By_Minute'] = df['Minute'].map(fraud_rate_by_minute)

    # 2. Flags for high-risk periods based on thresholds (mean fraud rate)
    day_thresh = fraud_rate_by_day.mean()
    hour_thresh = fraud_rate_by_hour.mean()
    minute_thresh = fraud_rate_by_minute.mean()

    df['High_Fraud_Day'] = (df['Fraud_Rate_By_Day'] > day_thresh).astype(int)
    df['High_Fraud_Hour'] = (df['Fraud_Rate_By_Hour'] > hour_thresh).astype(int)
    df['High_Fraud_Minute'] = (df['Fraud_Rate_By_Minute'] > minute_thresh).astype(int)
    return df

"""5. Interaction Features"""
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Sender Bank + Receiver Bank
    df['Bank_Pair'] = df['From Bank'].astype(str) + '_' + df['To Bank'].astype(str)
    pair_freq = df['Bank_Pair'].value_counts(normalize=True)
    df['Bank_Pair_freq_enc'] = df['Bank_Pair'].map(pair_freq)

    # 2. Sender Account + Receiver Account
    df['Account_Pair'] = df['Account'].astype(str) + '_' + df['Account.1'].astype(str)
    account_pair_freq = df['Account_Pair'].value_counts(normalize=True)
    df['Account_Pair_freq_enc'] = df['Account_Pair'].map(account_pair_freq)

    # 3. Payment Format + Hour
    df['PaymentFormat_Hour'] = df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    payment_hour_freq = df['PaymentFormat_Hour'].value_counts(normalize=True)
    df['PaymentFormat_Hour_freq_enc'] = df['PaymentFormat_Hour'].map(payment_hour_freq)

    # 4. Sender Account + Payment Format
    df['Sender_PaymentFormat'] = df['Account'].astype(str) + '_' + df['Payment Format'].astype(str)
    sender_payment_freq = df['Sender_PaymentFormat'].value_counts(normalize=True)
    df['Sender_PaymentFormat_freq_enc'] = df['Sender_PaymentFormat'].map(sender_payment_freq)

    # 5. Day + Hour interaction
    df['Day_Hour'] = df['Day'].astype(str) + '_' + df['Hour'].astype(str)
    day_hour_freq = df['Day_Hour'].value_counts(normalize=True)
    df['Day_Hour_freq_enc'] = df['Day_Hour'].map(day_hour_freq)

    # 6. From Bank + Payment Format + Hour
    df['Bank_Payment_Hour'] = df['From Bank'].astype(str) + '_' + df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    bank_pay_hour_freq = df['Bank_Payment_Hour'].value_counts(normalize=True)
    df['Bank_Payment_Hour_freq_enc'] = df['Bank_Payment_Hour'].map(bank_pay_hour_freq)
    return df

"""6. Behavioral Features"""
def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling Average Transaction Amount per Sender"""


    window_size = 5  # last 5 transactions

    df = df.sort_values('Timestamp')

    df['Rolling_Avg_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    df['Rolling_Avg_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    """Transaction Velocity: Number of Transactions per Sender in a Time Window"""

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9  # convert to seconds

    window_seconds = 3600  # 1 hour window

    def txn_velocity(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Sender'] = df.groupby('Account').apply(txn_velocity).reset_index(level=0, drop=True)

    df.drop(columns=['Timestamp_unix'], inplace=True)

    """Rolling Standard Deviation of Transaction Amounts (Sender & Receiver)"""

    window_size = 5  # last 5 transactions

    # Rolling std dev per sender
    df['Rolling_Std_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .std().reset_index(level=0, drop=True).fillna(0)

    # Rolling std dev per receiver
    df['Rolling_Std_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .std().reset_index(level=0, drop=True).fillna(0)

    """Transaction Amount Ratio (Current vs Rolling Mean) — Sender & Receiver"""

    # To avoid division by zero, add a small epsilon
    epsilon = 1e-6

    df['Rolling_Avg_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .mean().reset_index(level=0, drop=True)

    df['Amount_Received_to_Avg_Sender_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Sender'] + epsilon)

    df['Rolling_Avg_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    df['Amount_Paid_to_Avg_Sender_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Sender'] + epsilon)

    # Similarly for receiver
    df['Rolling_Avg_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .mean().reset_index(level=0, drop=True)

    df['Amount_Received_to_Avg_Receiver_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Receiver'] + epsilon)

    df['Rolling_Avg_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    df['Amount_Paid_to_Avg_Receiver_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Receiver'] + epsilon)

    """Rolling Unique Counterparties per Sender"""

    window_size_txns = 20  # last 20 transactions

    def rolling_unique_set(series, window):
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            window_slice = series.iloc[start:i+1]
            result.append(len(set(window_slice)))
        return pd.Series(result, index=series.index)

    df = df.sort_values('Timestamp')

    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1'].apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)

    """Transaction Velocity for Receiver (Last Hour)"""

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9

    window_seconds = 3600

    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)

    df.drop(columns=['Timestamp_unix'], inplace=True)
    return df

"""7. Categorical Encoding"""
def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Columns
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']

    # Frequency Encoding
    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)

    # Target Encoding
    for col in freq_target_cols:
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    """Checking all the columns and dropping Unnesscary columns"""

    for col in df.columns:
        print(col)

    # Frequency Encoding
    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)

    # Target Encoding (mean fraud rate)
    for col in ['Account', 'Account.1']:
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)

    drop_cols = [
        'Account', 'Account.1',
        'Unnamed: 0', 'Timestamp',
        'From Bank', 'To Bank',
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat', 'Day_Hour', 'Bank_Payment_Hour',
        'Amount Received', 'Amount Paid'
    ]

    df_model = df.drop(columns=drop_cols)
    return df_model

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps."""
    df = convert_timestamp(df)
    df = create_amount_features(df)
    df = create_account_features(df)
    df = create_time_features(df)
    df = create_interaction_features(df)
    df = create_behavioral_features(df)
    df = encode_categorical_columns(df)
    logger.info("Feature engineering applied.")
    # Write column names to JSON
    with open('data/feature_columns.json', 'w') as f:
        json.dump(df.columns.tolist(), f)
    # Save DataFrame to CSV
    df.to_csv('data/feature_engineered.csv', index=False)
    # Log confirmation
    logger.info("Feature-engineered data saved to 'feature_engineered.csv'. Columns saved to 'feature_columns.json'.")
    return df

if __name__ == "__main__":

    df = pd.read_csv("C:/Users/anhng/Downloads/VSCode-Python/vlba-fd/git-ver1/data/transactions_train.csv")
    df = convert_timestamp(df)
    df = create_amount_features(df)
    df = create_account_features(df)
    df = create_time_features(df)
    df = create_interaction_features(df)
    df = create_behavioral_features(df)
    df = encode_categorical_columns(df)
    logger.info("Feature engineering applied.")

    # # Write column names to JSON - usually extracted from trained model
    # with open('data/feature_columns.json', 'w') as f:
    #     json.dump(df.columns.tolist(), f)
    # # Extract column names and data types to CSV - for examination
    # col_info = pd.DataFrame({
    #     "column": df.columns,
    #     "dtype": df.dtypes.astype(str).values
    # })
    # col_info.to_csv("data/feature_column_dtypes.csv", index=False)

    # Save DataFrame to CSV
    df.to_csv('data/feature_engineered.csv', index=False)
    # Log confirmation
    logger.info("Feature-engineered data saved to 'feature_engineered.csv'. Columns saved to 'feature_columns.json'.")