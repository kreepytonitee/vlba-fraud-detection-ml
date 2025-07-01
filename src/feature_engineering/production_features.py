import pandas as pd
import numpy as np
import os
import fsspec
from google.cloud import bigquery

def flag_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).astype(int)

def rolling_unique_set(series, window):
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_slice = series.iloc[start:i+1]
        result.append(len(set(window_slice)))
    return pd.Series(result, index=series.index)

def load_lookup_series(name, lookup_dir='lookups/'):
    path = lookup_dir + f'{name}_lookup.csv'
    try:
        fs = fsspec.open(path).fs
        if fs.exists(path):
            return pd.read_csv(path, index_col=0).squeeze("columns")
        else:
            print(f"Warning: Lookup file not found: {path}. Returning empty Series. This might lead to NaNs.")
            return pd.Series()
    except Exception as e:
        print(f"Error loading lookup {name} from {path}: {e}. Returning empty Series.")
        return pd.Series()

def generate_production_features(
    input_path='data/transactions_production.csv',
    output_bq_table='vlba-fd.fd.feature_engineered_production',
    lookup_dir='lookups/'
):
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded raw production data from {input_path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw production data file not found at {input_path}. Please run data_clean_split.py first.")
        return

    # Create transaction_id as unique entity for Feature Store (composite key)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['transaction_id'] = (
        df['Account'].astype(str) + "_" +
        df['Account.1'].astype(str) + "_" +
        df['From Bank'].astype(str) + "_" +
        df['To Bank'].astype(str) + "_" +
        df['Timestamp'].astype(str)
    )
    print("transaction_id created for production data.")

    # 1. Convert Timestamp and extract time components
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    # 2. Transaction Amount Features
    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1)
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])

    # 3. Account-Based Features
    sender_txn_counts = df['Account'].value_counts(normalize=False)
    df['sender_total_txn'] = df['Account'].map(sender_txn_counts).fillna(0).astype(int)
    receiver_txn_counts = df['Account.1'].value_counts(normalize=False)
    df['receiver_total_txn'] = df['Account.1'].map(receiver_txn_counts).fillna(0).astype(int)
    unique_receivers = df.groupby('Account')['Account.1'].nunique()
    df['unique_receivers_per_sender'] = df['Account'].map(unique_receivers).fillna(0).astype(int)
    unique_senders = df.groupby('Account.1')['Account'].nunique()
    df['unique_senders_per_receiver'] = df['Account.1'].map(unique_senders).fillna(0).astype(int)
    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds().fillna(-1)
    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds().fillna(-1)

    # 4. Interaction Features (Frequency Encodings)
    df['Bank_Pair'] = df['From Bank'].astype(str) + '_' + df['To Bank'].astype(str)
    pair_freq = df['Bank_Pair'].value_counts(normalize=True)
    df['Bank_Pair_freq_enc'] = df['Bank_Pair'].map(pair_freq).fillna(0)
    df['Account_Pair'] = df['Account'].astype(str) + '_' + df['Account.1'].astype(str)
    account_pair_freq = df['Account_Pair'].value_counts(normalize=True)
    df['Account_Pair_freq_enc'] = df['Account_Pair'].map(account_pair_freq).fillna(0)
    df['PaymentFormat_Hour'] = df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    payment_hour_freq = df['PaymentFormat_Hour'].value_counts(normalize=True)
    df['PaymentFormat_Hour_freq_enc'] = df['PaymentFormat_Hour'].map(payment_hour_freq).fillna(0)
    df['Sender_PaymentFormat'] = df['Account'].astype(str) + '_' + df['Payment Format'].astype(str)
    sender_payment_freq = df['Sender_PaymentFormat'].value_counts(normalize=True)
    df['Sender_PaymentFormat_freq_enc'] = df['Sender_PaymentFormat'].map(sender_payment_freq).fillna(0)
    df['Day_Hour'] = df['Day'].astype(str) + '_' + df['Hour'].astype(str)
    day_hour_freq = df['Day_Hour'].value_counts(normalize=True)
    df['Day_Hour_freq_enc'] = df['Day_Hour'].map(day_hour_freq).fillna(0)
    df['Bank_Payment_Hour'] = df['From Bank'].astype(str) + '_' + df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    bank_pay_hour_freq = df['Bank_Payment_Hour'].value_counts(normalize=True)
    df['Bank_Payment_Hour_freq_enc'] = df['Bank_Payment_Hour'].map(bank_pay_hour_freq).fillna(0)

    # 5. Behavioral Features (Rolling averages/stds/velocities)
    window_size = 5
    df = df.sort_values('Timestamp')
    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    window_seconds = 3600
    def txn_velocity(group):
        times = group['Timestamp_unix'].values
        return pd.Series([(times >= t - window_seconds).sum() for t in times], index=group.index)
    df['Txn_Velocity_Sender'] = df.groupby('Account').apply(txn_velocity).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')
    df['Rolling_Std_Amount_Received_Sender'] = df.groupby('Account')['Amount Received'].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid'].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received'].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    df['Rolling_Std_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid'].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    epsilon = 1e-6
    df['Rolling_Avg_Amount_Received_Sender'] = df.groupby('Account')['Amount Received'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Rolling_Avg_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Amount_Received_to_Avg_Sender_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Sender'] + epsilon)
    df['Amount_Paid_to_Avg_Sender_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Sender'] + epsilon)
    df['Amount_Received_to_Avg_Receiver_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Receiver'] + epsilon)
    df['Amount_Paid_to_Avg_Receiver_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Receiver'] + epsilon)
    window_size_txns = 20
    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1'].apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)
    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        return pd.Series([(times >= t - window_seconds).sum() for t in times], index=group.index)
    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')

    # 6. Categorical Encoding (One-Hot Encoding)
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']
    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc).fillna(0)
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc).fillna(0)

    # --- Load Lookups ---
    fraud_rate_by_day = load_lookup_series('Fraud_Rate_By_Day', lookup_dir)
    fraud_rate_by_hour = load_lookup_series('Fraud_Rate_By_Hour', lookup_dir)
    fraud_rate_by_minute = load_lookup_series('Fraud_Rate_By_Minute', lookup_dir)
    sender_fraud_rate = load_lookup_series('sender_fraud_rate', lookup_dir)
    receiver_fraud_rate = load_lookup_series('receiver_fraud_rate', lookup_dir)
    from_bank_target_enc = load_lookup_series('From_Bank_target_enc', lookup_dir)
    to_bank_target_enc = load_lookup_series('To_Bank_target_enc', lookup_dir)
    account_target_enc = load_lookup_series('Account_target_enc', lookup_dir)
    account1_target_enc = load_lookup_series('Account.1_target_enc', lookup_dir)
    df['Fraud_Rate_By_Day'] = df['Day'].map(fraud_rate_by_day).fillna(0)
    df['Fraud_Rate_By_Hour'] = df['Hour'].map(fraud_rate_by_hour).fillna(0)
    df['Fraud_Rate_By_Minute'] = df['Minute'].map(fraud_rate_by_minute).fillna(0)
    df['sender_fraud_rate'] = df['Account'].map(sender_fraud_rate).fillna(0)
    df['receiver_fraud_rate'] = df['Account.1'].map(receiver_fraud_rate).fillna(0)
    df['From Bank_target_enc'] = df['From Bank'].map(from_bank_target_enc).fillna(0)
    df['To Bank_target_enc'] = df['To Bank'].map(to_bank_target_enc).fillna(0)
    df['Account_target_enc'] = df['Account'].map(account_target_enc).fillna(0)
    df['Account.1_target_enc'] = df['Account.1'].map(account1_target_enc).fillna(0)

    # Load thresholds for High_Fraud_Day/Hour/Minute flags from training
    day_thresh_series = load_lookup_series('day_thresh', lookup_dir)
    day_thresh = day_thresh_series.iloc[0] if not day_thresh_series.empty else 0
    hour_thresh_series = load_lookup_series('hour_thresh', lookup_dir)
    hour_thresh = hour_thresh_series.iloc[0] if not hour_thresh_series.empty else 0
    minute_thresh_series = load_lookup_series('minute_thresh', lookup_dir)
    minute_thresh = minute_thresh_series.iloc[0] if not minute_thresh_series.empty else 0
    df['High_Fraud_Day'] = (df['Fraud_Rate_By_Day'] > day_thresh).astype(int)
    df['High_Fraud_Hour'] = (df['Fraud_Rate_By_Hour'] > hour_thresh).astype(int)
    df['High_Fraud_Minute'] = (df['Fraud_Rate_By_Minute'] > minute_thresh).astype(int)

    # Drop interim columns and all identifiers except transaction_id (entity)
    drop_cols = [
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat', 'Day_Hour', 'Bank_Payment_Hour',
        'Amount Received', 'Amount Paid', 'Account', 'Account.1', 'From Bank', 'To Bank', 'Timestamp'
    ]
    df_processed = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # --- Export to BigQuery instead of CSV ---
    print(f"Exporting feature engineered production data to BigQuery table: {output_bq_table}")
    df_processed.to_gbq(output_bq_table, project_id="vlba-fd", if_exists="replace")
    print(f"Feature engineered production data exported to BigQuery: {output_bq_table}.")

if __name__ == "__main__":
    print("Running production feature engineering process for Feature Store...")
    if not os.path.exists('data/transactions_production.csv'):
        print("Warning: 'data/transactions_production.csv' not found. Please run 'data_preprocessing/data_clean_split.py' first.")
    if not os.path.exists('lookups'):
        print("Warning: 'lookups' directory not found. Please run 'feature_engineering/train_features.py' first to generate lookups.")
    generate_production_features()
