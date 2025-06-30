# fraud_detection_pipeline/feature_engineering/production_features.py

import pandas as pd
import numpy as np
import os
import fsspec

# Helper functions for feature engineering, potentially reused from train_features.py
def flag_outliers_iqr(series):
    """
    Flags outliers in a numerical series using the Interquartile Range (IQR) method.
    In a real production system, Q1 and Q3 values from training data should be loaded
    and used here to maintain consistency. For this example, it's recomputed.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).astype(int)

def rolling_unique_set(series, window):
    """
    Calculates the number of unique elements in a rolling window for a series.
    """
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_slice = series.iloc[start:i+1]
        result.append(len(set(window_slice)))
    return pd.Series(result, index=series.index)

def load_lookup_series(name, lookup_dir='lookups/'):
    """
    Loads a lookup series (e.g., target encoding mapping) from a CSV file.
    These lookups are generated during the training feature engineering phase.
    Works with both local paths and GCS paths (e.g., gs://your-bucket/lookups/).
    """
    path = lookup_dir + f'{name}_lookup.csv'

    try:
        # Use fsspec to open GCS and local files transparently
        fs = fsspec.open(path).fs
        if fs.exists(path):
            return pd.read_csv(path, index_col=0).squeeze("columns")
        else:
            print(f"Warning: Lookup file not found: {path}. Returning empty Series. This might lead to NaNs.")
            return pd.Series()
    except Exception as e:
        print(f"Error loading lookup {name} from {path}: {e}. Returning empty Series.")
        return pd.Series()

def generate_production_features(input_path='data/transactions_production.csv',
                                 output_path='production_feature_engineered.csv',
                                 lookup_dir='lookups/'):
    """
    Generates features for the production dataset using pre-computed lookup tables
    from the training phase for target-dependent features.

    Args:
        input_path (str): Path to the raw production data CSV (without target).
        output_path (str): Path to save the feature-engineered production data.
        lookup_dir (str): Directory where lookup tables generated during training are stored.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded raw production data from {input_path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw production data file not found at {input_path}. Please run data_clean_split.py first.")
        return

    # 1. Convert Timestamp and extract time components
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    print("Timestamp converted and time components extracted for production data.")

    # 2. Transaction Amount Features
    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1)
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])
    print("Transaction amount features generated for production data.")

    # 3. Account-Based Features
    # For production, these counts/stats should ideally be rolling or pre-computed
    # to avoid re-calculating on every new small batch. For demonstration, we compute within the batch.
    sender_txn_counts = df['Account'].value_counts(normalize=False)
    df['sender_total_txn'] = df['Account'].map(sender_txn_counts).fillna(0).astype(int)

    receiver_txn_counts = df['Account.1'].value_counts(normalize=False)
    df['receiver_total_txn'] = df['Account.1'].map(receiver_txn_counts).fillna(0).astype(int)

    unique_receivers = df.groupby('Account')['Account.1'].nunique()
    df['unique_receivers_per_sender'] = df['Account'].map(unique_receivers).fillna(0).astype(int)

    unique_senders = df.groupby('Account.1')['Account'].nunique()
    df['unique_senders_per_receiver'] = df['Account.1'].map(unique_senders).fillna(0).astype(int)

    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Sender'] = df['Time_Since_Last_Txn_Sender'].fillna(-1)

    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Receiver'] = df['Time_Since_Last_Txn_Receiver'].fillna(-1)
    print("Account-based features generated for production data.")

    # 4. Interaction Features (Frequency Encodings)
    df['Bank_Pair'] = df['From Bank'].astype(str) + '_' + df['To Bank'].astype(str)
    pair_freq = df['Bank_Pair'].value_counts(normalize=True) # Freq based on current production batch
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
    print("Interaction features generated for production data.")

    # 5. Behavioral Features (Rolling averages/stds/velocities)
    window_size = 5
    df = df.sort_values('Timestamp')

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9
    window_seconds = 3600

    def txn_velocity(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Sender'] = df.groupby('Account').apply(txn_velocity).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')

    df['Rolling_Std_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
        .rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

    epsilon = 1e-6

    df['Rolling_Avg_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
                                               .rolling(window=window_size, min_periods=1)\
                                               .mean().reset_index(level=0, drop=True)

    df['Rolling_Avg_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    df['Rolling_Avg_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .mean().reset_index(level=0, drop=True)

    df['Rolling_Avg_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .mean().reset_index(level=0, drop=True)

    df['Amount_Received_to_Avg_Sender_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Sender'] + epsilon)
    df['Amount_Paid_to_Avg_Sender_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Sender'] + epsilon)
    df['Amount_Received_to_Avg_Receiver_Ratio'] = df['Amount Received'] / (df['Rolling_Avg_Amount_Received_Receiver'] + epsilon)
    df['Amount_Paid_to_Avg_Receiver_Ratio'] = df['Amount Paid'] / (df['Rolling_Avg_Amount_Paid_Receiver'] + epsilon)

    window_size_txns = 20
    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1']\
        .apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9

    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')
    print("Behavioral features generated for production data.")

    # 6. Categorical Encoding (One-Hot Encoding)
    # Frequency encoding for production data is often re-computed on the batch
    # or requires a stored lookup of frequencies from training data.
    # Here, we re-compute for the current batch.
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']

    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc).fillna(0) # Fill unknown with 0 frequency

    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc).fillna(0)
    print("Categorical features encoded for production data.")

    # --- Enrich Production Data with Lookup Table Features (Target Encodings and Fraud Rates) ---
    # These features are crucial and MUST use the values learned from the training set
    # to avoid data leakage and ensure consistency.
    print("Loading and applying lookup tables for target-dependent features...")
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
    print("Lookup-based features applied to production data.")

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
    print("High fraud period flags generated for production data.")

    # Drop interim columns that are not features for the model
    drop_cols_interim = [
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat', 'Day_Hour', 'Bank_Payment_Hour'
    ]
    df_processed = df.drop(columns=[col for col in drop_cols_interim if col in df.columns], errors='ignore')

    # Save the processed feature engineered production data
    df_processed.to_csv(output_path, index=False)
    print(f"Feature engineered production data saved to: {output_path}. Final shape: {df_processed.shape}")


if __name__ == "__main__":
    # This block allows the script to be run directly for production feature engineering.
    # It assumes 'data/transactions_production.csv' and the 'lookups' directory exist.
    print("Running production feature engineering process...")
    if not os.path.exists('data/transactions_production.csv'):
        print("Warning: 'data/transactions_production.csv' not found. Please run 'data_preprocessing/data_clean_split.py' first.")
    if not os.path.exists('lookups'):
        print("Warning: 'lookups' directory not found. Please run 'feature_engineering/train_features.py' first to generate lookups.")
    generate_production_features()

