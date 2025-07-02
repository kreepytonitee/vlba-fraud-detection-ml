import pandas as pd
import numpy as np
import os
import fsspec

def flag_outliers_iqr(series):
    """
    Flags outliers in a numerical series using the Interquartile Range (IQR) method.

    Args:
        series (pd.Series): The input numerical series.

    Returns:
        pd.Series: A series of integers (0 or 1) indicating whether each value is an outlier.
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

    Args:
        series (pd.Series): The input series.
        window (int): The size of the rolling window.

    Returns:
        pd.Series: A series containing the count of unique elements for each rolling window.
    """
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_slice = series.iloc[start:i+1]
        result.append(len(set(window_slice)))
    return pd.Series(result, index=series.index)

def generate_training_features(input_path='data/transactions_train.csv',
                               output_path='feature_engineered.csv',
                               feature_cols_output_path='feature_columns.txt',
                               lookup_dir='lookups/'):
    """
    Generates all features for the training dataset and exports lookup tables
    (e.g., target encodings, fraud rates) based on the training data.

    Args:
        input_path (str): Path to the training data CSV.
        output_path (str): Path to save the feature-engineered training data.
        feature_cols_output_path (str): Path to save the list of final feature columns.
        lookup_dir (str): Directory to save lookup tables for production inference.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded training data from {input_path}. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {input_path}. Please run data_clean_split.py first.")
        return

    # # Ensure lookup directory exists
    # os.makedirs(lookup_dir, exist_ok=True)

    # 1. Convert Timestamp to datetime and arrange in ascending order
    # This is crucial for time-series related features and preventing data leakage.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    print("Timestamp converted and data sorted.")

    # 2. Transaction Amount Features
    # Log-transform to reduce skewness and handle potential outliers.
    df['Log_Amount_Received'] = np.log1p(df['Amount Received'])
    df['Log_Amount_Paid'] = np.log1p(df['Amount Paid'])
    df['Amount_Diff'] = abs(df['Amount Received'] - df['Amount Paid'])
    df['Amount_Ratio'] = df['Amount Received'] / (df['Amount Paid'] + 1) # Add 1 to denominator to avoid division by zero
    df['Outlier_Amount_Received'] = flag_outliers_iqr(df['Amount Received'])
    df['Outlier_Amount_Paid'] = flag_outliers_iqr(df['Amount Paid'])
    print("Transaction amount features generated.")

    # 3. Account-Based Features
    # Aggregate statistics per sender and receiver.
    sender_agg = df.groupby('Account').agg(
        sender_total_txn=('Is Laundering', 'count'),
        sender_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    sender_agg['sender_fraud_rate'] = sender_agg['sender_fraud_txn'] / sender_agg['sender_total_txn']

    receiver_agg = df.groupby('Account.1').agg(
        receiver_total_txn=('Is Laundering', 'count'),
        receiver_fraud_txn=('Is Laundering', 'sum')
    ).reset_index()
    receiver_agg['receiver_fraud_rate'] = receiver_agg['receiver_fraud_txn'] / receiver_agg['receiver_total_txn']

    df = df.merge(sender_agg[['Account', 'sender_total_txn', 'sender_fraud_rate']], on='Account', how='left')
    df = df.merge(receiver_agg[['Account.1', 'receiver_total_txn', 'receiver_fraud_rate']], on='Account.1', how='left')

    # Unique counterparties per sender/receiver to detect unusual behavior.
    unique_receivers = df.groupby('Account')['Account.1'].nunique().reset_index().rename(columns={'Account.1':'unique_receivers_per_sender'})
    df = df.merge(unique_receivers, on='Account', how='left')

    unique_senders = df.groupby('Account.1')['Account'].nunique().reset_index().rename(columns={'Account':'unique_senders_per_receiver'})
    df = df.merge(unique_senders, on='Account.1', how='left')

    # Time since last transaction to capture frequency.
    df['Time_Since_Last_Txn_Sender'] = df.groupby('Account')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Sender'] = df['Time_Since_Last_Txn_Sender'].fillna(-1) # Fill NaN for first transaction

    df['Time_Since_Last_Txn_Receiver'] = df.groupby('Account.1')['Timestamp'].diff().dt.total_seconds()
    df['Time_Since_Last_Txn_Receiver'] = df['Time_Since_Last_Txn_Receiver'].fillna(-1)
    print("Account-based features generated.")

    # 4. Time-based features
    # Extract temporal components and calculate fraud rates by time.
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    fraud_rate_by_day = df.groupby('Day')['Is Laundering'].mean()
    fraud_rate_by_hour = df.groupby('Hour')['Is Laundering'].mean()
    fraud_rate_by_minute = df.groupby('Minute')['Is Laundering'].mean()

    df['Fraud_Rate_By_Day'] = df['Day'].map(fraud_rate_by_day)
    df['Fraud_Rate_By_Hour'] = df['Hour'].map(fraud_rate_by_hour)
    df['Fraud_Rate_By_Minute'] = df['Minute'].map(fraud_rate_by_minute)

    # Define thresholds for high-risk periods (mean fraud rate).
    day_thresh = fraud_rate_by_day.mean()
    hour_thresh = fraud_rate_by_hour.mean()
    minute_thresh = fraud_rate_by_minute.mean()

    df['High_Fraud_Day'] = (df['Fraud_Rate_By_Day'] > day_thresh).astype(int)
    df['High_Fraud_Hour'] = (df['Fraud_Rate_By_Hour'] > hour_thresh).astype(int)
    df['High_Fraud_Minute'] = (df['Fraud_Rate_By_Minute'] > minute_thresh).astype(int)
    print("Time-based features generated.")

    # 5. Interaction Features (Frequency Encodings for pairs of categorical variables)
    df['Bank_Pair'] = df['From Bank'].astype(str) + '_' + df['To Bank'].astype(str)
    pair_freq = df['Bank_Pair'].value_counts(normalize=True)
    df['Bank_Pair_freq_enc'] = df['Bank_Pair'].map(pair_freq)

    df['Account_Pair'] = df['Account'].astype(str) + '_' + df['Account.1'].astype(str)
    account_pair_freq = df['Account_Pair'].value_counts(normalize=True)
    df['Account_Pair_freq_enc'] = df['Account_Pair'].map(account_pair_freq)

    df['PaymentFormat_Hour'] = df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    payment_hour_freq = df['PaymentFormat_Hour'].value_counts(normalize=True)
    df['PaymentFormat_Hour_freq_enc'] = df['PaymentFormat_Hour'].map(payment_hour_freq)

    df['Sender_PaymentFormat'] = df['Account'].astype(str) + '_' + df['Payment Format'].astype(str)
    sender_payment_freq = df['Sender_PaymentFormat'].value_counts(normalize=True)
    df['Sender_PaymentFormat_freq_enc'] = df['Sender_PaymentFormat'].map(sender_payment_freq)

    df['Day_Hour'] = df['Day'].astype(str) + '_' + df['Hour'].astype(str)
    day_hour_freq = df['Day_Hour'].value_counts(normalize=True)
    df['Day_Hour_freq_enc'] = df['Day_Hour'].map(day_hour_freq)

    df['Bank_Payment_Hour'] = df['From Bank'].astype(str) + '_' + df['Payment Format'].astype(str) + '_' + df['Hour'].astype(str)
    bank_pay_hour_freq = df['Bank_Payment_Hour'].value_counts(normalize=True)
    df['Bank_Payment_Hour_freq_enc'] = df['Bank_Payment_Hour'].map(bank_pay_hour_freq)
    print("Interaction features generated.")

    # 6. Behavioral Features (Rolling averages/stds/velocities)
    window_size = 5 # For rolling averages and std deviations
    df = df.sort_values('Timestamp') # Re-sort to ensure correct rolling calculations

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9 # Convert to seconds for velocity
    window_seconds = 3600 # 1 hour window for transaction velocity

    def txn_velocity_sender(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Sender'] = df.groupby('Account').apply(txn_velocity_sender).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore') # Drop temporary unix timestamp

    df['Rolling_Std_Amount_Received_Sender'] = df.groupby('Account')['Amount Received']\
                                                .rolling(window=window_size, min_periods=1)\
                                                .std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Sender'] = df.groupby('Account')['Amount Paid']\
                                            .rolling(window=window_size, min_periods=1)\
                                            .std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Received_Receiver'] = df.groupby('Account.1')['Amount Received']\
                                                  .rolling(window=window_size, min_periods=1)\
                                                  .std().reset_index(level=0, drop=True).fillna(0)

    df['Rolling_Std_Amount_Paid_Receiver'] = df.groupby('Account.1')['Amount Paid']\
                                              .rolling(window=window_size, min_periods=1)\
                                              .std().reset_index(level=0, drop=True).fillna(0)

    epsilon = 1e-6 # Small value to avoid division by zero in ratios
    
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

    window_size_txns = 20 # For rolling unique counterparties
    df['Rolling_Unique_Receivers_Sender'] = df.groupby('Account')['Account.1'].apply(lambda x: rolling_unique_set(x, window_size_txns)).reset_index(level=0, drop=True)

    df['Timestamp_unix'] = df['Timestamp'].astype('int64') // 10**9 # Re-create for receiver velocity
    def txn_velocity_receiver(group):
        times = group['Timestamp_unix'].values
        counts = []
        for i, t in enumerate(times):
            counts.append(((times >= t - window_seconds) & (times <= t)).sum())
        return pd.Series(counts, index=group.index)

    df['Txn_Velocity_Receiver'] = df.groupby('Account.1').apply(txn_velocity_receiver).reset_index(level=0, drop=True)
    df.drop(columns=['Timestamp_unix'], inplace=True, errors='ignore')
    print("Behavioral features generated.")


    # 7. Categorical Encoding (Frequency and Target Encoding, One-Hot Encoding)
    freq_target_cols = ['From Bank', 'To Bank']
    one_hot_cols = ['Receiving Currency', 'Payment Currency', 'Payment Format']

    # Frequency Encoding for specified columns
    for col in freq_target_cols:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)

    # Target Encoding for specified columns (mean fraud rate)
    for col in freq_target_cols:
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)

    # One-Hot Encoding for other categorical columns
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True) # drop_first avoids multicollinearity
    print("Categorical features encoded.")

    # Frequency Encoding for Account and Account.1
    for col in ['Account', 'Account.1']:
        freq_enc = df[col].value_counts(normalize=True)
        df[col + '_freq_enc'] = df[col].map(freq_enc)

    # Target Encoding for Account and Account.1
    for col in ['Account', 'Account.1']:
        target_enc = df.groupby(col)['Is Laundering'].mean()
        df[col + '_target_enc'] = df[col].map(target_enc)
    print("Account-level frequency and target encodings generated.")

    # Drop unnecessary columns that are not features for the model, or have been transformed
    drop_cols = [
        'Bank_Pair', 'Account_Pair', 'PaymentFormat_Hour', 'Sender_PaymentFormat', 'Day_Hour', 'Bank_Payment_Hour',
        'Amount Received', 'Amount Paid', # Original amount columns are no longer needed after log transform and ratios
    ]
    # Filter out columns that might not exist after one-hot encoding or previous drops
    df_model = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Save the list of final feature columns for consistency in production
    # Exclude target and identifier columns from this list.
    final_feature_columns = [col for col in df_model.columns if col not in ['Is Laundering', 'Timestamp', 'Account', 'Account.1', 'From Bank', 'To Bank',  'Day', 'Hour', 'Minute', 'Unnamed: 0']]
    with fsspec.open(feature_cols_output_path, 'w') as f:
        for col in final_feature_columns:
            f.write(col + '\n')
    print(f"Final feature column names saved to: {feature_cols_output_path}")

    # Save the processed feature engineered dataset
    df_model.to_csv(output_path, index=False)
    print(f"Feature engineered training data saved to: {output_path}. Final shape: {df_model.shape}")

    # --- Export Lookup Tables for Production Inference ---
    # These lookups capture information learned from the training data that needs
    # to be applied consistently to new (production) data.
    print("Exporting lookup tables for production features...")

    # Ensure all lookups are based on the final df_model to capture all changes
    df_for_lookups = df_model.copy() # Use df_model after drops/encodings

    # Fraud rates by time components
    fraud_rate_by_day.to_csv(lookup_dir + 'Fraud_Rate_By_Day_lookup.csv')
    fraud_rate_by_hour.to_csv(lookup_dir + 'Fraud_Rate_By_Hour_lookup.csv')
    fraud_rate_by_minute.to_csv(lookup_dir + 'Fraud_Rate_By_Minute_lookup.csv')

    # Account-based fraud rates
    df_for_lookups.groupby('Account')['sender_fraud_rate'].first().to_csv(lookup_dir + 'sender_fraud_rate_lookup.csv')
    df_for_lookups.groupby('Account.1')['receiver_fraud_rate'].first().to_csv(lookup_dir + 'receiver_fraud_rate_lookup.csv')

    # Target encodings for banks and accounts
    df_for_lookups.groupby('From Bank')['From Bank_target_enc'].first().to_csv(lookup_dir + 'From_Bank_target_enc_lookup.csv')
    df_for_lookups.groupby('To Bank')['To Bank_target_enc'].first().to_csv(lookup_dir + 'To_Bank_target_enc_lookup.csv')
    df_for_lookups.groupby('Account')['Account_target_enc'].first().to_csv(lookup_dir + 'Account_target_enc_lookup.csv')
    df_for_lookups.groupby('Account.1')['Account.1_target_enc'].first().to_csv(lookup_dir + 'Account.1_target_enc_lookup.csv')

    # Thresholds for High_Fraud_Day/Hour/Minute flags
    pd.Series({'day_thresh': day_thresh}).to_csv(lookup_dir + 'day_thresh_lookup.csv')
    pd.Series({'hour_thresh': hour_thresh}).to_csv(lookup_dir + 'hour_thresh_lookup.csv')
    pd.Series({'minute_thresh': minute_thresh}).to_csv(lookup_dir + 'minute_thresh_lookup.csv')

    print(f"All lookup tables exported to: {lookup_dir}")


if __name__ == "__main__":
    # This block allows the script to be run directly for training feature engineering.
    # It will create the 'lookups' directory if it doesn't exist.
    print("Running training feature engineering process...")
    # Ensure data_clean_split.py has been run to create 'transactions_train.csv'
    if not os.path.exists('data/transactions_train.csv'):
        print("Warning: 'data/transactions_train.csv' not found. Please run 'data_preprocessing/data_clean_split.py' first.")
    os.makedirs('lookups', exist_ok=True)
    generate_training_features()

