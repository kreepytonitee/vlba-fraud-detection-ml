import pandas as pd

class InMemoryFeatureStore:
    def __init__(self):
        self.features = {} # Stores features keyed by a unique identifier (e.g., transaction_id)

    def add_features(self, transaction_id: str, features: dict):
        """Adds or updates features for a given transaction."""
        self.features[transaction_id] = features
        print(f"Features for transaction {transaction_id} added/updated.")

    def get_features(self, transaction_id: str) -> dict:
        """Retrieves features for a given transaction."""
        features = self.features.get(transaction_id)
        if features is None:
            raise KeyError(f"Transaction ID {transaction_id} not found")
        return features

    def get_all_features_as_dataframe(self) -> pd.DataFrame:
        """Returns all stored features as a Pandas DataFrame."""
        if not self.features:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.features, orient='index')

if __name__ == "__main__":
    feature_store = InMemoryFeatureStore()

    # Simulate some feature engineering for a transaction
    transaction_1_data = {
        'amount_sent': 100, 'amount_received': 90, 'date_time': pd.to_datetime('2025-01-01 10:00:00'),
        'currency_sent': 'USD', 'currency_received': 'USD', 'bank_sender': 'BankA', 'bank_receiver': 'BankX'
    }
    from src.features.feature_engineering import apply_feature_engineering
    transaction_1_df = pd.DataFrame([transaction_1_data])
    engineered_features_1 = apply_feature_engineering(transaction_1_df).iloc[0].drop(columns=['date_time', 'currency_sent', 'currency_received', 'bank_sender', 'bank_receiver', 'is_laundering'], errors='ignore').to_dict()
    feature_store.add_features("txn_12345", engineered_features_1)

    transaction_2_data = {
        'amount_sent': 5000, 'amount_received': 4800, 'date_time': pd.to_datetime('2025-01-01 10:05:00'),
        'currency_sent': 'USD', 'currency_received': 'USD', 'bank_sender': 'BankB', 'bank_receiver': 'BankY'
    }
    transaction_2_df = pd.DataFrame([transaction_2_data])
    engineered_features_2 = apply_feature_engineering(transaction_2_df).iloc[0].drop(columns=['date_time', 'currency_sent', 'currency_received', 'bank_sender', 'bank_receiver', 'is_laundering'], errors='ignore').to_dict()
    feature_store.add_features("txn_67890", engineered_features_2)

    print("\nRetrieving features for txn_12345:")
    print(feature_store.get_features("txn_12345"))

    print("\nAll features in store:")
    print(feature_store.get_all_features_as_dataframe())