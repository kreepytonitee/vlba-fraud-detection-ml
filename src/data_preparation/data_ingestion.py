import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage (assuming transactions.csv is in data/raw/)
    raw_data = load_data("https://storage.cloud.google.com/vlba-fd-data/raw/transactions.csv")
    if not raw_data.empty:
        print(raw_data.head())