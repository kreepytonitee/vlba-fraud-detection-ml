import requests
import json

BASE_URL = "https://fraud-detection-service-429587376884.europe-west10.run.app"

def test_with_real_data():
    # Step 1: Get real transaction
    print("Getting real transaction from dataset...")
    response = requests.get(f"{BASE_URL}/get-next-transaction")
    
    if response.status_code == 200:
        transaction_data = response.json()["transaction"]
        print(f"Got transaction: {transaction_data}")
        
        # Step 2: Use it for prediction
        print("\nMaking prediction...")
        pred_response = requests.post(f"{BASE_URL}/predict", json=transaction_data)
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            print(f"Prediction Result:")
            print(f"  Class: {result['prediction_class']}")
            print(f"  Probabilities: {result['prediction_probabilities']}")
            print(f"  Latency: {result['latency_ms']} ms")
            print(f"  True Label: {result.get('true_label', 'Unknown')}")
        else:
            print(f"Prediction failed: {pred_response.status_code} - {pred_response.text}")
    else:
        print(f"Failed to get transaction: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_with_real_data()