#!/usr/bin/env python3
"""
Script to test the Crop Recommender API
"""

import requests
import json

API_URL = "http://localhost:5000/api"

def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test model info endpoint"""
    print("\n2. Testing Model Info...")
    response = requests.get(f"{API_URL}/model-info")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_soil_types():
    """Test soil types endpoint"""
    print("\n3. Testing Soil Types...")
    response = requests.get(f"{API_URL}/soil-types")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_crops():
    """Test crops endpoint"""
    print("\n4. Testing Available Crops...")
    response = requests.get(f"{API_URL}/crops")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_prediction():
    """Test prediction endpoint"""
    print("\n5. Testing Crop Prediction...")
    
    test_data = {
        "soil_type": "Loamy",
        "soil_ph": 6.5,
        "temperature": 25.0,
        "humidity": 80.0,
        "nitrogen": 70.0,
        "phosphorus": 60.0,
        "potassium": 50.0
    }
    
    print(f"   Input Data: {json.dumps(test_data, indent=2)}")
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def test_invalid_prediction():
    """Test prediction with invalid data"""
    print("\n6. Testing Invalid Prediction...")
    
    invalid_data = {
        "soil_type": "InvalidSoil",
        "soil_ph": 6.5,
        "temperature": 25.0,
        "humidity": 80.0,
        "nitrogen": 70.0,
        "phosphorus": 60.0,
        "potassium": 50.0
    }
    
    print(f"   Input Data: {json.dumps(invalid_data, indent=2)}")
    response = requests.post(
        f"{API_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")

def main():
    print("="*60)
    print("Testing Crop Recommender API")
    print("="*60)
    print("\nMake sure the API is running at http://localhost:5000")
    
    try:
        test_health()
        test_model_info()
        test_soil_types()
        test_crops()
        test_prediction()
        test_invalid_prediction()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API. Make sure it's running at http://localhost:5000")
    except Exception as e:
        print(f"\nError during testing: {e}")

if __name__ == '__main__':
    main()
