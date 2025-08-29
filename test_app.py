#!/usr/bin/env python3
"""
Test script for Flask app with new tokenizer
"""

import requests
import json

def test_text_prediction():
    """Test the text prediction endpoint"""
    print("🧪 Testing text prediction API...")
    
    url = "http://127.0.0.1:10000/predict"
    data = {
        "plot": "A young boy discovers he has magical powers and goes on an adventure."
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Genres: {result['genres']}")
            return True
        else:
            print("❌ Failed!")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_text_prediction()
