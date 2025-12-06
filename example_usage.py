"""
Example usage of Knowledge Hub API
"""

import requests
import json

# API Base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test API health"""
    response = requests.get(f"{BASE_URL}/")
    print("Health Check:", response.json())


def test_recommendations():
    """Test method recommendations"""
    data = {
        "features": [
            {
                "key": "FEAT-1",
                "summary": "add 'stop' goal to the maven-activemq-plugin",
                "description": "This would be useful in at least two scenarios. In one of them, we have a hudson building our system, and somehow the shutdown hook never gets called.",
                "components": [],
                "project": "ActiveMQ Classic",
                "issuetype": "New Feature",
                "priority": "Major",
                "status": "Open",
                "reporter": "Test User",
                "methods": []
            }
        ],
        "top_n": 5,
        "blacklists": ["java.io", "java.util", "java.lang"]
    }
    
    response = requests.post(f"{BASE_URL}/api/recommendations", json=data)
    print("\n=== Recommendations ===")
    print(json.dumps(response.json(), indent=2))


def test_predict_priority():
    """Test priority prediction"""
    data = {
        "summary": "Database connection timeout",
        "description": "Application fails to connect to database after 30 seconds causing system downtime"
    }
    
    response = requests.post(f"{BASE_URL}/api/predict/priority", json=data)
    print("\n=== Priority Prediction ===")
    print(json.dumps(response.json(), indent=2))


def test_predict_issuetype():
    """Test issue type prediction"""
    data = {
        "summary": "Add REST API endpoint for user statistics",
        "description": "Need to expose user activity statistics via a new REST API endpoint"
    }
    
    response = requests.post(f"{BASE_URL}/api/predict/issuetype", json=data)
    print("\n=== Issue Type Prediction ===")
    print(json.dumps(response.json(), indent=2))


def test_predict_components():
    """Test component prediction"""
    data = {
        "summary": "Memory leak in caching layer",
        "description": "The cache grows indefinitely and causes out of memory errors after extended usage"
    }
    
    response = requests.post(f"{BASE_URL}/api/predict/components", json=data)
    print("\n=== Component Prediction ===")
    print(json.dumps(response.json(), indent=2))


def test_clustering():
    """Test clustering"""
    data = {
        "summary": "Performance issue in search functionality",
        "description": "Search queries take more than 5 seconds to complete, affecting user experience"
    }
    
    response = requests.post(f"{BASE_URL}/api/cluster", json=data)
    print("\n=== Clustering ===")
    print(json.dumps(response.json(), indent=2))


def test_chat():
    """Test RAG chatbot (requires OpenAI API key)"""
    data = {
        "query": "How can I fix ActiveMQ connection issues?",
        "top_k": 3
    }
    
    response = requests.post(f"{BASE_URL}/api/chat", json=data)
    print("\n=== Chat (RAG) ===")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Chat unavailable: {response.json()}")


if __name__ == "__main__":
    print("="*60)
    print("Knowledge Hub API - Example Usage")
    print("="*60)
    
    # Test all endpoints
    test_health_check()
    test_recommendations()
    test_predict_priority()
    test_predict_issuetype()
    test_predict_components()
    test_clustering()
    test_chat()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
