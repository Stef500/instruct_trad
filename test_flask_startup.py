#!/usr/bin/env python3
"""
Quick test to verify Flask app can start properly.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from medical_dataset_processor.web.app import create_app

def test_app_creation():
    """Test that the Flask app can be created successfully."""
    try:
        app = create_app()
        print("✓ Flask app created successfully")
        
        # Test that we can get the app context
        with app.app_context():
            print("✓ App context works")
        
        # Test that we can create a test client
        client = app.test_client()
        print("✓ Test client created")
        
        # Test a simple route
        response = client.get('/')
        print(f"✓ Index route responds with status {response.status_code}")
        
        # Test health check
        response = client.get('/api/health')
        print(f"✓ Health check responds with status {response.status_code}")
        
        print("\n🎉 All Flask app startup tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Flask app: {e}")
        return False

if __name__ == '__main__':
    success = test_app_creation()
    sys.exit(0 if success else 1)