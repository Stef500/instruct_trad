#!/usr/bin/env python3
"""
Startup script for the Medical Dataset Processor Flask web application.
"""
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from medical_dataset_processor.web.app import create_app


def main():
    """Main entry point for the web server."""
    # Load environment variables
    host = os.environ.get('WEB_HOST', '0.0.0.0')
    port = int(os.environ.get('WEB_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    # Check required environment variables
    required_vars = ['DEEPL_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)
    
    # Create and run the Flask app
    try:
        app = create_app()
        
        print(f"Starting Medical Dataset Processor Web Interface...")
        print(f"Server running on http://{host}:{port}")
        print(f"Health check available at http://{host}:{port}/api/health")
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        print(f"Error starting application: {e}")
        if "Invalid DeepL API key" in str(e):
            print("Note: This error is expected when using test API keys.")
            print("Please provide a valid DEEPL_API_KEY for production use.")
        sys.exit(1)


if __name__ == '__main__':
    main()