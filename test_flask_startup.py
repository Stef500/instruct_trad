#!/usr/bin/env python3
"""
Simple Flask test app to verify Docker configuration without requiring valid API keys.
"""
import os
import sys
from pathlib import Path
from flask import Flask, jsonify

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def create_test_app():
    """Create a minimal Flask app for testing Docker configuration."""
    app = Flask(__name__)
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'message': 'Docker configuration test successful',
            'environment': {
                'DEEPL_API_KEY': 'configured' if os.environ.get('DEEPL_API_KEY') else 'missing',
                'SECRET_KEY': 'configured' if os.environ.get('SECRET_KEY') else 'missing',
                'TARGET_LANGUAGE': os.environ.get('TARGET_LANGUAGE', 'not set'),
                'WEB_HOST': os.environ.get('WEB_HOST', 'not set'),
                'WEB_PORT': os.environ.get('WEB_PORT', 'not set')
            }
        })
    
    @app.route('/')
    def index():
        """Simple index page."""
        return jsonify({
            'message': 'Medical Dataset Processor Docker Test',
            'status': 'running',
            'health_check': '/api/health'
        })
    
    return app


def main():
    """Main entry point for the test server."""
    host = os.environ.get('WEB_HOST', '0.0.0.0')
    port = int(os.environ.get('WEB_PORT', 5000))
    
    app = create_test_app()
    
    print(f"Starting Docker test server...")
    print(f"Server running on http://{host}:{port}")
    print(f"Health check available at http://{host}:{port}/api/health")
    
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()