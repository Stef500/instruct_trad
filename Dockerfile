# Dockerfile for Medical Dataset Processor Flask Web Application
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies in system Python (no virtual env in Docker)
RUN uv pip install --system --no-cache-dir flask flask-cors python-dotenv deepl openai pyyaml reportlab datasets click rich pytest pytest-cov

# Copy source code
COPY src/ ./src/
COPY datasets.yaml ./

# Create minimal Flask test server used by tests/test_docker.py
RUN set -e; \
    cat > /app/test_flask_startup.py <<'PY'
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

    @app.route("/api/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "message": "Docker configuration test successful",
            "environment": {
                "DEEPL_API_KEY": "configured" if os.environ.get("DEEPL_API_KEY") else "missing",
                "SECRET_KEY": "configured" if os.environ.get("SECRET_KEY") else "missing",
                "TARGET_LANGUAGE": os.environ.get("TARGET_LANGUAGE", "not set"),
                "WEB_HOST": os.environ.get("WEB_HOST", "not set"),
                "WEB_PORT": os.environ.get("WEB_PORT", "not set")
            }
        })

    @app.route("/")
    def index():
        return jsonify({
            "message": "Medical Dataset Processor Docker Test",
            "status": "running",
            "health_check": "/api/health"
        })

    return app


def main():
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_PORT", 5000))
    app = create_test_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
PY
RUN chmod +x /app/test_flask_startup.py

# Create necessary directories
RUN mkdir -p data/sessions logs output

# Set environment variables
ENV PYTHONPATH=/app/src
ENV FLASK_APP=medical_dataset_processor.web.app:create_app
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the Flask application using our custom startup script
CMD ["python", "src/medical_dataset_processor/web/run_server.py"]