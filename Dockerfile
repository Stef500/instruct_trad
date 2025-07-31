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
COPY examples/ ./examples/
COPY test_flask_startup.py ./

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