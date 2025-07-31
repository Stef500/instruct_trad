# Docker Deployment Guide - Medical Dataset Processor Web Interface

This guide explains how to deploy the Medical Dataset Processor web interface using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed
- Valid API keys for DeepL (and optionally OpenAI)

## Quick Start

1. **Clone the repository and navigate to the project directory**

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```bash
   DEEPL_API_KEY=your_actual_deepl_api_key_here
   OPENAI_API_KEY=your_actual_openai_api_key_here
   SECRET_KEY=your-secure-secret-key-for-flask-sessions
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the web interface**
   - Open your browser and go to: http://localhost:5000
   - Health check endpoint: http://localhost:5000/api/health

5. **Stop the application**
   ```bash
   docker-compose down
   ```

## Configuration Options

### Environment Variables

The following environment variables can be configured in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPL_API_KEY` | DeepL API key (required) | - |
| `OPENAI_API_KEY` | OpenAI API key (optional) | - |
| `SECRET_KEY` | Flask session secret key | `dev-secret-key-change-in-production` |
| `TARGET_LANGUAGE` | Target language for translation | `FR` |
| `WEB_HOST` | Web server host | `0.0.0.0` |
| `WEB_PORT` | Web server port | `5000` |
| `SESSION_TIMEOUT` | Session timeout in seconds | `86400` (24 hours) |
| `AUTO_SAVE_INTERVAL` | Auto-save interval in seconds | `5` |
| `TRANSLATION_MAX_RETRIES` | Max retries for translation API | `3` |
| `TRANSLATION_BASE_DELAY` | Base delay for retries | `1.0` |
| `TRANSLATION_MAX_DELAY` | Max delay for retries | `60.0` |

### Port Configuration

By default, the application runs on port 5000. To change this:

1. Update the `WEB_PORT` environment variable in your `.env` file
2. Update the port mapping in `docker-compose.yml`:
   ```yaml
   ports:
     - "8080:5000"  # Maps host port 8080 to container port 5000
   ```

## Data Persistence

The Docker configuration includes volume mounts for:

- `./data` - Session data and temporary files
- `./output` - Generated output files
- `./logs` - Application logs

These directories will be created automatically and persist between container restarts.

## Testing the Deployment

Run the included test script to verify your Docker configuration:

```bash
python test_docker.py
```

This will test:
- Docker image build
- Docker Compose configuration
- Environment variable setup
- Container startup and health check

## Troubleshooting

### Container won't start

1. Check the logs:
   ```bash
   docker-compose logs medical-dataset-processor
   ```

2. Verify your environment variables:
   ```bash
   docker-compose config
   ```

### Invalid API key errors

- Ensure your `DEEPL_API_KEY` is valid and has sufficient quota
- Check the DeepL API documentation for key format requirements

### Port conflicts

If port 5000 is already in use:

1. Change the host port in `docker-compose.yml`:
   ```yaml
   ports:
     - "8080:5000"  # Use port 8080 instead
   ```

2. Access the application at http://localhost:8080

### Permission issues

If you encounter permission issues with mounted volumes:

```bash
sudo chown -R $USER:$USER data output logs
```

## Production Deployment

For production deployment:

1. **Use a strong secret key**:
   ```bash
   SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

2. **Set up proper logging**:
   - Configure log rotation
   - Monitor application logs

3. **Use a reverse proxy** (nginx, Apache) for SSL termination

4. **Monitor resource usage**:
   ```bash
   docker stats medical-dataset-processor-web
   ```

5. **Regular backups** of the `data` directory

## Health Monitoring

The application includes a health check endpoint at `/api/health` that returns:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-31T12:00:00",
  "version": "1.0.0"
}
```

You can use this endpoint for monitoring and load balancer health checks.

## Scaling

For high-traffic scenarios, you can run multiple instances:

```bash
docker-compose up -d --scale medical-dataset-processor=3
```

Note: You'll need to configure a load balancer (like nginx) to distribute traffic between instances.