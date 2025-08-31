# Docker Usage Guide for Academic Assistant

## Overview
This guide explains how to use Docker to containerize and deploy the Academic Assistant API with both OpenAI SDK and Ollama (Mistral 7B) integration.

## Prerequisites
- Docker installed on your system
- Docker Compose installed
- (Optional) NVIDIA Docker runtime for GPU support with Ollama

## Quick Start with Docker Compose

### 1. Environment Setup
Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```bash
OPENAI_API_KEY=your_actual_openai_api_key
SECRET_KEY=your_secure_secret_key_at_least_32_characters
```

### 2. Start All Services
```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f acad-assistant
docker-compose logs -f ollama
```

### 3. Download Mistral 7B Model
After Ollama container is running:
```bash
# Connect to Ollama container and pull the model
docker-compose exec ollama ollama pull mistral:7b

# Verify model is downloaded
docker-compose exec ollama ollama list
```

### 4. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Root**: http://localhost:8000/

## Individual Container Management

### Build and Run API Only
```bash
# Build the image
docker build -t acad-assistant .

# Run with environment variables
docker run -d \
  --name acad-assistant \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e SECRET_KEY=your_secret \
  acad-assistant
```

### Run Ollama Separately
```bash
# Pull and run Ollama
docker pull ollama/ollama:latest
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama:latest

# Pull Mistral 7B model
docker exec ollama ollama pull mistral:7b
```

## Production Deployment

### 1. Update Environment Variables
For production, update your `.env` file:
```bash
DEBUG=False
LOG_LEVEL=WARNING
API_HOST=0.0.0.0
SECRET_KEY=your_very_secure_production_secret_key
```

### 2. Use Production Docker Compose
```bash
# Start in production mode
docker-compose -f docker-compose.yml up -d

# Scale the API service
docker-compose up -d --scale acad-assistant=3
```

### 3. Health Monitoring
```bash
# Check service health
docker-compose ps

# Monitor resource usage
docker stats

# Check service logs
docker-compose logs --tail=100 -f acad-assistant
```

## GPU Support for Ollama

If you have an NVIDIA GPU, uncomment the GPU section in `docker-compose.yml`:

```yaml
ollama:
  # ... other configuration
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Requirements:
- NVIDIA Docker runtime installed
- NVIDIA GPU with CUDA support

## Useful Docker Commands

### Container Management
```bash
# Stop all services
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# Restart specific service
docker-compose restart acad-assistant

# View container logs in real-time
docker-compose logs -f acad-assistant
```

### Development
```bash
# Rebuild and restart after code changes
docker-compose up -d --build acad-assistant

# Execute commands in running container
docker-compose exec acad-assistant bash

# Run tests in container
docker-compose exec acad-assistant pytest tests/
```

### Data Management
```bash
# Backup Ollama models
docker run --rm -v ollama_data:/source -v $(pwd):/backup alpine tar czf /backup/ollama_backup.tar.gz -C /source .

# Restore Ollama models
docker run --rm -v ollama_data:/target -v $(pwd):/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /target
```

## Troubleshooting

### Common Issues

1. **Ollama not responding**
   ```bash
   # Check if Ollama is running
   docker-compose ps ollama
   
   # Restart Ollama
   docker-compose restart ollama
   
   # Check Ollama logs
   docker-compose logs ollama
   ```

2. **API can't connect to Ollama**
   ```bash
   # Verify network connectivity
   docker-compose exec acad-assistant curl http://ollama:11434/api/tags
   
   # Check environment variables
   docker-compose exec acad-assistant env | grep OLLAMA
   ```

3. **Out of memory errors**
   ```bash
   # Increase Docker memory limits
   # Check Docker Desktop settings or Docker daemon configuration
   
   # Monitor memory usage
   docker stats
   ```

4. **Model not found**
   ```bash
   # List available models
   docker-compose exec ollama ollama list
   
   # Pull missing model
   docker-compose exec ollama ollama pull mistral:7b
   ```

### Performance Optimization

1. **For CPU-only deployment**:
   - Ensure sufficient RAM (8GB+ recommended)
   - Use SSD storage for better I/O performance

2. **For GPU deployment**:
   - Ensure NVIDIA drivers are up to date
   - Allocate sufficient GPU memory
   - Monitor GPU usage with `nvidia-smi`

### Security Considerations

1. **Environment Variables**:
   - Never commit `.env` files with real API keys
   - Use Docker secrets in production
   - Rotate API keys regularly

2. **Network Security**:
   - Use reverse proxy (nginx) in production
   - Enable HTTPS/TLS
   - Implement rate limiting

3. **Container Security**:
   - Run containers as non-root user (already configured)
   - Keep base images updated
   - Scan images for vulnerabilities

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t acad-assistant .
      - name: Run tests
        run: docker run --rm acad-assistant pytest tests/
```

### Docker Hub Deployment
```bash
# Tag and push to Docker Hub
docker tag acad-assistant your-username/acad-assistant:latest
docker push your-username/acad-assistant:latest
```

This completes the Docker setup and usage guide for the Academic Assistant API!
