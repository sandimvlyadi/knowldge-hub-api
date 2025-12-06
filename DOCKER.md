# Docker Deployment Guide

## 🐳 Quick Start

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- `.env` file dengan `OPENAI_API_KEY` (opsional)

## 📋 Deployment Options

### Option 1: Docker Compose (Recommended)

Cara termudah untuk deploy dengan semua konfigurasi sudah siap.

#### First Time Setup

```bash
# 1. Train models terlebih dahulu
docker-compose --profile training up trainer

# 2. Start API service
docker-compose up -d

# 3. Check logs
docker-compose logs -f api
```

#### Subsequent Runs

```bash
# Start service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop service
docker-compose down
```

### Option 2: Docker Manual Build

Untuk control lebih granular.

#### Build Image

```bash
docker build -t knowledge-hub-api:latest .
```

#### Train Models (First Time Only)

```bash
docker run --rm \
  -v $(pwd)/trained_models:/app/trained_models \
  -v $(pwd)/graphs.json:/app/graphs.json \
  knowledge-hub-api:latest \
  python train_models.py
```

#### Run API

```bash
docker run -d \
  --name knowledge-hub-api \
  -p 8000:8000 \
  -v $(pwd)/trained_models:/app/trained_models \
  --env-file .env \
  knowledge-hub-api:latest
```

#### Manage Container

```bash
# View logs
docker logs -f knowledge-hub-api

# Stop container
docker stop knowledge-hub-api

# Start container
docker start knowledge-hub-api

# Remove container
docker rm knowledge-hub-api

# Execute command inside container
docker exec -it knowledge-hub-api bash
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
TOKENIZERS_PARALLELISM=false
```

### Volume Mounts

Docker Compose automatically mounts:
- `./trained_models` → Models persistence
- `./graphs.json` → Training data
- `./features.json` → Reference data

## 🚀 Production Deployment

### Multi-stage Build (Optimized)

Create `Dockerfile.prod`:
```dockerfile
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*
COPY app/ ./app/
COPY trained_models/ ./trained_models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### With Nginx Reverse Proxy

Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - backend
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    networks:
      - backend
    restart: always

networks:
  backend:
    driver: bridge
```

## 📊 Monitoring

### Health Check

```bash
# Container health status
docker ps

# API health check
curl http://localhost:8000/
```

### Resource Usage

```bash
# Real-time stats
docker stats knowledge-hub-api

# Memory and CPU limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          memory: 2G
```

## 🐛 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api

# Check if models exist
docker exec knowledge-hub-api ls -la /app/trained_models/
```

### Out of Memory

Increase Docker memory limit or add swap:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
```

### Port already in use

```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
docker-compose run -p 8001:8000 api
```

### Models not found

```bash
# Retrain models
docker-compose --profile training up trainer

# Or manually
docker exec knowledge-hub-api python train_models.py
```

## 🔐 Security Best Practices

1. **Never commit `.env` file**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use secrets for sensitive data**
   ```yaml
   services:
     api:
       secrets:
         - openai_key
   
   secrets:
     openai_key:
       file: ./secrets/openai_key.txt
   ```

3. **Run as non-root user**
   Add to Dockerfile:
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

4. **Scan for vulnerabilities**
   ```bash
   docker scan knowledge-hub-api
   ```

## 📈 Scaling

### Horizontal Scaling with Docker Compose

```bash
# Scale to 3 instances
docker-compose up -d --scale api=3

# Add load balancer (nginx/traefik)
```

### With Kubernetes

```bash
# Build and push to registry
docker tag knowledge-hub-api registry.example.com/knowledge-hub-api
docker push registry.example.com/knowledge-hub-api

# Deploy to k8s
kubectl apply -f k8s/deployment.yaml
```

## 📝 Maintenance

### Update Image

```bash
# Pull latest code
git pull

# Rebuild
docker-compose build

# Restart with new image
docker-compose up -d
```

### Backup Models

```bash
# Backup trained models
docker cp knowledge-hub-api:/app/trained_models ./backup/

# Or use volume backup
docker run --rm -v knowledge-hub-api_trained_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz -C /data .
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Full cleanup
docker system prune -a --volumes
```
