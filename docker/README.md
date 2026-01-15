# Docker Setup for ArXiv Classifier

This directory contains Docker configuration for containerizing the ArXiv Classifier system.

## Quick Start

### Development Mode (with hot reload)

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Production Mode

```bash
cd docker
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## Services

| Service  | Port | Description                     |
| -------- | ---- | ------------------------------- |
| `api`    | 8000 | Django REST API for predictions |
| `gradio` | 7860 | Interactive web demo            |

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│    Gradio Demo      │────▶│    Django API       │
│    (Port 7860)      │     │    (Port 8000)      │
└─────────────────────┘     └─────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │   Model Artifacts   │
                            │   (Volume Mount)    │
                            └─────────────────────┘
```

## Environment Variables

### API Service

| Variable        | Default                             | Description                                  |
| --------------- | ----------------------------------- | -------------------------------------------- |
| `MODEL_PATH`    | `/app/artifacts/test_scibert.keras` | Path to model file                           |
| `MODEL_TYPE`    | `scibert`                           | Model type: `baseline`, `scibert`, `specter` |
| `DEBUG`         | `false`                             | Django debug mode                            |
| `ALLOWED_HOSTS` | `localhost`                         | Comma-separated allowed hosts                |

### Gradio Service

| Variable             | Default           | Description         |
| -------------------- | ----------------- | ------------------- |
| `API_URL`            | `http://api:8000` | Internal API URL    |
| `GRADIO_SERVER_NAME` | `0.0.0.0`         | Gradio bind address |

## Building Individual Images

```bash
# Build API image
docker build -f docker/Dockerfile.api -t arxiv-api:latest .

# Build Gradio image
docker build -f docker/Dockerfile.gradio -t arxiv-gradio:latest .
```

## Useful Commands

```bash
# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f api

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild without cache
docker compose build --no-cache

# Check health status
docker compose ps
```

## Customizing Model

To use a different trained model:

1. Ensure the model file is in the `artifacts/` directory
2. Update the `MODEL_PATH` environment variable:

```yaml
# docker-compose.yml
environment:
  - MODEL_PATH=/app/artifacts/your_model.keras
  - MODEL_TYPE=baseline  # or scibert, specter
```

## Production Considerations

The `docker-compose.prod.yml` includes:

- **Gunicorn** instead of Django's development server
- **Resource limits** for memory management
- **Log rotation** to prevent disk fill
- **Health checks** for service monitoring

### Recommended Production Setup

```bash
# Use environment file for secrets
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  --env-file .env.prod up -d
```
