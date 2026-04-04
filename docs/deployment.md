# Deployment Guide

Guide to deploying HybridRAG as a backend-first Python service.

## Pre-Deployment Checklist

- [ ] MongoDB Atlas cluster configured and accessible
- [ ] Vector search indexes created
- [ ] API keys secured (use secrets management)
- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Monitoring configured (Langfuse recommended)

## Deployment Options

## Blessed Deployment Shape

The publish-ready reference path is:

- MongoDB Atlas 8.2+ in cloud, or `mongodb/mongodb-atlas-local:preview` on `mongodb://localhost:27018/?directConnection=true` for deterministic local validation
- Voyage for embeddings and reranking
- OpenAI API or an OpenAI-compatible endpoint for generation
- FastAPI as the primary service surface

The release gate is designed to fail fast on this stack. Unsupported capabilities should error explicitly rather than silently degrading.

### Option 1: Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Expose port
EXPOSE 8000

# Run the FastAPI service
CMD ["uvicorn", "hybridrag.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  hybridrag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - MONGODB_DATABASE=${MONGODB_DATABASE}
      - VOYAGE_API_KEY=${VOYAGE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
      - OPENAI_EXTRA_HEADERS=${OPENAI_EXTRA_HEADERS}
    env_file:
      - .env
    restart: unless-stopped
```

#### Build and Run

```bash
# Build image
docker build -t hybridrag:latest .

# Run container
docker run -p 8000:8000 --env-file .env hybridrag:latest

# Or use docker-compose
docker-compose up -d
```

### Option 2: Cloud Platform Deployment

#### Railway

1. Connect your GitHub repository
2. Set environment variables in Railway dashboard
3. Railway auto-detects Python and installs dependencies
4. Deploy!

**railway.json:**
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn hybridrag.api.main:create_app --factory --host 0.0.0.0 --port $PORT"
  }
}
```

#### Render

1. Create new Web Service
2. Connect GitHub repository
3. Set build command: `pip install -e ".[all]"`
4. Set start command: `uvicorn hybridrag.api.main:create_app --factory --host 0.0.0.0 --port $PORT`
5. Add environment variables
6. Deploy!

#### Heroku

**Procfile:**
```
web: uvicorn hybridrag.api.main:create_app --factory --host 0.0.0.0 --port $PORT
```

**runtime.txt:**
```
python-3.11.7
```

Deploy:
```bash
heroku create your-app-name
heroku config:set MONGODB_URI=...
heroku config:set VOYAGE_API_KEY=...
git push heroku main
```

### Option 3: Local API Run

```bash
uvicorn hybridrag.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Environment Configuration

### Production Environment Variables

```bash
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net
MONGODB_DATABASE=hybridrag_prod

# Voyage AI
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx

# LLM
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-5.4
# Optional for OpenAI-compatible gateways:
# OPENAI_BASE_URL=https://your-endpoint/v1
# OPENAI_EXTRA_HEADERS={"api-key":"..."}

# Observability
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Security
HYBRIDRAG_API_KEY=your-api-key-here
CORS_ORIGINS=https://yourdomain.com
HYBRIDRAG_RATE_LIMIT_PER_WINDOW=60
HYBRIDRAG_RATE_LIMIT_WINDOW_SECONDS=60

# Performance
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
```

### Secrets Management

**Never commit secrets!** Use:

- **AWS Secrets Manager** (AWS deployments)
- **Google Secret Manager** (GCP deployments)
- **Azure Key Vault** (Azure deployments)
- **HashiCorp Vault** (self-hosted)
- **Environment variables** (platform-provided)

## MongoDB Atlas Setup

### 1. Create Production Cluster

1. Log into MongoDB Atlas
2. Create M10+ cluster (M0 free tier has 3-index limit - use MongoDB Community Edition for free tier instead)
3. Configure backup and monitoring

### 2. Configure Network Access

- Whitelist deployment server IPs
- Or use VPC peering for private networks

### 3. Create Vector Search Index

```javascript
// In Atlas UI: Search → Create Search Index → JSON Editor
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
}
```

### 4. Create Text Search Index

```javascript
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": {
        "type": "string"
      }
    }
  }
}
```

## Release Validation

Before publish, the blessed stack should pass:

```bash
make release-gate-fast
make release-gate-live
```

`release-gate-live` is designed for real seeded validation against `atlas-local:preview`, Voyage, and an OpenAI-compatible generation endpoint.

## Monitoring & Observability

### Langfuse Integration

HybridRAG automatically sends traces to Langfuse if configured:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Monitor:
- Query latency
- Token usage
- Error rates
- User sessions

### Application Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Checks

```python
@app.get("/health")
async def health():
    try:
        await rag.get_stats()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Performance Optimization

### 1. Connection Pooling

MongoDB connection pooling is automatic with Motor. Configure pool size:

```python
settings = Settings(
    mongodb_uri="mongodb+srv://...",
    mongodb_max_pool_size=50,
)
```

### 2. Caching

Enable caching for repeated queries:

```python
settings = Settings(
    enable_query_cache=True,
    cache_ttl_seconds=3600,
)
```

### 3. Async Processing

Use async/await throughout:

```python
# Good: Concurrent requests
results = await asyncio.gather(*[
    rag.query(q) for q in queries
])
```

### 4. Batch Operations

Batch embeddings and database writes:

```python
settings = Settings(
    embedding_batch_size=32,
    write_batch_size=100,
)
```

## Security

### 1. API Key Security

- Never commit API keys
- Rotate keys regularly
- Use least-privilege access

### 2. Input Validation

```python
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    query: str
    session_id: str

    @validator('query')
    def validate_query(cls, v):
        if len(v) > 10000:
            raise ValueError('Query too long')
        return v
```

### 3. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, ...):
    ...
```

### 4. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Scaling

### Horizontal Scaling

Run multiple instances behind a load balancer:

```bash
# Instance 1
chainlit run src/hybridrag/ui/chat.py --port 8000

# Instance 2
chainlit run src/hybridrag/ui/chat.py --port 8001

# Load balancer (nginx)
upstream hybridrag {
    server localhost:8000;
    server localhost:8001;
}
```

### Vertical Scaling

Increase MongoDB cluster size:
- M10 → M20 → M30 for more throughput
- Enable auto-scaling in Atlas

## Backup & Recovery

### MongoDB Backups

1. Enable continuous backups in Atlas
2. Set retention policy (30+ days recommended)
3. Test restore procedures

### Application State

HybridRAG stores all state in MongoDB, so backups cover:
- Documents
- Embeddings
- Knowledge graph
- Conversation memory

## Troubleshooting

### Common Issues

**Issue**: Slow queries
- **Solution**: Check MongoDB indexes, increase cluster size

**Issue**: High API costs
- **Solution**: Enable caching, optimize query modes

**Issue**: Memory errors
- **Solution**: Reduce batch sizes, increase server RAM

**Issue**: Connection timeouts
- **Solution**: Check network, increase timeout settings

## Next Steps

- [Installation Guide](installation.md) - Set up HybridRAG
- [Configuration Guide](configuration.md) - Configure settings
- [API Reference](api.md) - Use the Python SDK
