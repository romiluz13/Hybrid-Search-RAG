# Installation Guide

Complete guide to installing and setting up HybridRAG.

## System Requirements

### Hardware Requirements

**Minimum (Development/Testing):**
- **CPU**: 2+ cores (x86_64 or ARM64)
- **RAM**: 4 GB
- **Storage**: 5 GB free space
- **Network**: Stable internet connection (for API calls)

**Recommended (Production):**
- **CPU**: 4+ cores (x86_64 or ARM64)
- **RAM**: 8 GB or more
- **Storage**: 20 GB+ free space (for document storage and embeddings)
- **Network**: Stable, low-latency internet connection

**Note**: HybridRAG does **not** require a GPU. All embeddings and LLM inference are handled via API calls to:
- Voyage AI (embeddings & reranking)
- Anthropic Claude / OpenAI / Google Gemini (LLM generation)

### Software Requirements

- **Python**: 3.11 or higher (3.12 recommended)
- **Operating System**:
  - macOS 10.15+ (Catalina or later)
  - Linux (Ubuntu 20.04+, Debian 11+, or similar)
  - Windows 10/11 (with WSL2 recommended)
- **MongoDB**: MongoDB Community Edition (recommended for free tier) or MongoDB Atlas (M2+ for production)
- **API Keys**:
  - Voyage AI API key (required)
  - At least one LLM provider key (Anthropic, OpenAI, or Google Gemini)

## Installation Methods

### Method 1: Full Installation (Recommended)

Install all features including UI, CLI, API, and evaluation tools:

```bash
# Clone the repository
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all optional dependencies
pip install -e ".[all]"
```

### Method 2: Minimal Installation

Install only core functionality:

```bash
git clone https://github.com/romiluz13/Hybrid-Search-RAG.git
cd Hybrid-Search-RAG

python -m venv venv
source venv/bin/activate

# Install core only
pip install -e .
```

### Method 3: Selective Installation

Install specific feature sets:

```bash
# Core + UI (Chainlit)
pip install -e ".[ui]"

# Core + CLI
pip install -e ".[cli]"

# Core + API (FastAPI)
pip install -e ".[api]"

# Core + Document Processing
pip install -e ".[ingestion]"

# Core + Observability (Langfuse)
pip install -e ".[observability]"

# Core + Evaluation (RAGAS)
pip install -e ".[evaluation]"

# Combine multiple
pip install -e ".[ui,cli,api]"
```

## Configuration

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
cat > .env << EOF
# MongoDB Atlas Connection
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=hybridrag

# Voyage AI (Required)
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxx

# LLM Provider (Choose at least one)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx
# OR
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
# OR
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxx

# Optional: Observability
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
EOF
```

### 2. Set Up MongoDB

**Option A: MongoDB Community Edition (Recommended for Free Tier)**

MongoDB Community Edition includes native full-text search and vector search capabilities without index limitations:

1. Download and install [MongoDB Community Edition](https://www.mongodb.com/try/download/community) (7.0+)
2. Start MongoDB: `mongod --dbpath /path/to/data`
3. Create a database user (optional for local development)
4. Use connection string: `mongodb://localhost:27017` (or `mongodb://localhost:27017/hybridrag`)

**Option B: MongoDB Atlas (For Production)**

**Important**: Atlas M0 free tier has a **3 search/vector index limit** which prevents full hybrid search. For free tier users, use MongoDB Community Edition instead.

For production use:
1. Create an account at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create an **M10+ cluster** (M0 free tier not recommended - see [Issue #2](https://github.com/romiluz13/Hybrid-Search-RAG/issues/2))
3. Create a database user
4. Whitelist your IP address (or use `0.0.0.0/0` for development)
5. Get your connection string and add it to `.env`

**Important**: Enable Vector Search:
- **Community Edition**: Vector search is enabled by default in 7.0+
- **Atlas**: Go to Atlas Search → Create Search Index → Create a vector search index on your collection

### 3. Get API Keys

**Voyage AI** (Required):
1. Sign up at [Voyage AI](https://www.voyageai.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Add to `.env` as `VOYAGE_API_KEY`

**LLM Provider** (Choose one or more):
- **Anthropic**: [Get API key](https://console.anthropic.com/)
- **OpenAI**: [Get API key](https://platform.openai.com/api-keys)
- **Google Gemini**: [Get API key](https://makersuite.google.com/app/apikey)

## Verification

### Test Installation

```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify installation
python -c "import hybridrag; print('✓ HybridRAG installed')"

# Check dependencies
python -c "
import voyageai
import motor
import anthropic
print('✓ All core dependencies installed')
"
```

### Test MongoDB Connection

```bash
python -c "
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()
client = AsyncIOMotorClient(os.getenv('MONGODB_URI'))
print('✓ MongoDB connection successful')
"
```

### Test API Keys

```bash
python -c "
import os
from voyageai import Client
from dotenv import load_dotenv

load_dotenv()
client = Client(api_key=os.getenv('VOYAGE_API_KEY'))
print('✓ Voyage AI API key valid')
"
```

## Quick Start

After installation, launch the web UI:

```bash
chainlit run src/hybridrag/ui/chat.py
```

Then open `http://localhost:8000` in your browser.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` for optional dependencies
- **Solution**: Install the specific optional dependency group: `pip install -e ".[ui]"`

**Issue**: MongoDB connection timeout
- **Solution**: Check your IP is whitelisted in Atlas, verify connection string format

**Issue**: Voyage AI API errors
- **Solution**: Verify API key is correct and account has credits

**Issue**: Import errors after installation
- **Solution**: Ensure you're in the virtual environment: `source venv/bin/activate`

**Issue**: Chainlit UI not starting
- **Solution**: Install UI dependencies: `pip install -e ".[ui]"`

**Issue**: MongoDB Atlas M0 "maximum number of FTS indexes reached" error
- **Solution**: Atlas M0 free tier has a 3-index limit that prevents full hybrid search. Switch to MongoDB Community Edition for unlimited indexes on a free stack, or upgrade to Atlas M10+ for production use. See [Issue #2](https://github.com/romiluz13/Hybrid-Search-RAG/issues/2) for details.

### Getting Help

- Check [GitHub Issues](https://github.com/romiluz13/Hybrid-Search-RAG/issues)
- Review [Configuration Guide](configuration.md)
- See [API Reference](api.md)

## Next Steps

- [Configuration Guide](configuration.md) - Configure HybridRAG settings
- [Query Modes](query-modes.md) - Understand different query modes
- [API Reference](api.md) - Use the Python SDK
- [Deployment Guide](deployment.md) - Deploy to production
