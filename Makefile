# HybridRAG Makefile
# World-class developer experience for MongoDB Atlas RAG
#
# Quick Start:
#   make setup     - Initial project setup
#   make dev       - Start development mode
#   make test      - Run tests
#   make help      - Show all commands
#
# Reference: https://github.com/romiluz13/Hybrid-Search-RAG

.PHONY: help setup install install-dev install-all mongo-up mongo-down demo demo-full notebooks-setup first-time-setup dev test lint format clean build docker run-api run-ui example-smoke contract-tests release-gate-fast release-gate-live test-integration test-cov test-quick

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

#---------------------------------------------------------------------------
# Help
#---------------------------------------------------------------------------

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)HybridRAG$(NC) - State-of-the-art RAG with MongoDB Atlas + Voyage AI"
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make $(YELLOW)<target>$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Install:$(NC)"
	@grep -E '^(setup|install|install-dev|install-all|mongo-up|mongo-down|demo|demo-full|first-time-setup|notebooks-setup):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@grep -E '^(dev|run-api|run-ui|run-cli|notebooks):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@grep -E '^(test|test-cov|test-quick|test-integration|example-smoke|contract-tests|release-gate-fast|release-gate-live|benchmark|benchmark-save|lint|format|typecheck):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Build & Deploy:$(NC)"
	@grep -E '^(build|docker|clean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

#---------------------------------------------------------------------------
# Setup & Installation
#---------------------------------------------------------------------------

setup: ## Initial setup: create venv, install deps, copy env
	@echo "$(BLUE)Setting up HybridRAG...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[all]"
	@if [ ! -f ".env" ]; then \
		echo "$(GREEN)Copying .env.example to .env...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)Please edit .env with your API keys$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Edit $(YELLOW).env$(NC) with your API keys"
	@echo "  2. Run $(YELLOW)make dev$(NC) to verify installation"
	@echo "  3. Run $(YELLOW)make test$(NC) to run tests"
	@echo ""

install: ## Install core dependencies only
	@$(PIP) install -e .

install-dev: ## Install with development tools
	@$(PIP) install -e ".[dev]"

install-all: ## Install all dependencies (including optional)
	@$(PIP) install -e ".[all]"

mongo-up: ## Start local MongoDB (atlas-local:preview on mongodb://localhost:27018)
	@echo "$(BLUE)Starting local MongoDB (atlas-local:preview)...$(NC)"
	@docker compose -f docker/docker-compose.local.yml up -d
	@echo "$(GREEN)Waiting for MongoDB to be healthy...$(NC)"
	@for i in $$(seq 1 30); do \
		h=$$(docker inspect --format '{{.State.Health.Status}}' hybridrag-mongodb-atlas-local-preview 2>/dev/null); \
		if [ "$$h" = "healthy" ]; then echo "$(GREEN)MongoDB ready on mongodb://localhost:27018$(NC)"; break; fi; \
		sleep 2; \
	done

mongo-down: ## Stop local MongoDB
	@docker compose -f docker/docker-compose.local.yml down

demo: ## See MongoDB hybrid search in 60s (NO API keys; starts local MongoDB)
	@echo "$(BLUE)HybridRAG demo — no API keys required$(NC)"
	@make mongo-up
	@$(VENV)/bin/python scripts/demo.py

demo-full: ## Full generative RAG demo (requires VOYAGE_API_KEY + LLM key in .env)
	@echo "$(BLUE)HybridRAG full demo — requires VOYAGE_API_KEY + LLM key in .env$(NC)"
	@make mongo-up
	@$(VENV)/bin/python examples/01_quickstart.py

notebooks-setup: ## Install Jupyter Lab for the notebooks
	@$(PIP) install jupyterlab ipykernel

first-time-setup: ## Complete setup for new developers (ends with a working demo)
	@echo "$(BLUE)First-time HybridRAG setup...$(NC)"
	@echo ""
	@echo "$(GREEN)Step 1/4: Installing dependencies$(NC)"
	@./setup.sh --all || true
	@echo ""
	@echo "$(GREEN)Step 2/4: Installing pre-commit hooks$(NC)"
	@$(PIP) install pre-commit
	@$(VENV)/bin/pre-commit install || true
	@echo ""
	@echo "$(GREEN)Step 3/4: Starting local MongoDB$(NC)"
	@make mongo-up
	@echo ""
	@echo "$(GREEN)Step 4/4: Running the no-keys demo (see MongoDB value now)$(NC)"
	@$(VENV)/bin/python scripts/demo.py || echo "$(YELLOW)Demo needs Docker running: start Docker Desktop then 'make demo'$(NC)"
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. $(YELLOW)make demo$(NC)        — re-run the no-keys MongoDB demo"
	@echo "  2. Edit $(YELLOW).env$(NC)        — add VOYAGE_API_KEY + an LLM key"
	@echo "  3. $(YELLOW)make demo-full$(NC)   — full generative RAG (needs keys)"
	@echo "  4. $(YELLOW)make run-api$(NC)     — start the FastAPI server"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""

#---------------------------------------------------------------------------
# Development
#---------------------------------------------------------------------------

dev: ## Verify installation and show package info
	@echo "$(BLUE)HybridRAG Development Environment$(NC)"
	@echo ""
	@$(VENV)/bin/python -c "import hybridrag; print(f'Version: {hybridrag.__version__}')"
	@$(VENV)/bin/python -c "from hybridrag import create_hybridrag, SYSTEM_PROMPT, detect_query_type; print('Core imports: OK')"
	@$(VENV)/bin/python -c "from hybridrag.enhancements import build_vector_search_filters; print('Enhancements: OK')"
	@echo ""
	@echo "$(GREEN)All imports successful!$(NC)"

run-api: ## Start the FastAPI server
	@echo "$(BLUE)Starting HybridRAG API on http://localhost:8000$(NC)"
	@$(UVICORN) hybridrag.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui: ## Start the Chainlit UI
	@echo "$(BLUE)Starting HybridRAG Chat UI$(NC)"
	@$(VENV)/bin/chainlit run src/hybridrag/ui/chat.py --port 8001

run-cli: ## Run the HybridRAG CLI
	@$(VENV)/bin/hybridrag

notebooks: ## Start Jupyter Lab with examples
	@echo "$(BLUE)Starting Jupyter Lab...$(NC)"
	@echo "$(GREEN)Opening notebooks directory$(NC)"
	@$(VENV)/bin/jupyter lab notebooks/

#---------------------------------------------------------------------------
# Testing & Quality
#---------------------------------------------------------------------------

test: ## Run all tests (excludes MongoDB-backed integration tests)
	@echo "$(BLUE)Running tests...$(NC)"
	@$(PYTEST) tests/ -v -m "not integration" --ignore=tests/test_lightrag.py

test-cov: ## Run tests with coverage report (excludes integration tests)
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) tests/ -v -m "not integration" --cov=src/hybridrag --cov-report=html --ignore=tests/test_lightrag.py
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

test-quick: ## Run fast unit tests only (excludes integration tests)
	@$(PYTEST) tests/enhancements/ -v -m "not integration"

test-integration: ## Run MongoDB-backed integration tests (requires live MongoDB on localhost:27018)
	@echo "$(BLUE)Running integration tests (requires MongoDB on localhost:27018)...$(NC)"
	@$(PYTEST) tests/ -v -m "integration" --ignore=tests/test_lightrag.py

example-smoke: ## Run example and API contract smoke tests
	@$(PYTEST) tests/examples/ tests/api/test_query_api_features.py tests/core/test_reference_sources.py -v

contract-tests: ## Run canonical API and integration contract tests
	@$(PYTEST) tests/api/test_query_api_features.py tests/integration/test_full_rag_pipeline.py -v

release-gate-fast: ## Run the fast publish gate for the blessed stack
	@$(VENV)/bin/python -m compileall src/hybridrag tests
	@$(VENV)/bin/ruff check src/hybridrag tests
	@$(PYTEST) tests/api/test_query_api_features.py tests/core/ tests/examples/ tests/integration/test_full_rag_pipeline.py -v

release-gate-live: ## Run the deterministic seeded live gate with real providers
	@echo "$(BLUE)Using blessed local MongoDB stack: mongodb://localhost:27018/?directConnection=true$(NC)"
	@$(VENV)/bin/python tests/e2e_real_test.py

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	@$(PYTEST) tests/benchmarks/ -v -m benchmark --benchmark-only || echo "$(YELLOW)No benchmarks found yet$(NC)"

benchmark-save: ## Run and save benchmark baseline
	@echo "$(BLUE)Running and saving benchmark baseline...$(NC)"
	@$(PYTEST) tests/benchmarks/ -v -m benchmark --benchmark-save=baseline || echo "$(YELLOW)No benchmarks found yet$(NC)"
	@echo "$(GREEN)Baseline saved! Compare with: pytest --benchmark-compare$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	@$(VENV)/bin/ruff check src/hybridrag tests
	@echo "$(GREEN)Linting passed!$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(VENV)/bin/black src/hybridrag tests
	@$(VENV)/bin/isort src/hybridrag tests
	@echo "$(GREEN)Formatting complete!$(NC)"

typecheck: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@$(VENV)/bin/mypy src/hybridrag --ignore-missing-imports

#---------------------------------------------------------------------------
# Build & Deploy
#---------------------------------------------------------------------------

build: ## Build distribution packages
	@echo "$(BLUE)Building packages...$(NC)"
	@$(PIP) install build
	@$(VENV)/bin/python -m build
	@echo "$(GREEN)Build complete! Packages in dist/$(NC)"

docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t hybridrag:latest .

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf src/*.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf htmlcov/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

#---------------------------------------------------------------------------
# MongoDB Atlas Setup
#---------------------------------------------------------------------------

atlas-check: ## Check MongoDB connection
	@echo "$(BLUE)Checking MongoDB Atlas connection...$(NC)"
	@$(VENV)/bin/python -c "\
from hybridrag.config import get_settings; \
from pymongo import MongoClient; \
s = get_settings(); \
c = MongoClient(s.mongodb_uri.get_secret_value(), serverSelectionTimeoutMS=5000, connectTimeoutMS=5000); \
print(f'Connected to: {c.server_info()[\"version\"]}'); \
print(f'Database: {s.mongodb_database}')"

atlas-indexes: ## Show MongoDB Atlas index status
	@echo "$(BLUE)Checking Atlas Search indexes...$(NC)"
	@$(VENV)/bin/python -c "\
from hybridrag.config import get_settings; \
from pymongo import MongoClient; \
s = get_settings(); \
c = MongoClient(s.MONGODB_URI); \
db = c[s.MONGODB_DATABASE]; \
for coll in db.list_collection_names(): \
    print(f'\\n{coll}:'); \
    for idx in db[coll].list_indexes(): \
        print(f'  - {idx[\"name\"]}')"

#---------------------------------------------------------------------------
# Quick Commands
#---------------------------------------------------------------------------

audit: ## Run pip-audit for dependency vulnerabilities
	@echo "$(BLUE)Running pip-audit...$(NC)"
	@$(VENV)/bin/pip-audit || echo "$(YELLOW)pip-audit not installed. Run: pip install pip-audit$(NC)"

check: lint test ## Run linting and tests

ci: lint typecheck test ## Run full CI suite

reset: clean setup ## Clean and re-setup
