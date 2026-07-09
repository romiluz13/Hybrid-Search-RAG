#!/bin/bash
#
# HybridRAG Setup Script
# World-class RAG with MongoDB Atlas + Voyage AI
#
# Usage:
#   ./setup.sh           # Full setup
#   ./setup.sh --quick   # Quick install (skip optional deps)
#   ./setup.sh --help    # Show help
#
# Requirements:
#   - Python 3.11+
#   - pip
#   - (optional) MongoDB Atlas account
#   - (optional) Voyage AI API key
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
MIN_PYTHON_VERSION="3.11"
INSTALL_EXTRAS="all"

#---------------------------------------------------------------------------
# Helper Functions
#---------------------------------------------------------------------------

print_banner() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                                               ║${NC}"
    echo -e "${BLUE}║   ${GREEN}HybridRAG${BLUE}                                                  ║${NC}"
    echo -e "${BLUE}║   State-of-the-art RAG with MongoDB Atlas + Voyage AI        ║${NC}"
    echo -e "${BLUE}║                                                               ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick       Quick install with core dependencies only"
    echo "  --dev         Install with development tools"
    echo "  --all         Install all dependencies (default)"
    echo "  --no-venv     Skip virtual environment creation"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh              # Full setup with all dependencies"
    echo "  ./setup.sh --quick      # Quick setup for production"
    echo "  ./setup.sh --dev        # Setup for development"
    echo ""
}

check_python_version() {
    log_info "Checking Python version..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python ${MIN_PYTHON_VERSION}+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        log_success "Python ${PYTHON_VERSION} detected"
    else
        log_error "Python ${MIN_PYTHON_VERSION}+ required. Found: ${PYTHON_VERSION}"
        exit 1
    fi
}

create_venv() {
    if [ "$SKIP_VENV" = true ]; then
        log_warning "Skipping virtual environment creation"
        return
    fi

    if [ -d "$VENV_DIR" ]; then
        log_info "Virtual environment already exists"
    else
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    fi

    # Activate
    source "$VENV_DIR/bin/activate"
    log_info "Virtual environment activated"
}

install_dependencies() {
    log_info "Upgrading pip..."
    pip install --upgrade pip --quiet

    log_info "Installing HybridRAG with [$INSTALL_EXTRAS] dependencies..."

    if [ "$INSTALL_EXTRAS" = "core" ]; then
        pip install -e . --quiet
    else
        pip install -e ".[$INSTALL_EXTRAS]" --quiet
    fi

    log_success "Dependencies installed"
}

setup_env_file() {
    if [ -f ".env" ]; then
        log_info ".env file already exists"
    else
        if [ -f ".env.example" ]; then
            log_info "Creating .env from .env.example..."
            cp .env.example .env
            log_success ".env file created"
        else
            log_warning ".env.example not found, skipping .env creation"
        fi
    fi
}

verify_installation() {
    log_info "Verifying installation..."

    # Test imports
    if python3 -c "import hybridrag" 2>/dev/null; then
        VERSION=$(python3 -c "import hybridrag; print(hybridrag.__version__)")
        log_success "HybridRAG v${VERSION} installed successfully"
    else
        log_error "Installation verification failed"
        exit 1
    fi

    # Test core imports
    if python3 -c "from hybridrag import create_hybridrag, SYSTEM_PROMPT, detect_query_type" 2>/dev/null; then
        log_success "Core imports verified"
    else
        log_warning "Some imports may have issues"
    fi
}

print_next_steps() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete!                            ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo ""
    echo "  1. ${YELLOW}Configure API Keys${NC}"
    echo "     Edit .env and add your API keys:"
    echo "     - MONGODB_URI (MongoDB Atlas connection string)"
    echo "     - VOYAGE_API_KEY (from https://dash.voyageai.com/)"
    echo "     - LLM API key (Anthropic, OpenAI, or Gemini)"
    echo ""
    echo "  2. ${YELLOW}Activate Environment${NC}"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  3. ${YELLOW}Verify Setup${NC}"
    echo "     make dev"
    echo ""
    echo "  4. ${YELLOW}Run Tests${NC}"
    echo "     make test"
    echo ""
    echo "  5. ${YELLOW}Start Building${NC}"
    echo "     - make run-api     # Start API server"
    echo "     - make run-ui      # Start chat UI"
    echo "     - make run-cli     # Run CLI"
    echo ""
    echo -e "${BLUE}Documentation:${NC} https://github.com/romiluz13/Hybrid-Search-RAG"
    echo ""
}

#---------------------------------------------------------------------------
# Parse Arguments
#---------------------------------------------------------------------------

SKIP_VENV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            INSTALL_EXTRAS="core"
            shift
            ;;
        --dev)
            INSTALL_EXTRAS="dev"
            shift
            ;;
        --all)
            INSTALL_EXTRAS="all"
            shift
            ;;
        --no-venv)
            SKIP_VENV=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

#---------------------------------------------------------------------------
# Main
#---------------------------------------------------------------------------

main() {
    print_banner

    check_python_version
    create_venv
    install_dependencies
    setup_env_file
    verify_installation

    print_next_steps
}

# Run
main
