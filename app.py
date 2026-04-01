#!/usr/bin/env python3
"""
HybridRAG Chat UI - Chainlit Entry Point

Run with:
    chainlit run app.py -w    # Development with hot reload
    chainlit run app.py       # Production
"""

import os
import sys

# Add project paths
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, "src"))

# Import and re-export Chainlit handlers
from src.hybridrag.ui.chat import on_chat_start, on_message  # noqa: E402

# These are automatically registered by Chainlit when it imports this module
__all__ = ["on_chat_start", "on_message"]
