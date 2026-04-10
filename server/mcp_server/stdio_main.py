#!/usr/bin/env python
"""Standalone stdio entry point for the Glitchbox MCP server.

Usage:
    cd server && uv run python -m mcp_server.stdio_main
"""

import sys
from pathlib import Path

# Ensure the server directory is on sys.path for grpc_server imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.server import mcp  # noqa: E402

if __name__ == "__main__":
    mcp.run(transport="stdio")
