"""MCP server module for Glitchbox generation control."""

from .server import mcp
from .sse import mount_mcp_sse

__all__ = ["mcp", "mount_mcp_sse"]
