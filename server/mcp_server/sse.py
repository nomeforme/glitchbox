"""SSE transport mount for embedding the MCP server in a FastAPI/Starlette app."""

from starlette.applications import Starlette
from starlette.routing import Mount

from .server import mcp


def mount_mcp_sse(app: Starlette) -> None:
    """Mount the MCP SSE transport on a FastAPI/Starlette app at /mcp/.

    Endpoints:
        GET  /mcp/sse        -- SSE event stream (client connects here)
        POST /mcp/messages/  -- client sends messages here
    """
    app.mount("/mcp", Mount(path="", app=mcp.sse_app()))
