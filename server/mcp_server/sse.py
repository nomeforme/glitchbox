"""Streamable HTTP transport mount for embedding the MCP server in a FastAPI/Starlette app."""

from starlette.applications import Starlette
from starlette.routing import Mount

from .server import mcp


def mount_mcp_http(app: Starlette) -> None:
    """Mount the MCP Streamable HTTP transport on a FastAPI/Starlette app at /mcp/.

    Endpoint:
        POST /mcp/mcp  -- Streamable HTTP endpoint (client connects here)

    The session manager must be started separately via
    ``mcp.session_manager.run()`` during the application lifespan.
    """
    mcp.settings.streamable_http_path = "/"
    app.mount("/mcp", Mount(path="", app=mcp.streamable_http_app()))
