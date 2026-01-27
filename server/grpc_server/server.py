"""
gRPC server lifecycle management.

This module provides the GRPCServer class that handles starting, stopping,
and managing the gRPC server alongside the main FastAPI application.
"""

import threading
import logging
from typing import Optional, Callable
from concurrent import futures

try:
    import grpc
    from . import generation_control_pb2_grpc as pb2_grpc
    from .service import GenerationControlServicer
    GRPC_AVAILABLE = True
except ImportError as e:
    GRPC_AVAILABLE = False
    grpc = None
    pb2_grpc = None
    GenerationControlServicer = None
    print(f"[gRPC] Warning: gRPC imports failed: {e}")
    print("[gRPC] Run: python -m grpc_tools.protoc -I grpc/protos --python_out=grpc --grpc_python_out=grpc grpc/protos/generation_control.proto")


class GRPCServer:
    """
    Manages the gRPC server lifecycle.

    This class handles starting the gRPC server in a background thread
    and provides methods for graceful shutdown.
    """

    def __init__(self,
                 port: int = 50051,
                 host: str = "0.0.0.0",
                 max_workers: int = 10,
                 on_curation_switch: Optional[Callable[[int], None]] = None,
                 debug: bool = False):
        """
        Initialize the gRPC server.

        Args:
            port: Port number to listen on (default: 50051)
            host: Host address to bind to (default: 0.0.0.0)
            max_workers: Maximum number of thread pool workers
            on_curation_switch: Callback for curation switching
            debug: Enable debug logging
        """
        self._port = port
        self._host = host
        self._max_workers = max_workers
        self._debug = debug
        self._server = None  # Optional grpc.Server
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._logger = logging.getLogger(__name__)

        # Create the servicer
        if GRPC_AVAILABLE:
            self._servicer = GenerationControlServicer(
                on_curation_switch=on_curation_switch,
                debug=debug
            )
        else:
            self._servicer = None

    def _log(self, message: str):
        """Log a message if debug is enabled."""
        if self._debug:
            self._logger.info(f"[gRPC Server] {message}")
        print(f"[gRPC] {message}")

    @property
    def servicer(self):
        """Get the servicer instance for accessing pending params."""
        return self._servicer

    def start(self):
        """
        Start the gRPC server in a background thread.

        The server will begin accepting connections immediately.
        """
        if not GRPC_AVAILABLE:
            self._log("gRPC not available - server not started")
            self._log("Run proto compilation first:")
            self._log("  cd server && python -m grpc_tools.protoc -I grpc/protos --python_out=grpc --grpc_python_out=grpc grpc/protos/generation_control.proto")
            return

        if self._running:
            self._log("Server already running")
            return

        self._log(f"Starting gRPC server on {self._host}:{self._port}")

        # Create the gRPC server
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        # Add the servicer
        pb2_grpc.add_GenerationControlServicer_to_server(
            self._servicer,
            self._server
        )

        # Bind to the address
        address = f"{self._host}:{self._port}"
        self._server.add_insecure_port(address)

        # Start the server
        self._server.start()
        self._running = True

        self._log(f"Server started on {address}")

    def stop(self, grace_period: float = 5.0):
        """
        Stop the gRPC server gracefully.

        Args:
            grace_period: Time in seconds to wait for pending RPCs to complete
        """
        if not self._running or self._server is None:
            self._log("Server not running")
            return

        self._log(f"Stopping server (grace period: {grace_period}s)...")

        # Stop accepting new RPCs and wait for existing ones to complete
        self._server.stop(grace_period)
        self._running = False

        self._log("Server stopped")

    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._running

    def get_pending_params(self):
        """
        Get and clear pending parameters from the servicer.

        Returns:
            Dictionary of pending parameter updates, or empty dict if unavailable
        """
        if self._servicer:
            return self._servicer.get_and_clear_pending_params()
        return {}

    def update_state(self, params: dict):
        """
        Update the servicer's current state.

        Args:
            params: Dictionary of parameter values to update
        """
        if self._servicer:
            self._servicer.update_current_state(params)
