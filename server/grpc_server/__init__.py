"""
gRPC module for generation control.

This module provides a gRPC server for controlling real-time diffusion
generation parameters alongside the existing WebSocket interface.
"""

from .server import GRPCServer
from .service import GenerationControlServicer

__all__ = ['GRPCServer', 'GenerationControlServicer']
