#!/usr/bin/env python3
"""
Minimal gRPC client for changing prompts on the diffusion server.

Usage:
    python grpc_prompt_example.py "your prompt here"
    python grpc_prompt_example.py "source prompt" "target prompt" 0.5
"""

import sys
import grpc

# Add the server directory to path for imports
sys.path.insert(0, 'server')

from grpc_server import generation_control_pb2 as pb2
from grpc_server import generation_control_pb2_grpc as pb2_grpc


def set_prompt(host: str, port: int, prompt: str, target_prompt: str = None, factor: float = None):
    """Set the generation prompt via gRPC."""
    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = pb2_grpc.GenerationControlStub(channel)

    # Build request
    if target_prompt and factor is not None:
        request = pb2.PromptRequest(
            prompt=prompt,
            target_prompt=target_prompt,
            prompt_travel_factor=factor
        )
    else:
        request = pb2.PromptRequest(prompt=prompt)

    # Send request
    response = stub.SetPrompt(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response


def get_state(host: str, port: int):
    """Get current state from server."""
    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = pb2_grpc.GenerationControlStub(channel)

    response = stub.GetCurrentState(pb2.GetStateRequest())
    print(f"Current prompt: {response.prompt}")
    print(f"Target prompt: {response.target_prompt}")
    print(f"Travel factor: {response.prompt_travel_factor}")
    return response


if __name__ == "__main__":
    HOST = "localhost"  # Change to server IP
    PORT = 50051

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python grpc_prompt_example.py 'prompt'")
        print("  python grpc_prompt_example.py 'source' 'target' 0.5")
        print("  python grpc_prompt_example.py --status")
        sys.exit(1)

    if sys.argv[1] == "--status":
        get_state(HOST, PORT)
    elif len(sys.argv) == 2:
        set_prompt(HOST, PORT, sys.argv[1])
    elif len(sys.argv) >= 4:
        set_prompt(HOST, PORT, sys.argv[1], sys.argv[2], float(sys.argv[3]))
    else:
        set_prompt(HOST, PORT, sys.argv[1])
