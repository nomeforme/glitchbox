#!/usr/bin/env python3
"""
Standalone gRPC client for remote prompt control.
Copy this file + the generated pb2 files to the remote machine.

Required files on remote machine:
  - grpc_remote_client.py (this file)
  - generation_control_pb2.py
  - generation_control_pb2_grpc.py

Install: pip install grpcio grpcio-tools

Usage:
    python grpc_remote_client.py --status
    python grpc_remote_client.py "your prompt here"
    python grpc_remote_client.py "source prompt" "target prompt" 0.5
"""

import sys
import grpc

# ============================================
# CHANGE THIS TO YOUR SERVER IP
SERVER_IP = "192.168.10.130"
SERVER_PORT = 50051
# ============================================

import generation_control_pb2 as pb2
import generation_control_pb2_grpc as pb2_grpc


def connect():
    channel = grpc.insecure_channel(f'{SERVER_IP}:{SERVER_PORT}')
    return pb2_grpc.GenerationControlStub(channel)


def set_prompt(prompt: str, target_prompt: str = None, factor: float = None, transition_frames: int = None):
    """Set the generation prompt.

    Args:
        prompt: The new prompt text
        target_prompt: Optional target prompt for interpolation
        factor: Optional interpolation factor (0.0-1.0)
        transition_frames: Optional frames for smooth transition (0-100)
    """
    stub = connect()

    kwargs = {'prompt': prompt}

    if target_prompt is not None:
        kwargs['target_prompt'] = target_prompt
    if factor is not None:
        kwargs['prompt_travel_factor'] = factor
    if transition_frames is not None:
        kwargs['transition_frames'] = transition_frames

    request = pb2.PromptRequest(**kwargs)
    response = stub.SetPrompt(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")


def get_state():
    """Get current server state."""
    stub = connect()
    response = stub.GetCurrentState(pb2.GetStateRequest())
    print(f"Prompt: {response.prompt}")
    print(f"Target: {response.target_prompt}")
    print(f"Factor: {response.prompt_travel_factor}")
    print(f"Seed: {response.seed}")
    print(f"Size: {response.width}x{response.height}")


def set_acid_params(strength=None, zoom=None, x_shift=None, y_shift=None):
    """Set acid effect parameters."""
    stub = connect()

    kwargs = {}
    if strength is not None:
        kwargs['acid_strength'] = strength
    if zoom is not None:
        kwargs['zoom_factor'] = zoom
    if x_shift is not None:
        kwargs['x_shift'] = x_shift
    if y_shift is not None:
        kwargs['y_shift'] = y_shift

    request = pb2.AcidParamsRequest(**kwargs)
    response = stub.SetAcidParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")


def set_generation_params(seed=None, steps=None, guidance=None, strength=None,
                          temporal_coherence=None, temporal_coherence_latent=None):
    """Set generation parameters."""
    stub = connect()

    kwargs = {}
    if seed is not None:
        kwargs['seed'] = seed
    if steps is not None:
        kwargs['num_inference_steps'] = steps
    if guidance is not None:
        kwargs['guidance_scale'] = guidance
    if strength is not None:
        kwargs['strength'] = strength
    if temporal_coherence is not None:
        kwargs['temporal_coherence'] = temporal_coherence
    if temporal_coherence_latent is not None:
        kwargs['temporal_coherence_latent'] = temporal_coherence_latent

    request = pb2.GenerationParamsRequest(**kwargs)
    response = stub.SetGenerationParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Server: {SERVER_IP}:{SERVER_PORT}")
        print("\nUsage:")
        print("  python grpc_remote_client.py --status")
        print("  python grpc_remote_client.py 'prompt'")
        print("  python grpc_remote_client.py 'prompt' --transition 30")
        print("  python grpc_remote_client.py 'source' 'target' 0.5")
        sys.exit(1)

    if sys.argv[1] == "--status":
        get_state()
    elif "--transition" in sys.argv:
        # Handle transition mode: prompt --transition frames
        transition_idx = sys.argv.index("--transition")
        prompt = sys.argv[1]
        frames = int(sys.argv[transition_idx + 1]) if transition_idx + 1 < len(sys.argv) else 30
        set_prompt(prompt, transition_frames=frames)
    elif len(sys.argv) == 2:
        set_prompt(sys.argv[1])
    elif len(sys.argv) >= 4:
        set_prompt(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    else:
        set_prompt(sys.argv[1])
