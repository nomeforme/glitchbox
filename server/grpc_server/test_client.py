#!/usr/bin/env python
"""
Simple test client for the gRPC generation control server.

Usage:
    python -m grpc_server.test_client [--host HOST] [--port PORT]

Example:
    python -m grpc_server.test_client --host localhost --port 50051
"""

import argparse
import grpc
from . import generation_control_pb2 as pb2
from . import generation_control_pb2_grpc as pb2_grpc


def test_set_prompt(stub, prompt: str):
    """Test SetPrompt RPC."""
    print(f"\n--- Testing SetPrompt ---")
    request = pb2.PromptRequest(
        prompt=prompt,
        target_prompt=f"target of {prompt}",
        prompt_travel_factor=0.5
    )
    response = stub.SetPrompt(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def test_set_generation_params(stub):
    """Test SetGenerationParams RPC."""
    print(f"\n--- Testing SetGenerationParams ---")
    request = pb2.GenerationParamsRequest(
        seed=42,
        num_inference_steps=4,
        guidance_scale=1.5,
        strength=0.6
    )
    response = stub.SetGenerationParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def test_set_controlnet_params(stub):
    """Test SetControlNetParams RPC."""
    print(f"\n--- Testing SetControlNetParams ---")
    request = pb2.ControlNetParamsRequest(
        controlnet_scale=0.8,
        controlnet_start=0.0,
        controlnet_end=0.9
    )
    response = stub.SetControlNetParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def test_set_lora_params(stub):
    """Test SetLoRAParams RPC."""
    print(f"\n--- Testing SetLoRAParams ---")
    request = pb2.LoRAParamsRequest(
        lora_scale=0.7,
        pipe_index=1
    )
    response = stub.SetLoRAParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def test_set_acid_params(stub):
    """Test SetAcidParams RPC."""
    print(f"\n--- Testing SetAcidParams ---")
    request = pb2.AcidParamsRequest(
        acid_strength=0.5,
        zoom_factor=1.2,
        x_shift=10,
        y_shift=-5
    )
    response = stub.SetAcidParams(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def test_get_current_state(stub):
    """Test GetCurrentState RPC."""
    print(f"\n--- Testing GetCurrentState ---")
    request = pb2.GetStateRequest()
    response = stub.GetCurrentState(request)
    print(f"Success: {response.success}")
    print(f"Prompt: {response.prompt}")
    print(f"Seed: {response.seed}")
    print(f"Guidance Scale: {response.guidance_scale}")
    print(f"Strength: {response.strength}")
    print(f"Pipe Index: {response.pipe_index}")
    print(f"Curation Index: {response.curation_index}")
    return response.success


def test_batch_update(stub):
    """Test BatchUpdate RPC."""
    print(f"\n--- Testing BatchUpdate ---")
    request = pb2.BatchUpdateRequest(
        prompt_params=pb2.PromptRequest(prompt="batch test prompt"),
        generation_params=pb2.GenerationParamsRequest(seed=123),
        acid_params=pb2.AcidParamsRequest(acid_strength=0.3)
    )
    response = stub.BatchUpdate(request)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    return response.success


def main():
    parser = argparse.ArgumentParser(description="Test gRPC generation control client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    args = parser.parse_args()

    address = f"{args.host}:{args.port}"
    print(f"Connecting to gRPC server at {address}...")

    try:
        channel = grpc.insecure_channel(address)
        stub = pb2_grpc.GenerationControlStub(channel)

        # Run all tests
        results = []
        results.append(("SetPrompt", test_set_prompt(stub, "a beautiful mountain landscape")))
        results.append(("SetGenerationParams", test_set_generation_params(stub)))
        results.append(("SetControlNetParams", test_set_controlnet_params(stub)))
        results.append(("SetLoRAParams", test_set_lora_params(stub)))
        results.append(("SetAcidParams", test_set_acid_params(stub)))
        results.append(("GetCurrentState", test_get_current_state(stub)))
        results.append(("BatchUpdate", test_batch_update(stub)))

        # Print summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        print("=" * 50)
        all_passed = True
        for name, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  {name}: {status}")
            if not success:
                all_passed = False

        print("=" * 50)
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
            return 1

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
