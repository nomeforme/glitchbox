#!/usr/bin/env python3
"""
Multi-prompt journey video recorder.

Connects to a remote StreamDiffusion server, sends a sequence of prompts with
smooth interpolation between each pair, captures the JPEG frames streamed over
ZMQ, and writes them to an MP4 video file.

Required files on this machine (copy from server repo):
  - generation_control_pb2.py
  - generation_control_pb2_grpc.py

Install:
  pip install grpcio zmq opencv-python numpy
"""

import argparse
import time
import threading
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import zmq
import grpc

import generation_control_pb2 as pb2
import generation_control_pb2_grpc as pb2_grpc

# ============================================
# SERVER CONNECTION
SERVER_IP = "192.168.10.130"
GRPC_PORT = 50053
ZMQ_PORT = 5555
# ============================================

# Default prompts — edit these or pass via --prompts-file
DEFAULT_PROMPTS = [
    "a majestic mountain at golden sunset, cinematic lighting",
    "a starry night sky over a calm ocean, milky way visible",
    "an underwater coral reef glowing with bioluminescence",
    "a neon cyberpunk cityscape in the rain, reflections on wet streets",
    "a vast alien desert with two moons rising, purple sky",
]


class FrameReceiver:
    """Background thread that receives JPEG frames from ZMQ and buffers the latest."""

    def __init__(self, server_ip: str, zmq_port: int):
        self.address = f"tcp://{server_ip}:{zmq_port}"
        self.latest_frame = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.RCVTIMEO, 2000)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self.address)
        print(f"[ZMQ] Connected to {self.address}")

        while not self._stop.is_set():
            try:
                data = sub.recv()
                frame = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.frame_count += 1
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[ZMQ] Error: {e}")
                break

        sub.close()
        ctx.term()

    def get_frame(self):
        with self.lock:
            return self.latest_frame


def load_prompts(path: str) -> list[str]:
    """Load prompts from a text file (one per line, blank lines / # comments ignored)."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def run_journey(
    prompts: list[str],
    server_ip: str,
    grpc_port: int,
    zmq_port: int,
    transition_frames: int,
    hold_frames: int,
    fps: int,
    output: str,
    loop: bool,
):
    # --- Connect gRPC ---
    channel = grpc.insecure_channel(f"{server_ip}:{grpc_port}")
    stub = pb2_grpc.GenerationControlStub(channel)

    # Verify connection
    try:
        state = stub.GetCurrentState(pb2.GetStateRequest())
        print(f"[gRPC] Connected — current prompt: {state.prompt!r}")
        print(f"[gRPC] Resolution: {state.width}x{state.height}")
    except grpc.RpcError as e:
        print(f"[gRPC] Connection failed: {e}")
        sys.exit(1)

    # --- Start frame receiver ---
    receiver = FrameReceiver(server_ip, zmq_port)
    receiver.start()

    # Wait for first frame so we know the resolution
    print("[ZMQ] Waiting for first frame...")
    deadline = time.time() + 10
    while receiver.get_frame() is None:
        if time.time() > deadline:
            print("[ZMQ] Timeout waiting for frames. Is the server streaming?")
            sys.exit(1)
        time.sleep(0.1)

    first_frame = receiver.get_frame()
    h, w = first_frame.shape[:2]
    print(f"[Video] Frame size: {w}x{h}")

    # --- Set up video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[Video] Failed to open {output} for writing")
        sys.exit(1)

    frame_interval = 1.0 / fps
    total_written = 0

    # Graceful shutdown
    stop_event = threading.Event()

    def on_signal(sig, _):
        print("\n[!] Interrupted — finishing video...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)

    # --- Disable the server's auto-scheduler so we control the factor ---
    stub.SetPromptTravelParams(pb2.PromptTravelParamsRequest(enabled=False))

    def capture_frames(n_frames: int):
        """Write n_frames to video at the configured fps."""
        nonlocal total_written
        for _ in range(n_frames):
            if stop_event.is_set():
                return
            t0 = time.time()
            frame = receiver.get_frame()
            if frame is not None:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                total_written += 1
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # --- Append first prompt again if looping ---
    if loop:
        prompts = prompts + [prompts[0]]

    # --- Run the journey ---
    print(f"\n{'='*60}")
    print(f"  Prompt Journey — {len(prompts)} prompts")
    print(f"  {transition_frames} transition frames + {hold_frames} hold frames each")
    print(f"  Output: {output} @ {fps} fps")
    print(f"{'='*60}\n")

    for i in range(len(prompts) - 1):
        if stop_event.is_set():
            break

        src = prompts[i]
        dst = prompts[i + 1]
        print(f"[{i+1}/{len(prompts)-1}] {src!r}")
        print(f"       -> {dst!r}")

        # Set source prompt and hold
        stub.SetPrompt(pb2.PromptRequest(
            prompt=src,
            target_prompt=dst,
            prompt_travel_factor=0.0,
        ))
        print(f"  Holding source for {hold_frames} frames...")
        capture_frames(hold_frames)

        # Interpolate source -> target
        print(f"  Transitioning over {transition_frames} frames...")
        for step in range(transition_frames):
            if stop_event.is_set():
                break
            factor = step / max(transition_frames - 1, 1)
            stub.SetPrompt(pb2.PromptRequest(
                prompt=src,
                target_prompt=dst,
                prompt_travel_factor=factor,
            ))
            capture_frames(1)

    # Hold on final prompt
    if not stop_event.is_set():
        print(f"  Holding final prompt for {hold_frames} frames...")
        stub.SetPrompt(pb2.PromptRequest(
            prompt=prompts[-1],
            prompt_travel_factor=0.0,
        ))
        capture_frames(hold_frames)

    # --- Finalize ---
    writer.release()
    receiver.stop()
    channel.close()

    duration = total_written / fps if fps else 0
    print(f"\nDone! Wrote {total_written} frames ({duration:.1f}s) to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-prompt journey video recorder for StreamDiffusion"
    )
    parser.add_argument(
        "--server", default=SERVER_IP,
        help=f"Server IP (default: {SERVER_IP})",
    )
    parser.add_argument(
        "--grpc-port", type=int, default=GRPC_PORT,
        help=f"gRPC port (default: {GRPC_PORT})",
    )
    parser.add_argument(
        "--zmq-port", type=int, default=ZMQ_PORT,
        help=f"ZMQ port (default: {ZMQ_PORT})",
    )
    parser.add_argument(
        "--prompts-file", type=str, default=None,
        help="Text file with one prompt per line (overrides built-in list)",
    )
    parser.add_argument(
        "--transition-frames", type=int, default=120,
        help="Frames for each prompt-to-prompt interpolation (default: 120)",
    )
    parser.add_argument(
        "--hold-frames", type=int, default=30,
        help="Frames to hold on each prompt before transitioning (default: 30)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video framerate (default: 30)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="prompt_journey.mp4",
        help="Output filename (default: prompt_journey.mp4)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop back to first prompt at the end",
    )

    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    if len(prompts) < 2:
        print("Need at least 2 prompts for a journey!")
        sys.exit(1)

    run_journey(
        prompts=prompts,
        server_ip=args.server,
        grpc_port=args.grpc_port,
        zmq_port=args.zmq_port,
        transition_frames=args.transition_frames,
        hold_frames=args.hold_frames,
        fps=args.fps,
        output=args.output,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
