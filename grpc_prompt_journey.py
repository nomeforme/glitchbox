#!/usr/bin/env python3
"""
Multi-prompt journey video recorder.

Sends the full journey spec to the server in one gRPC call, then records
the ZMQ frame stream into an MP4 video. The server generates all frames
autonomously — no frame-by-frame coordination needed.

Required files on this machine (copy from server repo):
  - generation_control_pb2.py
  - generation_control_pb2_grpc.py

Install:
  pip install grpcio pyzmq opencv-python numpy
"""

import argparse
import time
import threading
import signal
import sys

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
    "twisted bodies Pale beige sculptural forms with flowing organic shapes and smooth ovoid elements against black background.",
    "water Dramatic black and white portrait of face submerged in splashing water against dark background.",
    "twisted bodies Pale cream sculptural figures with flowing organic tendrils merging together against black background.",
    "water Dynamic water splash frozen mid-motion against black background, high contrast monochrome photography with crystalline droplet details.",
    "twisted bodies Pale beige sculptural figures in dynamic motion against black background, organic flowing forms intertwined.",
    "water Transparent water sculpture forming human face with dynamic splashing droplets on black background.",
]


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
    curation_index: int = None,
    pipe_index: int = None,
    input_video: str = None,
    controlnet_scale: float = None,
):
    # --- Connect gRPC ---
    channel = grpc.insecure_channel(f"{server_ip}:{grpc_port}")
    stub = pb2_grpc.GenerationControlStub(channel)

    # Verify connection
    try:
        state = stub.GetCurrentState(pb2.GetStateRequest())
        print(f"[gRPC] Connected — current prompt: {state.prompt!r}")
    except grpc.RpcError as e:
        print(f"[gRPC] Connection failed: {e}")
        sys.exit(1)

    # --- Switch curation / pipe if requested ---
    if curation_index is not None:
        print(f"[gRPC] Switching to curation index {curation_index} (this rebuilds the pipeline, may take a while)...")
        resp = stub.SwitchCuration(pb2.SwitchCurationRequest(curation_index=curation_index))
        print(f"[gRPC] Curation switch: {resp.message}")
        # Wait for the pipeline to finish rebuilding
        print("[gRPC] Waiting for pipeline reload...", end="", flush=True)
        time.sleep(5)
        for _ in range(60):  # up to 60 more seconds
            try:
                stub.GetCurrentState(pb2.GetStateRequest())
                break
            except grpc.RpcError:
                print(".", end="", flush=True)
                time.sleep(1)
        print(" done.")

    if pipe_index is not None:
        print(f"[gRPC] Setting pipe index to {pipe_index}")
        stub.SetLoRAParams(pb2.LoRAParamsRequest(pipe_index=pipe_index))

    if controlnet_scale is not None:
        print(f"[gRPC] Setting ControlNet scale to {controlnet_scale}")
        stub.SetControlNetParams(pb2.ControlNetParamsRequest(controlnet_scale=controlnet_scale))

    # --- Start ZMQ receiver FIRST (so we don't miss early frames) ---
    zmq_address = f"tcp://{server_ip}:{zmq_port}"
    zmq_ctx = zmq.Context()
    zmq_sub = zmq_ctx.socket(zmq.SUB)
    zmq_sub.setsockopt(zmq.RCVTIMEO, 5000)
    zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    zmq_sub.connect(zmq_address)
    print(f"[ZMQ] Connected to {zmq_address}")

    # --- Send the full journey to the server ---
    print(f"[gRPC] Starting journey: {len(prompts)} prompts, "
          f"{transition_frames} transition frames, {hold_frames} hold frames"
          f"{', looping' if loop else ''}"
          f"{f', input video: {input_video}' if input_video else ''}")

    journey_kwargs = dict(
        prompts=prompts,
        transition_frames=transition_frames,
        hold_frames=hold_frames,
        loop=loop,
    )
    if input_video:
        journey_kwargs['input_video'] = input_video

    response = stub.StartPromptJourney(pb2.PromptJourneyRequest(**journey_kwargs))

    if not response.success:
        print(f"[gRPC] Server rejected journey: {response.message}")
        sys.exit(1)

    total_frames = response.total_frames
    print(f"[gRPC] Server accepted: {response.message}")
    print(f"[gRPC] Total frames to generate: {total_frames}")

    # --- Wait for first frame to get resolution ---
    print("[ZMQ] Waiting for first frame...")
    first_frame = None
    for _ in range(60):  # up to 30s at 500ms intervals
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                first_frame = frame
                break
        except zmq.Again:
            continue

    if first_frame is None:
        print("[ZMQ] Timeout waiting for frames. Is the server in --headless mode?")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    print(f"[Video] Frame size: {w}x{h}")

    # --- Set up video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[Video] Failed to open {output} for writing")
        sys.exit(1)

    # Write the first frame
    if first_frame.shape[:2] != (h, w):
        first_frame = cv2.resize(first_frame, (w, h))
    writer.write(first_frame)
    frames_written = 1

    # Graceful shutdown
    stop_event = threading.Event()

    def on_signal(sig, _):
        print("\n[!] Interrupted — finishing video...")
        stop_event.set()
        # Tell server to stop too
        try:
            stub.StopPromptJourney(pb2.GetStateRequest())
        except Exception:
            pass

    signal.signal(signal.SIGINT, on_signal)

    # --- Record frames until journey completes ---
    print(f"\nRecording {total_frames} frames to {output} @ {fps} fps...")
    last_status_time = time.time()

    while not stop_event.is_set():
        try:
            data = zmq_sub.recv()
        except zmq.Again:
            # Check if journey is done
            try:
                status = stub.GetJourneyStatus(pb2.GetStateRequest())
                if status.completed:
                    print(f"\n[gRPC] Journey completed on server")
                    break
            except Exception:
                pass
            continue

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        frames_written += 1

        # Print progress periodically
        now = time.time()
        if now - last_status_time > 2.0:
            try:
                status = stub.GetJourneyStatus(pb2.GetStateRequest())
                pct = status.progress * 100
                print(f"  [{status.current_frame}/{status.total_frames}] "
                      f"{pct:.0f}% — {status.current_prompt[:50]}...",
                      end="\r")
                if status.completed:
                    print(f"\n[gRPC] Journey completed on server")
                    break
            except Exception:
                pass
            last_status_time = now

    # Drain any remaining frames in the ZMQ buffer
    zmq_sub.setsockopt(zmq.RCVTIMEO, 500)
    while True:
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                frames_written += 1
        except zmq.Again:
            break

    # --- Finalize ---
    writer.release()
    zmq_sub.close()
    zmq_ctx.term()
    channel.close()

    duration = frames_written / fps if fps else 0
    print(f"\nDone! Wrote {frames_written} frames ({duration:.1f}s) to {output}")


def run_scheduler(
    server_ip: str,
    grpc_port: int,
    zmq_port: int,
    fps: int,
    output: str,
    duration_seconds: float,
    curation_index: int = None,
    pipe_index: int = None,
    input_video: str = None,
    controlnet_scale: float = None,
    temporal_coherence: float = None,
    audio_file: str = None,
):
    """Record using the server's built-in prompt travel scheduler."""
    channel = grpc.insecure_channel(f"{server_ip}:{grpc_port}")
    stub = pb2_grpc.GenerationControlStub(channel)

    # Verify connection
    try:
        state = stub.GetCurrentState(pb2.GetStateRequest())
        print(f"[gRPC] Connected — current prompt: {state.prompt!r}")
    except grpc.RpcError as e:
        print(f"[gRPC] Connection failed: {e}")
        sys.exit(1)

    # Switch curation if requested
    if curation_index is not None:
        print(f"[gRPC] Switching to curation index {curation_index}...")
        resp = stub.SwitchCuration(pb2.SwitchCurationRequest(curation_index=curation_index))
        print(f"[gRPC] Curation switch: {resp.message}")
        print("[gRPC] Waiting for pipeline reload...", end="", flush=True)
        time.sleep(5)
        for _ in range(60):
            try:
                stub.GetCurrentState(pb2.GetStateRequest())
                break
            except grpc.RpcError:
                print(".", end="", flush=True)
                time.sleep(1)
        print(" done.")

    if pipe_index is not None:
        print(f"[gRPC] Setting pipe index to {pipe_index}")
        stub.SetLoRAParams(pb2.LoRAParamsRequest(pipe_index=pipe_index))

    if controlnet_scale is not None:
        print(f"[gRPC] Setting ControlNet scale to {controlnet_scale}")
        stub.SetControlNetParams(pb2.ControlNetParamsRequest(controlnet_scale=controlnet_scale))

    if temporal_coherence is not None:
        print(f"[gRPC] Setting temporal coherence to {temporal_coherence}")
        stub.SetGenerationParams(pb2.GenerationParamsRequest(
            temporal_coherence=temporal_coherence,
            temporal_coherence_latent=temporal_coherence,
        ))

    # If input video or audio specified, start a dummy journey to pass the paths
    if input_video or audio_file:
        journey_kwargs = dict(
            prompts=["_scheduler_", "_scheduler_"],
            transition_frames=999999,
            hold_frames=0,
            loop=False,
        )
        if input_video:
            journey_kwargs['input_video'] = input_video
        if audio_file:
            journey_kwargs['audio_file'] = audio_file
        stub.StartPromptJourney(pb2.PromptJourneyRequest(**journey_kwargs))
        # Immediately stop the journey so the scheduler takes over
        stub.StopPromptJourney(pb2.GetStateRequest())

    # Enable the server's prompt travel scheduler
    print("[gRPC] Enabling server prompt travel scheduler...")
    stub.SetPromptTravelParams(pb2.PromptTravelParamsRequest(
        enabled=True,
        use_prompt_scheduler=True,
        loop_prompts=True,
        oscillate=True,
        factor_increment=0.005,  # Slow, smooth transitions
        stabilize_duration=0,    # No pause at endpoints
    ))

    # Start ZMQ receiver
    zmq_address = f"tcp://{server_ip}:{zmq_port}"
    zmq_ctx = zmq.Context()
    zmq_sub = zmq_ctx.socket(zmq.SUB)
    zmq_sub.setsockopt(zmq.RCVTIMEO, 5000)
    zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    zmq_sub.connect(zmq_address)
    print(f"[ZMQ] Connected to {zmq_address}")

    # Wait for first frame
    print("[ZMQ] Waiting for first frame...")
    first_frame = None
    for _ in range(60):
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                first_frame = frame
                break
        except zmq.Again:
            continue

    if first_frame is None:
        print("[ZMQ] Timeout waiting for frames.")
        sys.exit(1)

    h, w = first_frame.shape[:2]
    print(f"[Video] Frame size: {w}x{h}")

    # Discard warmup frames (pipeline needs a few frames to settle)
    print("[ZMQ] Discarding warmup frames...", end="", flush=True)
    for _ in range(20):
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                first_frame = frame
        except zmq.Again:
            pass
    print(" done.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    writer.write(first_frame)
    frames_written = 1

    stop_event = threading.Event()

    def on_signal(sig, _):
        print("\n[!] Interrupted — finishing video...")
        stop_event.set()

    signal.signal(signal.SIGINT, on_signal)

    total_frames = int(duration_seconds * fps) if duration_seconds else 0
    print(f"\nRecording for {duration_seconds}s ({total_frames} frames) to {output} @ {fps} fps...")
    print("Press Ctrl+C to stop early.\n")
    start_time = time.time()

    while not stop_event.is_set():
        # Check duration
        if duration_seconds and (time.time() - start_time) >= duration_seconds:
            print(f"\n[Recording] Duration reached ({duration_seconds}s)")
            break

        try:
            data = zmq_sub.recv()
        except zmq.Again:
            continue

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        frames_written += 1

        if frames_written % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {frames_written} frames ({elapsed:.1f}s elapsed)")

    # Disable scheduler
    stub.SetPromptTravelParams(pb2.PromptTravelParamsRequest(enabled=False))

    # Drain remaining frames
    zmq_sub.setsockopt(zmq.RCVTIMEO, 500)
    while True:
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                frames_written += 1
        except zmq.Again:
            break

    writer.release()
    zmq_sub.close()
    zmq_ctx.term()
    channel.close()

    duration_actual = frames_written / fps if fps else 0
    print(f"\nDone! Wrote {frames_written} frames ({duration_actual:.1f}s) to {output}")


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
    parser.add_argument(
        "--curation", type=int, default=None,
        help="Curation index to switch to before starting (e.g. 27 for shaman_XL)",
    )
    parser.add_argument(
        "--pipe-index", type=int, default=None,
        help="Pipe index (LoRA weight blend) to use (e.g. 0=first, 2=50/50)",
    )
    parser.add_argument(
        "--input-video", type=str, default=None,
        help="Server-side path to input video for ControlNet depth guidance",
    )
    parser.add_argument(
        "--controlnet-scale", type=float, default=None,
        help="ControlNet conditioning scale (default: 0.55, try 1.0-1.5 for stronger guidance)",
    )
    parser.add_argument(
        "--temporal-coherence", type=float, default=None,
        help="Temporal coherence for noise/latent blending (default: 0.03, try 0.1-0.3 for smoother)",
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Server-side path to audio file for FFT-driven effects (zoom, LoRA switching). Duration defaults to audio length.",
    )
    parser.add_argument(
        "--scheduler", action="store_true",
        help="Use the server's built-in prompt travel scheduler instead of journey prompts",
    )
    parser.add_argument(
        "--duration", type=float, default=30,
        help="Recording duration in seconds for --scheduler mode (default: 30)",
    )

    args = parser.parse_args()

    if args.scheduler:
        run_scheduler(
            server_ip=args.server,
            grpc_port=args.grpc_port,
            zmq_port=args.zmq_port,
            fps=args.fps,
            output=args.output,
            duration_seconds=args.duration,
            curation_index=args.curation,
            pipe_index=args.pipe_index,
            input_video=args.input_video,
            controlnet_scale=args.controlnet_scale,
            temporal_coherence=args.temporal_coherence,
            audio_file=args.audio,
        )
    else:
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
            curation_index=args.curation,
            pipe_index=args.pipe_index,
            input_video=args.input_video,
            controlnet_scale=args.controlnet_scale,
        )


if __name__ == "__main__":
    main()
