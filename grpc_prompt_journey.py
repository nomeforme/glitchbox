#!/usr/bin/env python3
"""
Multi-prompt journey & scheduler video recorder for StreamDiffusion.

Two modes:
  1. Journey mode (default): Send prompts, server interpolates and generates
  2. Scheduler mode (--scheduler): Use the server's built-in prompt travel scheduler

Required files on remote machine:
  - generation_control_pb2.py
  - generation_control_pb2_grpc.py

Install: pip install grpcio pyzmq opencv-python numpy
"""

import argparse
import os
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

SERVER_IP = "192.168.10.130"
GRPC_PORT = 50053
ZMQ_PORT = 5555

DEFAULT_PROMPTS = [
    "twisted bodies Pale beige sculptural forms with flowing organic shapes and smooth ovoid elements against black background.",
    "water Dramatic black and white portrait of face submerged in splashing water against dark background.",
    "twisted bodies Pale cream sculptural figures with flowing organic tendrils merging together against black background.",
    "water Dynamic water splash frozen mid-motion against black background, high contrast monochrome photography with crystalline droplet details.",
    "twisted bodies Pale beige sculptural figures in dynamic motion against black background, organic flowing forms intertwined.",
    "water Transparent water sculpture forming human face with dynamic splashing droplets on black background.",
]


def upload_file(server_ip, local_path):
    """Upload a local file to the server, return the server-side path."""
    import urllib.request
    import json as _json
    print(f"[Upload] Uploading: {local_path}...")
    with open(local_path, "rb") as f:
        data = f.read()
    filename = os.path.basename(local_path)
    req = urllib.request.Request(
        f"http://{server_ip}:7860/api/upload?filename={filename}",
        data=data, method="POST",
    )
    req.add_header("Content-Type", "application/octet-stream")
    resp = urllib.request.urlopen(req)
    server_path = _json.loads(resp.read())["path"]
    print(f"[Upload] Uploaded to server: {server_path}")
    return server_path


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def connect_and_setup(server_ip, grpc_port, curation_index=None, pipe_index=None,
                      controlnet_scale=None, temporal_coherence=None):
    """Connect to gRPC and apply initial settings. Returns (channel, stub)."""
    channel = grpc.insecure_channel(f"{server_ip}:{grpc_port}")
    stub = pb2_grpc.GenerationControlStub(channel)

    try:
        state = stub.GetCurrentState(pb2.GetStateRequest())
        print(f"[gRPC] Connected — current prompt: {state.prompt[:50]!r}...")
    except grpc.RpcError as e:
        print(f"[gRPC] Connection failed: {e}")
        sys.exit(1)

    if curation_index is not None:
        print(f"[gRPC] Switching to curation index {curation_index}...")
        resp = stub.SwitchCuration(pb2.SwitchCurationRequest(curation_index=curation_index))
        print(f"[gRPC] {resp.message}")
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

    return channel, stub


def zmq_connect(server_ip, zmq_port):
    address = f"tcp://{server_ip}:{zmq_port}"
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 5000)
    sub.setsockopt(zmq.RCVHWM, 2)  # Only buffer 2 frames
    sub.setsockopt(zmq.RCVBUF, 20 * 1024 * 1024)  # 20MB receive buffer
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(address)
    print(f"[ZMQ] Connected to {address}")
    return ctx, sub


def wait_for_first_frame(zmq_sub, warmup=20):
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
        print("[ZMQ] Timeout waiting for frames. Is the server in --headless mode?")
        sys.exit(1)

    print(f"[ZMQ] Discarding {warmup} warmup frames...", end="", flush=True)
    for _ in range(warmup):
        try:
            data = zmq_sub.recv()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                first_frame = frame
        except zmq.Again:
            pass
    print(" done.")
    return first_frame


def record_zmq_stream(zmq_sub, writer, h, w, fps, duration_seconds, stub=None):
    stop_event = threading.Event()

    def on_signal(sig, _):
        print("\n[!] Interrupted — finishing video...")
        stop_event.set()
        if stub:
            try:
                stub.StopPromptJourney(pb2.GetStateRequest())
            except Exception:
                pass

    signal.signal(signal.SIGINT, on_signal)

    frames_written = 0
    start_time = time.time()
    last_status_time = start_time

    if duration_seconds:
        print(f"\nRecording for {duration_seconds}s @ {fps} fps. Press Ctrl+C to stop early.\n")
    else:
        print(f"\nRecording @ {fps} fps. Press Ctrl+C to stop.\n")

    while not stop_event.is_set():
        if duration_seconds and (time.time() - start_time) >= duration_seconds:
            print(f"\n[Recording] Duration reached ({duration_seconds}s)")
            break

        try:
            data = zmq_sub.recv()
        except zmq.Again:
            if stub:
                try:
                    status = stub.GetJourneyStatus(pb2.GetStateRequest())
                    if status.completed:
                        print(f"\n[gRPC] Journey completed")
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

        now = time.time()
        if now - last_status_time > 3.0:
            elapsed = now - start_time
            print(f"  {frames_written} frames ({elapsed:.1f}s elapsed)", end="\r")
            last_status_time = now

    # Drain remaining
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

    return frames_written


def run_journey(prompts, server_ip, grpc_port, zmq_port, transition_frames,
                hold_frames, fps, output, loop, input_video=None, init_image=None, **setup_kwargs):
    channel, stub = connect_and_setup(server_ip, grpc_port, **setup_kwargs)

    # Upload init image if provided (overrides input_video)
    if init_image:
        input_video = upload_file(server_ip, init_image)
    zmq_ctx, zmq_sub = zmq_connect(server_ip, zmq_port)

    print(f"[gRPC] Starting journey: {len(prompts)} prompts, "
          f"{transition_frames} transition frames, {hold_frames} hold frames")

    journey_kwargs = dict(
        prompts=prompts, transition_frames=transition_frames,
        hold_frames=hold_frames, loop=loop,
    )
    if input_video:
        journey_kwargs['input_video'] = input_video

    response = stub.StartPromptJourney(pb2.PromptJourneyRequest(**journey_kwargs))
    if not response.success:
        print(f"[gRPC] Server rejected: {response.message}")
        sys.exit(1)
    print(f"[gRPC] {response.message}")

    first_frame = wait_for_first_frame(zmq_sub)
    h, w = first_frame.shape[:2]
    print(f"[Video] Frame size: {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    writer.write(first_frame)

    frames_written = 1 + record_zmq_stream(zmq_sub, writer, h, w, fps, None, stub)

    writer.release()
    zmq_sub.close()
    zmq_ctx.term()
    channel.close()
    print(f"\nDone! Wrote {frames_written} frames ({frames_written/fps:.1f}s) to {output}")


def run_scheduler(server_ip, grpc_port, zmq_port, fps, output, duration_seconds,
                  input_video=None, audio_file=None, init_image=None, **setup_kwargs):
    channel, stub = connect_and_setup(server_ip, grpc_port, **setup_kwargs)

    # Upload init image if provided (overrides input_video)
    if init_image:
        input_video = upload_file(server_ip, init_image)

    # Upload audio file to server if it's a local path
    server_audio_path = audio_file
    local_audio_path = None
    if audio_file and not audio_file.startswith("/"):
        local_audio_path = audio_file
        server_audio_path = upload_file(server_ip, audio_file)
    elif audio_file:
        local_audio_path = None  # Audio is already on server

    # Server-side output path
    server_output = f"/tmp/glitchbox_output_{int(time.time())}.mp4"

    # Pass all params via a dummy journey
    journey_kwargs = dict(
        prompts=["_scheduler_", "_scheduler_"],
        transition_frames=999999, hold_frames=0, loop=False,
        duration=duration_seconds or 0,
        output_path=server_output,
        fps=fps,
    )
    if input_video:
        journey_kwargs['input_video'] = input_video
    if server_audio_path:
        journey_kwargs['audio_file'] = server_audio_path
    stub.StartPromptJourney(pb2.PromptJourneyRequest(**journey_kwargs))
    stub.StopPromptJourney(pb2.GetStateRequest())

    print("[gRPC] Enabling server prompt travel scheduler...")
    stub.SetPromptTravelParams(pb2.PromptTravelParamsRequest(
        enabled=True, use_prompt_scheduler=True,
        loop_prompts=True, oscillate=True,
        factor_increment=0.005, stabilize_duration=0,
    ))

    print(f"\nServer generating video to: {server_output}")
    print(f"Duration: {duration_seconds}s @ {fps}fps")
    print("Waiting for generation to complete... (Ctrl+C to abort)\n")

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    while True:
        time.sleep(2)
        try:
            status = stub.GetJourneyStatus(pb2.GetStateRequest())
            total = status.total_frames if status.total_frames > 0 else "?"
            print(f"  Frame {status.current_frame}/{total} generated...", end="\r")
            if status.completed:
                print(f"\n\n[Server] Generation complete: {status.current_frame} frames")
                break
        except grpc.RpcError as e:
            print(f"\n[gRPC] Connection lost: {e}")
            break

    channel.close()

    # Download the video from server via HTTP
    video_only = output.replace(".mp4", "_noaudio.mp4") if (audio_file or local_audio_path) else output
    print(f"\nDownloading {server_output}...")
    import urllib.request
    download_url = f"http://{server_ip}:7860/api/download?path={server_output}"
    try:
        urllib.request.urlretrieve(download_url, video_only)
        print(f"Downloaded: {video_only}")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print(f"Download manually: scp plantoidz@{server_ip}:{server_output} {video_only}")
        return

    # Merge audio into video if audio was provided
    audio_source = local_audio_path or audio_file
    if audio_source and os.path.exists(audio_source):
        print(f"Merging audio from {audio_source}...")
        import subprocess
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_only,
                "-i", audio_source,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output,
            ], check=True, capture_output=True)
            os.remove(video_only)
            print(f"Final video with audio: {output}")
        except Exception as e:
            print(f"Audio merge failed: {e}")
            print(f"Video without audio: {video_only}")
    else:
        if video_only != output:
            os.rename(video_only, output)
        print(f"Saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-prompt journey & scheduler video recorder for StreamDiffusion"
    )
    parser.add_argument("--server", default=SERVER_IP)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT)
    parser.add_argument("--zmq-port", type=int, default=ZMQ_PORT)
    parser.add_argument("--prompts-file", type=str, default=None, help="Prompts file (one per line)")
    parser.add_argument("--transition-frames", type=int, default=120)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default="prompt_journey.mp4")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--curation", type=int, default=None, help="Curation index (e.g. 21)")
    parser.add_argument("--pipe-index", type=int, default=None, help="LoRA pipe index")
    parser.add_argument("--input-video", type=str, default=None, help="Server-side video path or 'noise'")
    parser.add_argument("--init-image", type=str, default=None, help="Local init image to upload and use as input")
    parser.add_argument("--controlnet-scale", type=float, default=None, help="ControlNet scale")
    parser.add_argument("--temporal-coherence", type=float, default=None, help="Temporal coherence (try 0.2-0.4)")
    parser.add_argument("--audio", type=str, default=None, help="Server-side audio file for FFT effects")
    parser.add_argument("--scheduler", action="store_true", help="Use server's prompt travel scheduler")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds for --scheduler mode")

    args = parser.parse_args()

    setup_kwargs = dict(
        curation_index=args.curation, pipe_index=args.pipe_index,
        controlnet_scale=args.controlnet_scale, temporal_coherence=args.temporal_coherence,
    )

    if args.scheduler:
        run_scheduler(
            server_ip=args.server, grpc_port=args.grpc_port, zmq_port=args.zmq_port,
            fps=args.fps, output=args.output, duration_seconds=args.duration,
            input_video=args.input_video, audio_file=args.audio,
            init_image=args.init_image, **setup_kwargs,
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
            prompts=prompts, server_ip=args.server, grpc_port=args.grpc_port,
            zmq_port=args.zmq_port, transition_frames=args.transition_frames,
            hold_frames=args.hold_frames, fps=args.fps, output=args.output,
            loop=args.loop, input_video=args.input_video,
            init_image=args.init_image, **setup_kwargs,
        )


if __name__ == "__main__":
    main()
