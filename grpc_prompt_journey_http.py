#!/usr/bin/env python3
"""
Lightweight HTTP client for GLITCHBOX server.

No dependencies beyond Python stdlib — runs on Raspberry Pi, etc.

Supports journey mode and scheduler mode, same CLI as grpc_prompt_journey.py.
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error
import uuid


SERVER_IP = "192.168.10.130"
HTTP_PORT = 7860

DEFAULT_PROMPTS = [
    "twisted bodies Pale beige sculptural forms with flowing organic shapes and smooth ovoid elements against black background.",
    "water Dramatic black and white portrait of face submerged in splashing water against dark background.",
    "twisted bodies Pale cream sculptural figures with flowing organic tendrils merging together against black background.",
    "water Dynamic water splash frozen mid-motion against black background, high contrast monochrome photography with crystalline droplet details.",
    "twisted bodies Pale beige sculptural figures in dynamic motion against black background, organic flowing forms intertwined.",
    "water Transparent water sculpture forming human face with dynamic splashing droplets on black background.",
]


def _post(server_ip, port, path, data=None):
    """POST JSON to server, return parsed response."""
    url = f"http://{server_ip}:{port}{path}"
    body = json.dumps(data).encode() if data else b""
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=300)
    return json.loads(resp.read())


def _get(server_ip, port, path):
    """GET from server, return parsed response."""
    url = f"http://{server_ip}:{port}{path}"
    resp = urllib.request.urlopen(url, timeout=30)
    return json.loads(resp.read())


def upload_file(server_ip, port, local_path):
    """Upload a local file to the server, return the server-side path."""
    print(f"[Upload] Uploading: {local_path}...")
    with open(local_path, "rb") as f:
        data = f.read()
    filename = os.path.basename(local_path)
    req = urllib.request.Request(
        f"http://{server_ip}:{port}/api/upload?filename={filename}",
        data=data, method="POST",
    )
    req.add_header("Content-Type", "application/octet-stream")
    resp = urllib.request.urlopen(req)
    server_path = json.loads(resp.read())["path"]
    print(f"[Upload] Uploaded to server: {server_path}")
    return server_path


def download_file(server_ip, port, server_path, local_path):
    """Download a file from the server."""
    print(f"\nDownloading {server_path}...")
    url = f"http://{server_ip}:{port}/api/download?path={server_path}"
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Manual: scp plantoidz@{server_ip}:{server_path} {local_path}")
        return False


def merge_audio(video_path, audio_path, output_path):
    """Merge audio into video using ffmpeg."""
    print(f"Merging audio from {audio_path}...")
    import subprocess
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ], check=True, capture_output=True)
        os.remove(video_path)
        print(f"Final video with audio: {output_path}")
    except Exception as e:
        print(f"Audio merge failed: {e}")
        print(f"Video without audio: {video_path}")


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def switch_curation(server_ip, port, curation_index):
    """Switch curation and wait for pipeline reload."""
    print(f"[HTTP] Switching to curation index {curation_index}...")
    _post(server_ip, port, "/api/update_curation_index", {"curation_index": curation_index})
    print("[HTTP] Waiting for pipeline reload...", end="", flush=True)
    time.sleep(5)
    for _ in range(60):
        try:
            _get(server_ip, port, "/api/journey/status")
            break
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" done.")


def set_params(server_ip, port, **params):
    """Set generation params (strength, temporal_coherence, etc.)."""
    params = {k: v for k, v in params.items() if v is not None}
    if params:
        print(f"[HTTP] Setting params: {params}")
        _post(server_ip, port, "/api/params/generation", params)


def poll_until_done(server_ip, port):
    """Poll journey status until completed. Returns (frame_count, output_file)."""
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    while True:
        time.sleep(2)
        try:
            status = _get(server_ip, port, "/api/journey/status")
            total = status["total_frames"] if status["total_frames"] > 0 else "?"
            print(f"  Frame {status['current_frame']}/{total} generated...", end="\r")
            if status["completed"]:
                print(f"\n\n[Server] Generation complete: {status['current_frame']} frames")
                return status["current_frame"], status.get("output_file", "")
        except Exception as e:
            print(f"\n[HTTP] Connection lost: {e}")
            return 0, ""


def run_journey(prompts, server_ip, port, fps, output, transition_frames,
                hold_frames, loop, input_video=None, init_image=None,
                audio_file=None, curation_index=None, strength=None,
                temporal_coherence=None, pipe_index=None, controlnet_scale=None):
    # Setup
    if curation_index is not None:
        switch_curation(server_ip, port, curation_index)

    set_params(server_ip, port, strength=strength,
               temporal_coherence=temporal_coherence,
               temporal_coherence_latent=temporal_coherence,
               controlnet_scale=controlnet_scale)

    # Upload files
    if init_image:
        input_video = upload_file(server_ip, port, init_image)

    server_audio_path = audio_file
    local_audio_path = None
    if audio_file and os.path.exists(audio_file):
        local_audio_path = audio_file
        server_audio_path = upload_file(server_ip, port, audio_file)

    server_output = f"/tmp/glitchbox_output_{int(time.time())}.mp4"

    print(f"[HTTP] Starting journey: {len(prompts)} prompts, "
          f"{transition_frames} transition frames, {hold_frames} hold frames")

    journey_data = {
        "prompts": prompts,
        "transition_frames": transition_frames,
        "hold_frames": hold_frames,
        "loop": loop,
        "output_path": server_output,
        "fps": fps,
    }
    if input_video:
        journey_data["input_video"] = input_video
    if server_audio_path:
        journey_data["audio_file"] = server_audio_path

    result = _post(server_ip, port, "/api/journey/start", journey_data)
    if result.get("status") != "success":
        print(f"[HTTP] Server rejected: {result.get('message')}")
        sys.exit(1)
    print(f"[HTTP] {result['message']}")

    print(f"\nServer generating video to: {server_output}")
    print("Waiting for generation to complete... (Ctrl+C to abort)\n")

    _, actual_output = poll_until_done(server_ip, port)
    actual_output = actual_output or server_output

    # Download
    video_only = output.replace(".mp4", "_noaudio.mp4") if local_audio_path else output
    if not download_file(server_ip, port, actual_output, video_only):
        return

    if local_audio_path and os.path.exists(local_audio_path):
        merge_audio(video_only, local_audio_path, output)
    else:
        if video_only != output:
            os.rename(video_only, output)
        print(f"Saved to: {output}")


def run_scheduler(server_ip, port, fps, output, duration_seconds,
                  input_video=None, audio_file=None, init_image=None,
                  curation_index=None, strength=None, temporal_coherence=None,
                  pipe_index=None, controlnet_scale=None):
    # Setup
    if curation_index is not None:
        switch_curation(server_ip, port, curation_index)

    set_params(server_ip, port, strength=strength,
               temporal_coherence=temporal_coherence,
               temporal_coherence_latent=temporal_coherence,
               controlnet_scale=controlnet_scale)

    # Upload files
    if init_image:
        input_video = upload_file(server_ip, port, init_image)

    server_audio_path = audio_file
    local_audio_path = None
    if audio_file and os.path.exists(audio_file):
        local_audio_path = audio_file
        server_audio_path = upload_file(server_ip, port, audio_file)

    server_output = f"/tmp/glitchbox_output_{int(time.time())}.mp4"

    # Start dummy journey to set input/output paths
    journey_data = {
        "prompts": ["_scheduler_", "_scheduler_"],
        "transition_frames": 999999,
        "hold_frames": 0,
        "loop": False,
        "duration": duration_seconds or 0,
        "output_path": server_output,
        "fps": fps,
    }
    if input_video:
        journey_data["input_video"] = input_video
    if server_audio_path:
        journey_data["audio_file"] = server_audio_path

    _post(server_ip, port, "/api/journey/start", journey_data)
    _post(server_ip, port, "/api/journey/stop")

    print("[HTTP] Enabling server prompt travel scheduler...")
    _post(server_ip, port, "/api/params/prompt_travel", {
        "enabled": True,
        "use_prompt_scheduler": True,
        "loop_prompts": True,
        "oscillate": True,
        "factor_increment": 0.005,
        "stabilize_duration": 0,
    })

    print(f"\nServer generating video to: {server_output}")
    print(f"Duration: {duration_seconds}s @ {fps}fps")
    print("Waiting for generation to complete... (Ctrl+C to abort)\n")

    _, actual_output = poll_until_done(server_ip, port)
    actual_output = actual_output or server_output

    # Download
    video_only = output.replace(".mp4", "_noaudio.mp4") if (audio_file or local_audio_path) else output
    if not download_file(server_ip, port, actual_output, video_only):
        return

    audio_source = local_audio_path or audio_file
    if audio_source and os.path.exists(audio_source):
        merge_audio(video_only, audio_source, output)
    else:
        if video_only != output:
            os.rename(video_only, output)
        print(f"Saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight HTTP client for GLITCHBOX (no dependencies)"
    )
    parser.add_argument("--server", default=SERVER_IP)
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    parser.add_argument("--prompts-file", type=str, default=None, help="Prompts file (one per line)")
    parser.add_argument("--transition-frames", type=int, default=120)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", "-o", type=str, default="prompt_journey.mp4")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--curation", type=int, default=None, help="Curation index (e.g. 21)")
    parser.add_argument("--pipe-index", type=int, default=None, help="LoRA pipe index")
    parser.add_argument("--input-video", type=str, default=None, help="Server-side video path or 'noise'")
    parser.add_argument("--init-image", type=str, default=None, help="Local init image to upload")
    parser.add_argument("--controlnet-scale", type=float, default=None, help="ControlNet scale")
    parser.add_argument("--temporal-coherence", type=float, default=None, help="Temporal coherence")
    parser.add_argument("--audio", type=str, default=None, help="Audio file for FFT effects")
    parser.add_argument("--strength", type=float, default=None, help="Denoising strength 0.0-1.0")
    parser.add_argument("--scheduler", action="store_true", help="Use server's prompt travel scheduler")
    parser.add_argument("--duration", type=float, default=30, help="Duration in seconds (--scheduler mode)")

    args = parser.parse_args()

    if args.scheduler:
        run_scheduler(
            server_ip=args.server, port=args.port,
            fps=args.fps, output=args.output, duration_seconds=args.duration,
            input_video=args.input_video, audio_file=args.audio,
            init_image=args.init_image, curation_index=args.curation,
            strength=args.strength, temporal_coherence=args.temporal_coherence,
            pipe_index=args.pipe_index, controlnet_scale=args.controlnet_scale,
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
            prompts=prompts, server_ip=args.server, port=args.port,
            fps=args.fps, output=args.output,
            transition_frames=args.transition_frames,
            hold_frames=args.hold_frames,
            loop=args.loop, input_video=args.input_video,
            init_image=args.init_image, audio_file=args.audio,
            curation_index=args.curation, strength=args.strength,
            temporal_coherence=args.temporal_coherence,
            pipe_index=args.pipe_index, controlnet_scale=args.controlnet_scale,
        )


if __name__ == "__main__":
    main()
