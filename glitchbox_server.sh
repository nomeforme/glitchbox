#!/bin/bash
# Entry point for the systemd service — runs the headless GLITCHBOX server.
# Edit the args below if you need to change the runtime configuration.

cd "$(dirname "$0")/server"
source .venv/bin/activate
exec python main.py \
    --pipeline img2imgSDXL_peft \
    --taesd \
    --headless \
    --use-latent-travel \
    --no-controlnet \
    --feedback-strength 0.2 \
    --use-lora-sound-control
