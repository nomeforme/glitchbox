#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "INFO: Resetting the v4l2loopback kernel module to ensure a clean state..."

# Unload and reload the module. Sudo is required.
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback

# Give the system a moment to make sure the device is ready.
sleep 0.5

echo "INFO: Module reset complete. Launching the Python script..."
echo "------------------------------------------------------------"

# Run your python script.
# Make sure you are in your virtual environment if you run this script directly.
# Or activate it here:
# source .venv/bin/activate
python depth_virtual_camera.py