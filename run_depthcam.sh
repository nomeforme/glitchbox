#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "INFO: Setting up v4l2loopback kernel module for virtual camera..."

# Unload the module if already loaded
sudo modprobe -r v4l2loopback 2>/dev/null || true

# Load the module with specific video device number (42) and label
sudo modprobe v4l2loopback video_nr=42 card_label="RealSense Depth" exclusive_caps=1

# Give the system a moment to make sure the device is ready.
sleep 0.5

# Verify the device was created
if [ ! -e /dev/video42 ]; then
    echo "ERROR: Failed to create /dev/video42"
    echo "Please check if v4l2loopback kernel module is installed:"
    echo "  sudo apt install v4l2loopback-dkms"
    exit 1
fi

echo "INFO: Virtual camera device /dev/video42 ready"

echo "INFO: Module reset complete. Launching the Python script..."
echo "------------------------------------------------------------"

# Run your python script.
# Make sure you are in your virtual environment if you run this script directly.
# Or activate it here:
# source .venv/bin/activate
python frontend_py/utils/depthcam/depth_virtual_camera.py