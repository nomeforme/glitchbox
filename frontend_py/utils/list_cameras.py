#!/usr/bin/env python3

import cv2
import subprocess
import re
import sys
import os
from typing import List, Dict, Tuple
import pyaudio
from config import MAX_CAMERA_INDEX

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = []
    
    print("\nAvailable Audio Devices:")
    print("-----------------------")
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {device_info['index']}: {device_info['name']}")
        print(f"    Input Channels: {device_info['maxInputChannels']}")
        print(f"    Output Channels: {device_info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
        print()
    
    p.terminate()

def get_video_devices() -> List[str]:
    """
    Get list of video devices by checking /dev/video*
    Returns a list of video device paths
    """
    devices = []
    for i in range(MAX_CAMERA_INDEX):
        path = f"/dev/video{i}"
        if os.path.exists(path):
            devices.append(path)
    return devices

def get_device_info(device_path: str) -> Dict[str, str]:
    """
    Get information about a video device
    Returns a dictionary with device info
    """
    info = {
        'path': device_path,
        'name': 'Unknown Device',
        'capabilities': []
    }
    
    # Try to get device name from sysfs
    try:
        device_num = device_path.split('video')[-1]
        sysfs_path = f"/sys/class/video4linux/video{device_num}/name"
        if os.path.exists(sysfs_path):
            with open(sysfs_path, 'r') as f:
                info['name'] = f.read().strip()
    except Exception:
        pass
    
    # Try to get capabilities using v4l2-ctl if available
    try:
        result = subprocess.run(['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
                              capture_output=True,
                              text=True,
                              check=False)
        if result.returncode == 0:
            info['capabilities'] = result.stdout.split('\n')
    except FileNotFoundError:
        pass
    
    return info

def test_camera(index: int) -> Tuple[bool, str, Dict]:
    """
    Test if a camera index can be opened
    Returns (success, message, device_info)
    """
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False, "Failed to open camera", {}
            
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "Failed to read frame", {}
            
        device_info = {
            'resolution': f"{int(width)}x{int(height)}",
            'fps': fps
        }
        return True, "Camera working", device_info
    except Exception as e:
        return False, str(e), {}

def main():
    print("Detecting cameras...\n")
    
    # First try to find video devices
    video_devices = get_video_devices()
    
    if not video_devices:
        print("No video devices found in /dev/video*")
        print("\nTrying OpenCV camera detection...")

        # Fallback to OpenCV detection
        for i in range(MAX_CAMERA_INDEX):
            success, message, device_info = test_camera(i)
            if success:
                print(f"\nCamera found at index {i}")
                print(f"Resolution: {device_info.get('resolution', 'Unknown')}")
                print(f"FPS: {device_info.get('fps', 'Unknown')}")
            elif "Failed to open camera" not in message:  # Only show errors that aren't just "no camera"
                print(f"Camera at index {i}: {message}")
    else:
        print(f"Found {len(video_devices)} video device(s):")
        for device_path in video_devices:
            print(f"\nDevice: {device_path}")
            
            # Get device info
            device_info = get_device_info(device_path)
            print(f"Name: {device_info['name']}")
            
            # Extract index from path
            match = re.search(r'/dev/video(\d+)', device_path)
            if match:
                index = int(match.group(1))
                success, message, camera_info = test_camera(index)
                print(f"Status: {'Working' if success else 'Not working'}")
                if success:
                    print(f"Resolution: {camera_info.get('resolution', 'Unknown')}")
                    print(f"FPS: {camera_info.get('fps', 'Unknown')}")
                if not success:
                    print(f"Error: {message}")
                
                if device_info['capabilities']:
                    print("\nSupported formats:")
                    for cap in device_info['capabilities']:
                        if cap.strip():
                            print(f"  {cap.strip()}")

    list_audio_devices()



if __name__ == "__main__":
    main()
