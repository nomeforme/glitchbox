#!/usr/bin/env python3
"""
Audio Device Listing Tool
Lists all available audio input devices that PyAudio can detect.
"""

import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    
    print("Available Audio Input Devices:")
    print("=" * 50)
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        
        # Only show input devices (devices with input channels > 0)
        if device_info['maxInputChannels'] > 0:
            print(f"Device {i}: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
            print(f"  Host API: {p.get_host_api_info_by_index(device_info['hostApi'])['name']}")
            print()
    
    p.terminate()

if __name__ == "__main__":
    try:
        list_audio_devices()
    except Exception as e:
        print(f"Error listing audio devices: {e}") 