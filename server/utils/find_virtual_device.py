#!/usr/bin/env python3
"""
Find Virtual Audio Device in PyAudio

This script helps you find the correct device index for the virtual audio device
that was created using the bash script.
"""

import pyaudio
import subprocess
import sys

def get_pulseaudio_sources():
    """Get PulseAudio sources using pactl"""
    try:
        result = subprocess.run(['pactl', 'list', 'short', 'sources'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []
    except FileNotFoundError:
        print("pactl not found")
        return []

def list_pyaudio_devices():
    """List all PyAudio devices with detailed information"""
    p = pyaudio.PyAudio()
    
    print("\nPyAudio Devices:")
    print("=" * 80)
    
    input_devices = []
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {device_info['index']}: {device_info['name']}")
        print(f"    Input Channels: {device_info['maxInputChannels']}")
        print(f"    Output Channels: {device_info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
        
        # Check if this is a PulseAudio device
        if 'pulse' in device_info['name'].lower():
            print(f"    *** PULSEAUDIO DEVICE ***")
            if device_info['maxInputChannels'] > 0:
                input_devices.append(i)
        
        print()
    
    p.terminate()
    return input_devices

def test_pulseaudio_device(device_index):
    """Test if a PulseAudio device can be opened"""
    p = pyaudio.PyAudio()
    
    try:
        # Try to open the device for input
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        print(f"✓ Successfully opened Device {device_index} for input")
        stream.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to open Device {device_index}: {e}")
        return False
    finally:
        p.terminate()

def find_virtual_device():
    """Find the virtual audio device"""
    print("🔍 Finding Virtual Audio Device...")
    print("=" * 50)
    
    # Get PulseAudio sources
    print("\n1. PulseAudio Sources:")
    print("-" * 30)
    pulse_sources = get_pulseaudio_sources()
    virtual_source = None
    
    for source in pulse_sources:
        if source and 'Virtual_Mixed_Audio' in source:
            print(f"✓ Found virtual source: {source}")
            virtual_source = source
            break
    
    if not virtual_source:
        print("✗ Virtual audio device not found in PulseAudio sources")
        print("Make sure you created it with: ./create_virtual_audio_device.sh <source1> <source2>")
        return
    
    # List PyAudio devices
    print("\n2. PyAudio Devices:")
    print("-" * 30)
    pulse_devices = list_pyaudio_devices()
    
    if not pulse_devices:
        print("✗ No PulseAudio devices found in PyAudio")
        return
    
    # Test PulseAudio devices
    print("\n3. Testing PulseAudio Devices:")
    print("-" * 30)
    
    working_devices = []
    for device_index in pulse_devices:
        if test_pulseaudio_device(device_index):
            working_devices.append(device_index)
    
    if working_devices:
        print(f"\n✅ Virtual audio device should be accessible through:")
        for device_index in working_devices:
            print(f"   Device {device_index} (PulseAudio interface)")
        
        print(f"\n📝 To use in your FFT analyzer, change the device index to: {working_devices[0]}")
        print(f"   Example: device = {working_devices[0]}")
    else:
        print("✗ No working PulseAudio devices found")

if __name__ == '__main__':
    find_virtual_device() 