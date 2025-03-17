#!/usr/bin/env python3
"""
Audio Device Lister

This script lists all available audio input and output devices 
using both PyAudio and sounddevice libraries for comparison.
"""

import pyaudio
import sounddevice as sd
import numpy as np

def list_pyaudio_devices():
    """List all audio devices using PyAudio"""
    print("\n" + "="*80)
    print("AUDIO DEVICES DETECTED BY PYAUDIO")
    print("="*80)
    
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    input_devices = []
    output_devices = []
    
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info.get('name')
        max_input_channels = device_info.get('maxInputChannels')
        max_output_channels = device_info.get('maxOutputChannels')
        default_sample_rate = device_info.get('defaultSampleRate')
        
        if max_input_channels > 0:
            input_devices.append((i, device_name, max_input_channels, default_sample_rate))
        
        if max_output_channels > 0:
            output_devices.append((i, device_name, max_output_channels, default_sample_rate))
    
    # List input devices
    print("\nINPUT DEVICES:")
    print("-" * 80)
    if input_devices:
        print(f"{'ID':<5} | {'Device Name':<40} | {'Channels':<8} | {'Sample Rate'}")
        print("-" * 80)
        for idx, name, channels, rate in input_devices:
            print(f"{idx:<5} | {name[:40]:<40} | {channels:<8} | {int(rate)}")
    else:
        print("No input devices found")
    
    # List output devices
    print("\nOUTPUT DEVICES:")
    print("-" * 80)
    if output_devices:
        print(f"{'ID':<5} | {'Device Name':<40} | {'Channels':<8} | {'Sample Rate'}")
        print("-" * 80)
        for idx, name, channels, rate in output_devices:
            print(f"{idx:<5} | {name[:40]:<40} | {channels:<8} | {int(rate)}")
    else:
        print("No output devices found")
    
    # Get default devices
    default_input = p.get_default_input_device_info()
    default_output = p.get_default_output_device_info()
    
    print("\nDEFAULT DEVICES:")
    print(f"Default Input Device: ID {default_input['index']} - {default_input['name']}")
    print(f"Default Output Device: ID {default_output['index']} - {default_output['name']}")
    
    p.terminate()

def list_sounddevice_devices():
    """List all audio devices using sounddevice"""
    print("\n" + "="*80)
    print("AUDIO DEVICES DETECTED BY SOUNDDEVICE")
    print("="*80)
    
    devices = sd.query_devices()
    
    input_devices = []
    output_devices = []
    
    for i, device in enumerate(devices):
        device_name = device['name']
        input_channels = device['max_input_channels']
        output_channels = device['max_output_channels']
        sample_rate = device['default_samplerate']
        
        # Check if it's an input device
        if input_channels > 0:
            input_devices.append((i, device_name, input_channels, sample_rate))
        
        # Check if it's an output device
        if output_channels > 0:
            output_devices.append((i, device_name, output_channels, sample_rate))
    
    # List input devices
    print("\nINPUT DEVICES:")
    print("-" * 80)
    if input_devices:
        print(f"{'ID':<5} | {'Device Name':<40} | {'Channels':<8} | {'Sample Rate'}")
        print("-" * 80)
        for idx, name, channels, rate in input_devices:
            print(f"{idx:<5} | {name[:40]:<40} | {channels:<8} | {int(rate)}")
    else:
        print("No input devices found")
    
    # List output devices
    print("\nOUTPUT DEVICES:")
    print("-" * 80)
    if output_devices:
        print(f"{'ID':<5} | {'Device Name':<40} | {'Channels':<8} | {'Sample Rate'}")
        print("-" * 80)
        for idx, name, channels, rate in output_devices:
            print(f"{idx:<5} | {name[:40]:<40} | {channels:<8} | {int(rate)}")
    else:
        print("No output devices found")
    
    # Get default devices
    try:
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        
        print("\nDEFAULT DEVICES:")
        print(f"Default Input Device: ID {default_input['index']} - {default_input['name']}")
        print(f"Default Output Device: ID {default_output['index']} - {default_output['name']}")
    except Exception as e:
        print(f"\nCould not determine default devices: {e}")

if __name__ == "__main__":
    try:
        list_pyaudio_devices()
    except Exception as e:
        print(f"Error listing PyAudio devices: {e}")
    
    try:
        list_sounddevice_devices()
    except Exception as e:
        print(f"Error listing sounddevice devices: {e}")
    
    print("\nNOTE: If you see different devices listed by PyAudio and sounddevice,")
    print("this may explain audio routing issues in your application.")