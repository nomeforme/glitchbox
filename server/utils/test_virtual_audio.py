#!/usr/bin/env python3
"""
Test Virtual Audio Device

This script demonstrates how to create and use a virtual audio device that combines two hardware sources.
It shows the audio devices before and after creating the virtual device.

Usage:
    python test_virtual_audio.py --device1 <index> --device2 <index> [options]
"""

import sys
import os
import time
import subprocess

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.virtual_audio_device import VirtualAudioDevice, list_audio_devices, list_pulseaudio_sinks
from utils.fft_analyzer import list_audio_devices as list_fft_devices

def test_virtual_audio_device(device1_index: int, device2_index: int, mix_ratio: float = 0.5):
    """
    Test the virtual audio device creation and usage.
    
    Args:
        device1_index: Index of the first input device
        device2_index: Index of the second input device
        mix_ratio: Ratio for mixing the two audio sources
    """
    
    print("=" * 80)
    print("VIRTUAL AUDIO DEVICE TEST")
    print("=" * 80)
    
    # Step 1: Show initial audio devices
    print("\n1. INITIAL AUDIO DEVICES:")
    print("-" * 40)
    list_audio_devices()
    
    # Step 2: Show PulseAudio sinks (if available)
    print("\n2. INITIAL PULSEAUDIO SINKS:")
    print("-" * 40)
    list_pulseaudio_sinks()
    
    # Step 3: Create virtual audio device
    print(f"\n3. CREATING VIRTUAL AUDIO DEVICE:")
    print("-" * 40)
    print(f"Combining device {device1_index} and device {device2_index}")
    print(f"Mix ratio: {mix_ratio:.2f}")
    
    try:
        # Create the virtual device
        virtual_device = VirtualAudioDevice(
            device1_index=device1_index,
            device2_index=device2_index,
            mix_ratio=mix_ratio,
            virtual_device_name="Test_Virtual_Mix"
        )
        
        # Start the virtual device
        virtual_device.start()
        
        # Step 4: Show audio devices after creating virtual device
        print("\n4. AUDIO DEVICES AFTER CREATING VIRTUAL DEVICE:")
        print("-" * 40)
        list_audio_devices()
        
        # Step 5: Show PulseAudio sinks after creating virtual device
        print("\n5. PULSEAUDIO SINKS AFTER CREATING VIRTUAL DEVICE:")
        print("-" * 40)
        list_pulseaudio_sinks()
        
        # Step 6: Test the virtual device for a few seconds
        print("\n6. TESTING VIRTUAL DEVICE (10 seconds):")
        print("-" * 40)
        print("Speak into both input devices to test the mixing...")
        
        for i in range(10, 0, -1):
            print(f"Testing for {i} more seconds...")
            time.sleep(1)
        
        # Step 7: Stop the virtual device
        print("\n7. STOPPING VIRTUAL DEVICE:")
        print("-" * 40)
        virtual_device.stop()
        
        # Step 8: Show final state
        print("\n8. FINAL AUDIO DEVICES:")
        print("-" * 40)
        list_audio_devices()
        
        print("\n9. FINAL PULSEAUDIO SINKS:")
        print("-" * 40)
        list_pulseaudio_sinks()
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Test failed!")
        return False
    
    return True

def interactive_device_selection():
    """Allow user to interactively select audio devices."""
    print("\nAvailable Audio Devices:")
    print("-" * 40)
    
    # Import here to avoid circular imports
    import pyaudio
    pa = pyaudio.PyAudio()
    
    input_devices = []
    
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_devices.append((i, device_info['name']))
            print(f"{len(input_devices)-1}: Device {i} - {device_info['name']}")
    
    pa.terminate()
    
    if len(input_devices) < 2:
        print("Error: Need at least 2 input devices to create a virtual device")
        return None, None
    
    try:
        print(f"\nSelect first device (0-{len(input_devices)-1}):")
        choice1 = int(input("> "))
        if choice1 < 0 or choice1 >= len(input_devices):
            print("Invalid choice")
            return None, None
        
        print(f"Select second device (0-{len(input_devices)-1}):")
        choice2 = int(input("> "))
        if choice2 < 0 or choice2 >= len(input_devices):
            print("Invalid choice")
            return None, None
        
        if choice1 == choice2:
            print("Error: Cannot select the same device twice")
            return None, None
        
        device1_index = input_devices[choice1][0]
        device2_index = input_devices[choice2][0]
        
        return device1_index, device2_index
        
    except ValueError:
        print("Invalid input")
        return None, None

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Virtual Audio Device")
    parser.add_argument("--device1", type=int, help="Index of first input device")
    parser.add_argument("--device2", type=int, help="Index of second input device")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Mix ratio (0.0-1.0, default: 0.5)")
    parser.add_argument("--interactive", action="store_true", help="Interactively select devices")
    parser.add_argument("--list-only", action="store_true", help="Only list devices and exit")
    
    args = parser.parse_args()
    
    if args.list_only:
        print("Available Audio Devices:")
        list_audio_devices()
        print("\nPulseAudio Sinks:")
        list_pulseaudio_sinks()
        return
    
    # Get device indices
    device1_index = args.device1
    device2_index = args.device2
    
    if args.interactive or (device1_index is None or device2_index is None):
        device1_index, device2_index = interactive_device_selection()
        if device1_index is None or device2_index is None:
            print("Failed to select devices")
            return
    
    # Run the test
    success = test_virtual_audio_device(device1_index, device2_index, args.mix_ratio)
    
    if success:
        print("\nVirtual audio device test completed successfully!")
        print("The virtual device was created, tested, and cleaned up properly.")
    else:
        print("\nVirtual audio device test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 