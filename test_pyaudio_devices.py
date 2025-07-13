#!/usr/bin/env python3
import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    
    print("\nAvailable PyAudio Devices:")
    print("=" * 80)
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {device_info['index']}: {device_info['name']}")
        print(f"    Input Channels: {device_info['maxInputChannels']}")
        print(f"    Output Channels: {device_info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
        print()
    
    p.terminate()

if __name__ == '__main__':
    list_audio_devices() 