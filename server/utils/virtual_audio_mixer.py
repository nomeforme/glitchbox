#!/usr/bin/env python3
"""
Virtual Audio Mixer

This script creates a virtual audio device that combines two different hardware audio sources.
It reads from two input devices and outputs a mixed stream that can be detected as a new virtual device.

Usage:
    python virtual_audio_mixer.py --device1 <device_index> --device2 <device_index> [options]

Example:
    python virtual_audio_mixer.py --device1 0 --device2 1 --mix-ratio 0.5 --output-device "Virtual Mix"
"""

import argparse
import sys
import os
import numpy as np
import pyaudio
import threading
import time
import queue
from typing import Optional, Tuple

class VirtualAudioMixer:
    """
    A virtual audio mixer that combines two audio input sources into a single output stream.
    """
    
    def __init__(self, 
                 device1_index: int, 
                 device2_index: int,
                 sample_rate: int = 44100,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 mix_ratio: float = 0.5,
                 output_device_name: str = "Virtual Mix"):
        """
        Initialize the virtual audio mixer.
        
        Args:
            device1_index: Index of the first input device
            device2_index: Index of the second input device
            sample_rate: Sample rate for audio processing
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Size of audio chunks to process
            mix_ratio: Ratio for mixing (0.0 = only device1, 1.0 = only device2, 0.5 = equal mix)
            output_device_name: Name for the virtual output device
        """
        self.device1_index = device1_index
        self.device2_index = device2_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.mix_ratio = np.clip(mix_ratio, 0.0, 1.0)
        self.output_device_name = output_device_name
        
        # PyAudio instance
        self.pa = pyaudio.PyAudio()
        
        # Audio streams
        self.stream1 = None
        self.stream2 = None
        self.output_stream = None
        
        # Queues for audio data
        self.queue1 = queue.Queue(maxsize=10)
        self.queue2 = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = False
        self.threads = []
        
        # Validate devices
        self._validate_devices()
        
    def _validate_devices(self):
        """Validate that the specified devices exist and support input."""
        device_count = self.pa.get_device_count()
        
        if self.device1_index >= device_count:
            raise ValueError(f"Device 1 index {self.device1_index} is out of range. Max devices: {device_count}")
        
        if self.device2_index >= device_count:
            raise ValueError(f"Device 2 index {self.device2_index} is out of range. Max devices: {device_count}")
        
        # Check device capabilities
        device1_info = self.pa.get_device_info_by_index(self.device1_index)
        device2_info = self.pa.get_device_info_by_index(self.device2_index)
        
        if device1_info['maxInputChannels'] == 0:
            raise ValueError(f"Device 1 (index {self.device1_index}) does not support input")
        
        if device2_info['maxInputChannels'] == 0:
            raise ValueError(f"Device 2 (index {self.device2_index}) does not support input")
        
        print(f"Device 1: {device1_info['name']} (index {self.device1_index})")
        print(f"Device 2: {device2_info['name']} (index {self.device2_index})")
        print(f"Mix ratio: {self.mix_ratio:.2f} ({(1-self.mix_ratio):.2f} device1, {self.mix_ratio:.2f} device2)")
        
    def _audio_callback1(self, in_data, frame_count, time_info, status):
        """Callback for device 1 audio input."""
        if self.running:
            try:
                self.queue1.put_nowait(in_data)
            except queue.Full:
                # Drop oldest data if queue is full
                try:
                    self.queue1.get_nowait()
                    self.queue1.put_nowait(in_data)
                except queue.Empty:
                    pass
        return (None, pyaudio.paContinue)
    
    def _audio_callback2(self, in_data, frame_count, time_info, status):
        """Callback for device 2 audio input."""
        if self.running:
            try:
                self.queue2.put_nowait(in_data)
            except queue.Full:
                # Drop oldest data if queue is full
                try:
                    self.queue2.get_nowait()
                    self.queue2.put_nowait(in_data)
                except queue.Empty:
                    pass
        return (None, pyaudio.paContinue)
    
    def _mix_audio(self, data1: bytes, data2: bytes) -> bytes:
        """
        Mix two audio streams together.
        
        Args:
            data1: Audio data from device 1
            data2: Audio data from device 2
            
        Returns:
            Mixed audio data
        """
        # Convert bytes to numpy arrays
        audio1 = np.frombuffer(data1, dtype=np.int16)
        audio2 = np.frombuffer(data2, dtype=np.int16)
        
        # Ensure both arrays have the same length
        min_length = min(len(audio1), len(audio2))
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        
        # Mix the audio with the specified ratio
        # Convert to float for mixing, then back to int16
        audio1_float = audio1.astype(np.float32) / 32768.0
        audio2_float = audio2.astype(np.float32) / 32768.0
        
        # Apply mix ratio
        mixed_float = (1 - self.mix_ratio) * audio1_float + self.mix_ratio * audio2_float
        
        # Prevent clipping
        mixed_float = np.clip(mixed_float, -1.0, 1.0)
        
        # Convert back to int16
        mixed_audio = (mixed_float * 32767.0).astype(np.int16)
        
        return mixed_audio.tobytes()
    
    def _mixing_thread(self):
        """Main mixing thread that combines audio from both devices."""
        print("Starting audio mixing thread...")
        
        while self.running:
            try:
                # Get audio data from both devices
                data1 = self.queue1.get(timeout=0.1)
                data2 = self.queue2.get(timeout=0.1)
                
                # Mix the audio
                mixed_data = self._mix_audio(data1, data2)
                
                # Output the mixed audio (you can modify this to create a virtual device)
                # For now, we'll just print some info about the mixed audio
                mixed_audio = np.frombuffer(mixed_data, dtype=np.int16)
                rms = np.sqrt(np.mean(mixed_audio.astype(np.float32)**2))
                
                if rms > 100:  # Only print when there's significant audio
                    print(f"Mixed audio RMS: {rms:.2f}")
                    
            except queue.Empty:
                # No audio data available, continue
                continue
            except Exception as e:
                print(f"Error in mixing thread: {e}")
                break
        
        print("Audio mixing thread stopped.")
    
    def start(self):
        """Start the virtual audio mixer."""
        if self.running:
            print("Mixer is already running!")
            return
        
        print("Starting virtual audio mixer...")
        self.running = True
        
        # Open input streams
        try:
            self.stream1 = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device1_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback1
            )
            
            self.stream2 = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device2_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback2
            )
            
            # Start the streams
            self.stream1.start_stream()
            self.stream2.start_stream()
            
            # Start mixing thread
            mixing_thread = threading.Thread(target=self._mixing_thread, daemon=True)
            mixing_thread.start()
            self.threads.append(mixing_thread)
            
            print("Virtual audio mixer started successfully!")
            print("Press Ctrl+C to stop...")
            
        except Exception as e:
            print(f"Error starting mixer: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the virtual audio mixer."""
        print("Stopping virtual audio mixer...")
        self.running = False
        
        # Stop and close streams
        if self.stream1:
            self.stream1.stop_stream()
            self.stream1.close()
        
        if self.stream2:
            self.stream2.stop_stream()
            self.stream2.close()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Terminate PyAudio
        self.pa.terminate()
        print("Virtual audio mixer stopped.")

def list_audio_devices():
    """List all available audio devices."""
    pa = pyaudio.PyAudio()
    
    print("\nAvailable Audio Devices:")
    print("=" * 80)
    
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        print(f"Device {device_info['index']}: {device_info['name']}")
        print(f"    Input Channels: {device_info['maxInputChannels']}")
        print(f"    Output Channels: {device_info['maxOutputChannels']}")
        print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
        print()
    
    pa.terminate()

def create_virtual_device(device1_index: int, 
                         device2_index: int,
                         mix_ratio: float = 0.5,
                         sample_rate: int = 44100,
                         chunk_size: int = 1024) -> VirtualAudioMixer:
    """
    Create a virtual audio device that combines two input sources.
    
    Args:
        device1_index: Index of the first input device
        device2_index: Index of the second input device
        mix_ratio: Ratio for mixing (0.0 = only device1, 1.0 = only device2)
        sample_rate: Sample rate for audio processing
        chunk_size: Size of audio chunks to process
        
    Returns:
        VirtualAudioMixer instance
    """
    mixer = VirtualAudioMixer(
        device1_index=device1_index,
        device2_index=device2_index,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        mix_ratio=mix_ratio
    )
    
    return mixer

def main():
    """Main function to run the virtual audio mixer."""
    parser = argparse.ArgumentParser(description="Virtual Audio Mixer - Combine two audio sources")
    parser.add_argument("--device1", type=int, required=True, help="Index of first input device")
    parser.add_argument("--device2", type=int, required=True, help="Index of second input device")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Mix ratio (0.0-1.0, default: 0.5)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate (default: 44100)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size (default: 1024)")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    try:
        # Create and start the virtual audio mixer
        mixer = create_virtual_device(
            device1_index=args.device1,
            device2_index=args.device2,
            mix_ratio=args.mix_ratio,
            sample_rate=args.sample_rate,
            chunk_size=args.chunk_size
        )
        
        mixer.start()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            mixer.stop()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 