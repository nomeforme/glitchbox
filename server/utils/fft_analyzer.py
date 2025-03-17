import argparse
import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(parent_directory)
# Append the parent directory to sys.path
sys.path.append(parent_directory)

from modules.fft.stream_analyzer import Stream_Analyzer
import time
import pyaudio

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


def get_stream_analyzer(
    frequency_bins = 3,
    device_index = None,
    sleep_between_frames = True
):

    ear = Stream_Analyzer(
                    device = device_index,        # Pyaudio (portaudio) device index, defaults to first mic input
                    rate   = None,               # Audio samplerate, None uses the default source settings
                    FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                    updates_per_second  = 500,   # How often to read the audio stream for new data
                    smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                    n_frequency_bins = frequency_bins, # The FFT features are grouped in bins
                    visualize = 1,               # Visualize the FFT features with PyGame
                    verbose   = 1,    # Print running statistics (latency, fps, ...)
                    height    = 960,     # Height, in pixels, of the visualizer window,
                    window_ratio = 1  # Float ratio of the visualizer window. e.g. 24/9
                    )
    
    return ear

def run_FFT_analyzer():

    ear = Stream_Analyzer(
        device = 3,        # Pyaudio (portaudio) device index, defaults to first mic input
        rate   = None,               # Audio samplerate, None uses the default source settings
        FFT_window_size_ms  = 60,    # Window size used for the FFT transform
        updates_per_second  = 500,   # How often to read the audio stream for new data
        smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
        n_frequency_bins = 3, # The FFT features are grouped in bins
        visualize = 1,               # Visualize the FFT features with PyGame
        verbose   = 0,    # Print running statistics (latency, fps, ...)
        height    = 960,     # Height, in pixels, of the visualizer window,
        window_ratio = 1  # Float ratio of the visualizer window. e.g. 24/9
    )
    fps = 60  #How often to update the FFT features + display
    last_update = time.time()
    print("All ready, starting audio measurements now...")
    fft_samples = 0
    while True:
        if (time.time() - last_update) > (1./fps):
            last_update = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
            fft_samples += 1
            if fft_samples % 20 == 0:
               print(f"Got fft_features #{binned_fft} of shape {binned_fft.shape}")
               print(f"Got fftx_features #{binned_fftx} of shape {binned_fftx.shape}")
        # elif sleep_between_frames:
        #     time.sleep(((1./fps)-(time.time()-last_update)) * 0.99)

if __name__ == '__main__':
    list_audio_devices()
    run_FFT_analyzer()
