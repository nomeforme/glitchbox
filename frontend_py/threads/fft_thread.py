from PySide6.QtCore import QThread, Signal
import numpy as np
from modules.fft.stream_analyzer import Stream_Analyzer
from config import NUM_FFT_BINS, FFT_WINDOW_SIZE_MS, SMOOTHING_LENGTH_MS, EQUALIZER_STRENGTH, ROLLING_STATS_WINDOW_S

class FFTAnalyzerThread(QThread):
    """Thread for handling real-time FFT audio analysis"""
    
    # Signal emitted when new FFT data is available
    fft_data_updated = Signal(dict)
    
    def __init__(self, input_device_index=None):
        super().__init__()
        self.input_device_index = input_device_index
        self.analyzer = None
        self.running = False
        
    def run(self):
        """Main thread execution"""
        self.running = True
        
        try:
            # Initialize the FFT analyzer
            self.analyzer = Stream_Analyzer(
                device = self.input_device_index,
                rate   = 44100,               # Audio samplerate
                FFT_window_size_ms  = FFT_WINDOW_SIZE_MS,     # Window size used for the FFT transform
                updates_per_second  = 500,    # How often to read the audio stream for new data
                smoothing_length_ms = SMOOTHING_LENGTH_MS,     # Apply temporal smoothing to reduce noisy features
                n_frequency_bins = NUM_FFT_BINS,         # The FFT features are grouped in bins
                rolling_stats_window_s = ROLLING_STATS_WINDOW_S,
                equalizer_strength = EQUALIZER_STRENGTH,
                visualize = 0,                # Visualize the FFT features with PyGame
                verbose   = 0,                # Print running statistics (latency, fps, ...)
                height    = 480,              # Height, in pixels, of the visualizer window
                window_ratio = 1              # Float ratio of the visualizer window
            )
            
            print("[FFT] FFT analyzer initialized")
            
            # Process audio data while the thread is running
            while self.running:
                # Get audio features from FFT analyzer
                raw_fftx, raw_fft, binned_fftx, binned_fft = self.analyzer.get_audio_features()

                normalized_energies = binned_fft / self.analyzer.bin_mean_values
                
                # Prepare data for emission
                fft_data = {
                    "binned_fft": binned_fft.tolist() if isinstance(binned_fft, np.ndarray) else binned_fft,
                    "normalized_energies": normalized_energies.tolist() if isinstance(normalized_energies, np.ndarray) else normalized_energies,
                    # "raw_fft": raw_fft.tolist() if isinstance(raw_fft, np.ndarray) else raw_fft
                }
                
                # Emit the signal with the FFT data
                self.fft_data_updated.emit(fft_data)
                
                # Sleep a bit to avoid maxing out CPU
                self.msleep(20)  # 50Hz update rate
                
        except Exception as e:
            print(f"[FFT] Error in FFT analyzer thread: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Release the FFT analyzer resources if available
        if self.analyzer:
            try:
                self.analyzer.stop()
                print("[FFT] Analyzer resources released")
            except Exception as e:
                print(f"[FFT] Error releasing analyzer resources: {e}")
            finally:
                self.analyzer = None
                
    def stop(self):
        """Stop the FFT analyzer processing"""
        print("[FFT] Stopping FFT analyzer thread")
        self.running = False
        
        # Wait for the thread to finish with timeout
        if not self.wait(2000):  # 2 second timeout
            print("[FFT] Thread did not finish in time, terminating")
            self.terminate()