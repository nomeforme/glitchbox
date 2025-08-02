from PySide6.QtCore import QThread, Signal
import numpy as np
import time
import sys
import os
import subprocess
import tempfile
import wave
import pyaudio
from config import NUM_FFT_BINS

class VideoAudioThread(QThread):
    """Thread for handling audio extraction from video files and FFT analysis"""
    
    # Signal emitted when new FFT data is available
    fft_data_updated = Signal(dict)
    audio_finished = Signal()
    
    def __init__(self, video_path=None, loop=True, fps=None):
        super().__init__()
        self.video_path = video_path
        self.loop = loop
        self.target_fps = fps
        self.running = False
        self.temp_audio_file = None
        self.audio_stream = None
        self.playback_stream = None
        self.pa = None
        self.video_fps = 30.0
        self.audio_duration = 0.0
        self.current_time = 0.0
        self.audio_playback_enabled = False
        
    def set_video_path(self, video_path):
        """Set the video file path"""
        self.video_path = video_path
        
    def set_loop(self, loop):
        """Set whether to loop the audio"""
        self.loop = loop
        
    def set_fps(self, fps):
        """Set target playback FPS"""
        self.target_fps = fps
        
    def set_audio_playback(self, enabled):
        """Enable or disable audio playback"""
        self.audio_playback_enabled = enabled
        if enabled:
            # If audio file is ready and no playback stream exists, create it
            if hasattr(self, 'temp_audio_file') and self.temp_audio_file and not self.playback_stream:
                self.create_playback_stream()
        elif not enabled and self.playback_stream:
            self.stop_playback_stream()
        
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file using ffmpeg"""
        if not video_path or not os.path.exists(video_path):
            print(f"[VideoAudio] Video file not found: {video_path}")
            return None
            
        try:
            # Create temporary file for audio
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            self.temp_audio_file = temp_path
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_path
            ]
            
            print(f"[VideoAudio] Extracting audio from video: {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[VideoAudio] Error extracting audio: {result.stderr}")
                return None
                
            # Get video FPS for synchronization
            fps_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
                video_path
            ]
            
            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
            if fps_result.returncode == 0:
                try:
                    fps_str = fps_result.stdout.strip()
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        self.video_fps = num / den if den > 0 else 30.0
                    else:
                        self.video_fps = float(fps_str)
                except:
                    self.video_fps = 30.0
            else:
                self.video_fps = 30.0
                
            # Get audio duration
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', temp_path
            ]
            
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
            if duration_result.returncode == 0:
                try:
                    self.audio_duration = float(duration_result.stdout.strip())
                except:
                    self.audio_duration = 0.0
                    
            print(f"[VideoAudio] Audio extracted successfully. Duration: {self.audio_duration:.2f}s, Video FPS: {self.video_fps:.2f}")
            return temp_path
            
        except Exception as e:
            print(f"[VideoAudio] Error extracting audio: {e}")
            return None
            
    def create_audio_stream(self, audio_file_path):
        """Create an audio stream from the extracted audio file"""
        try:
            self.pa = pyaudio.PyAudio()
            
            # Open the WAV file
            wf = wave.open(audio_file_path, 'rb')
            
            # Create a virtual audio device that reads from the WAV file
            # We'll create a custom stream reader that feeds the audio data to the FFT analyzer
            self.wf = wf
            self.audio_data = wf.readframes(wf.getnframes())
            self.audio_samples = np.frombuffer(self.audio_data, dtype=np.int16)
            self.sample_rate = wf.getframerate()
            self.current_sample = 0
            self.samples_per_chunk = int(self.sample_rate / 50)  # 50Hz update rate
            
            return True
            
        except Exception as e:
            print(f"[VideoAudio] Error creating audio stream: {e}")
            return False
            
    def create_playback_stream(self):
        """Create an audio playback stream for hearing the video audio"""
        try:
            if not self.pa or not self.temp_audio_file:
                print("[VideoAudio] Cannot create playback stream - missing PA or audio file")
                return
                
            self.playback_wf = wave.open(self.temp_audio_file, 'rb')
            
            # Get audio properties
            sample_width = self.playback_wf.getsampwidth()
            channels = self.playback_wf.getnchannels()
            framerate = self.playback_wf.getframerate()
            
            # Create playback stream
            self.playback_stream = self.pa.open(
                format=self.pa.get_format_from_width(sample_width),
                channels=channels,
                rate=framerate,
                output=True,
                frames_per_buffer=1024
            )
            
            # Start the stream
            self.playback_stream.start_stream()
            
            print("[VideoAudio] Audio playback enabled")
            
        except Exception as e:
            print(f"[VideoAudio] Error creating playback stream: {e}")
            
    def stop_playback_stream(self):
        """Stop the audio playback stream"""
        try:
            if self.playback_stream:
                self.playback_stream.stop_stream()
                self.playback_stream.close()
                self.playback_stream = None
                
            if hasattr(self, 'playback_wf') and self.playback_wf:
                self.playback_wf.close()
                self.playback_wf = None
            
        except Exception as e:
            print(f"[VideoAudio] Error stopping playback stream: {e}")
            
    def run(self):
        """Main thread execution"""
        if not self.video_path or not os.path.exists(self.video_path):
            print(f"[VideoAudio] Video file not found: {self.video_path}")
            return
            
        try:
            # Extract audio from video
            audio_file = self.extract_audio_from_video(self.video_path)
            if not audio_file:
                print("[VideoAudio] Failed to extract audio from video")
                return
                
            # Create audio stream
            if not self.create_audio_stream(audio_file):
                print("[VideoAudio] Failed to create audio stream")
                return
                
            print("[VideoAudio] Video audio stream created")
            
            # Automatically enable audio playback
            self.audio_playback_enabled = True
            self.create_playback_stream()
            
            self.running = True
            
            # Process audio data while the thread is running
            while self.running:
                # Get current audio chunk
                if self.current_sample >= len(self.audio_samples):
                    if self.loop:
                        self.current_sample = 0
                    else:
                        self.audio_finished.emit()
                        break
                
                # Extract audio chunk
                end_sample = min(self.current_sample + self.samples_per_chunk, len(self.audio_samples))
                audio_chunk = self.audio_samples[self.current_sample:end_sample]
                
                # Handle audio playback if enabled
                if self.audio_playback_enabled and self.playback_stream and hasattr(self, 'playback_wf'):
                    try:
                        # Read audio data for playback
                        playback_data = self.playback_wf.readframes(self.samples_per_chunk)
                        if playback_data:
                            self.playback_stream.write(playback_data)
                        else:
                            # End of playback file
                            if self.loop:
                                self.playback_wf.rewind()
                    except Exception as e:
                        print(f"[VideoAudio] Error during playback: {e}")
                
                # Pad if necessary for FFT
                if len(audio_chunk) < self.samples_per_chunk:
                    audio_chunk = np.pad(audio_chunk, (0, self.samples_per_chunk - len(audio_chunk)), 'constant')
                
                # Convert to float and normalize for FFT
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                
                # Perform FFT analysis
                fft_result = np.fft.fft(audio_chunk)
                fft_magnitude = np.abs(fft_result[:len(fft_result)//2])
                
                # Create frequency bins (simplified version)
                num_bins = NUM_FFT_BINS
                bin_size = len(fft_magnitude) // num_bins
                binned_fft = []
                
                for i in range(num_bins):
                    start_idx = i * bin_size
                    end_idx = start_idx + bin_size
                    if i == num_bins - 1:  # Last bin gets remaining samples
                        end_idx = len(fft_magnitude)
                    bin_energy = np.mean(fft_magnitude[start_idx:end_idx])
                    binned_fft.append(float(bin_energy))
                
                # Prepare data for emission
                fft_data = {
                    "binned_fft": binned_fft,
                    "normalized_energies": binned_fft,  # Simplified for now
                }
                
                # Emit the signal with the FFT data
                self.fft_data_updated.emit(fft_data)
                
                # Update sample position
                self.current_sample += self.samples_per_chunk
                
                # Sleep to maintain timing
                self.msleep(20)  # 50Hz update rate
                
        except Exception as e:
            print(f"[VideoAudio] Error in video audio thread: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Stop audio stream
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
            
        # Stop playback stream
        self.stop_playback_stream()
            
        # Close WAV file
        if hasattr(self, 'wf') and self.wf:
            try:
                self.wf.close()
            except:
                pass
            self.wf = None
            
        # Terminate PyAudio
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None
            
        # Clean up audio data
        if hasattr(self, 'audio_samples'):
            self.audio_samples = None
        if hasattr(self, 'audio_data'):
            self.audio_data = None
                
        # Clean up temporary audio file
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
                print("[VideoAudio] Temporary audio file removed")
            except Exception as e:
                print(f"[VideoAudio] Error removing temporary file: {e}")
            self.temp_audio_file = None
                
    def stop(self):
        """Stop the video audio thread"""
        print("[VideoAudio] Stopping video audio thread")
        self.running = False
        
        # Wait for the thread to finish with timeout
        if not self.wait(2000):  # 2 second timeout
            print("[VideoAudio] Thread did not finish in time, terminating")
            self.terminate()
            
        self.cleanup()
        
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup() 