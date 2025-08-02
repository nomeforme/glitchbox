"""
Video Display Component for synchronized video and audio playback
Uses VLC for reliable video/audio sync and FFT analysis for effects
"""

import os
import sys
import time
import threading
import subprocess
import signal
import wave
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import QTimer, Signal, QObject, Qt
from PySide6.QtGui import QPixmap, QImage
import cv2

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT


class VideoDisplaySignals(QObject):
    """Signals for video display component"""
    frame_ready = Signal(np.ndarray)
    fft_data_ready = Signal(np.ndarray)
    video_finished = Signal()


class VideoDisplay(QWidget):
    """Video display widget with integrated audio/video playback and FFT analysis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = VideoDisplaySignals()
        
        # Video state
        self.video_path = None
        self.is_playing = False
        self.loop_enabled = False
        
        # Processes and threads
        self.ffplay_process = None
        self.frame_thread = None
        self.fft_thread = None
        self.stop_event = threading.Event()
        
        # Video properties
        self.video_fps = 30.0
        self.frame_width = DISPLAY_WIDTH
        self.frame_height = DISPLAY_HEIGHT
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the video display UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setFixedSize(self.frame_width, self.frame_height)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setText("Video Display\n(Select video file)")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layout.addWidget(self.video_label)
        self.setLayout(layout)
        
    def set_video_path(self, video_path):
        """Set the video file path"""
        self.video_path = video_path
        self.stop_playback()  # Stop any current playback
        
        # Get video info
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[VideoDisplay] Video loaded: {os.path.basename(video_path)}")
                print(f"[VideoDisplay] Resolution: {width}x{height} @ {self.video_fps:.1f}fps")
            cap.release()
            
    def set_loop_enabled(self, enabled):
        """Enable or disable video looping"""
        self.loop_enabled = enabled
        
    def start_playback(self):
        """Start video playback with audio"""
        if not self.video_path or not os.path.exists(self.video_path):
            print("[VideoDisplay] No valid video file to play")
            return False
            
        if self.is_playing:
            print("[VideoDisplay] Already playing")
            return True
            
        try:
            self.stop_event.clear()
            
            # Start ffplay for audio/video sync
            self.start_ffplay()
            
            # Start frame extraction thread
            self.frame_thread = threading.Thread(target=self._frame_extraction_worker, daemon=True)
            self.frame_thread.start()
            
            # Start FFT analysis thread
            self.fft_thread = threading.Thread(target=self._fft_worker, daemon=True)
            self.fft_thread.start()
            
            self.is_playing = True
            print("[VideoDisplay] Playback started")
            return True
            
        except Exception as e:
            print(f"[VideoDisplay] Error starting playback: {e}")
            return False
            
    def stop_playback(self):
        """Stop video playback"""
        if not self.is_playing:
            return
            
        print("[VideoDisplay] Stopping playback")
        self.is_playing = False
        self.stop_event.set()
        
        # Stop ffplay
        if self.ffplay_process:
            try:
                # First try gentle termination
                print(f"[VideoDisplay] Terminating ffplay process (PID: {self.ffplay_process.pid})")
                self.ffplay_process.terminate()
                self.ffplay_process.wait(timeout=2)
                print("[VideoDisplay] ffplay terminated gracefully")
            except subprocess.TimeoutExpired:
                print("[VideoDisplay] ffplay didn't terminate gracefully, killing...")
                # Kill the entire process group
                try:
                    if os.name != 'nt':  # Unix-like systems
                        os.killpg(os.getpgid(self.ffplay_process.pid), signal.SIGKILL)
                    else:  # Windows
                        self.ffplay_process.kill()
                    self.ffplay_process.wait(timeout=1)
                    print("[VideoDisplay] ffplay killed successfully")
                except Exception as kill_error:
                    print(f"[VideoDisplay] Error killing ffplay: {kill_error}")
            except Exception as e:
                print(f"[VideoDisplay] Error stopping ffplay: {e}")
            finally:
                self.ffplay_process = None
                
        # Wait for threads to finish
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1)
            
        if self.fft_thread and self.fft_thread.is_alive():
            self.fft_thread.join(timeout=1)
            
        print("[VideoDisplay] Playback stopped")
        
    def start_ffplay(self):
        """Start ffplay for synchronized audio/video playback"""
        try:
            ffplay_cmd = [
                'ffplay',
                '-i', self.video_path,
                '-nodisp',  # No video window (we handle display separately)
                '-autoexit',  # Exit when finished
                '-hide_banner',  # Reduce console output
                '-loglevel', 'error'  # Only show errors
            ]
            
            if self.loop_enabled:
                ffplay_cmd.extend(['-loop', '0'])  # Loop infinitely
                
            print(f"[VideoDisplay] Starting ffplay: {' '.join(ffplay_cmd)}")
            self.ffplay_process = subprocess.Popen(
                ffplay_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=None if os.name == 'nt' else os.setsid  # New process group on Unix
            )
            
        except Exception as e:
            print(f"[VideoDisplay] Error starting ffplay: {e}")
            self.ffplay_process = None
            
    def _frame_extraction_worker(self):
        """Worker thread for extracting video frames"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("[VideoDisplay] Failed to open video for frame extraction")
                return
                
            frame_delay = 1.0 / self.video_fps
            start_time = time.time()
            frame_count = 0
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # End of video
                    if self.loop_enabled:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        continue
                    else:
                        self.signals.video_finished.emit()
                        break
                        
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.signals.frame_ready.emit(rgb_frame)
                
                # Timing control
                frame_count += 1
                target_time = start_time + (frame_count * frame_delay)
                current_time = time.time()
                sleep_time = target_time - current_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            cap.release()
            
        except Exception as e:
            print(f"[VideoDisplay] Error in frame extraction: {e}")
            
    def _fft_worker(self):
        """Worker thread for FFT analysis"""
        try:
            # Extract audio to temporary file
            temp_audio = self._extract_audio_to_temp()
            if not temp_audio:
                return
                
            # Open audio file
            wf = wave.open(temp_audio, 'rb')
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            
            # Calculate chunk size for FFT
            fft_size = 1024
            samples_per_frame = int(sample_rate / self.video_fps)
            
            print(f"[VideoDisplay] FFT analysis started - {sample_rate}Hz, {channels}ch")
            
            while not self.stop_event.is_set():
                # Read audio data
                frames = wf.readframes(samples_per_frame)
                if not frames:
                    # End of audio
                    if self.loop_enabled:
                        wf.rewind()
                        continue
                    else:
                        break
                        
                # Convert to numpy array
                audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Handle stereo by taking left channel
                if channels == 2:
                    audio_data = audio_data[::2]
                    
                # Pad or truncate to FFT size
                if len(audio_data) >= fft_size:
                    audio_chunk = audio_data[:fft_size]
                else:
                    audio_chunk = np.pad(audio_data, (0, fft_size - len(audio_data)), 'constant')
                    
                # Compute FFT
                fft_data = np.abs(np.fft.rfft(audio_chunk.astype(np.float32)))
                self.signals.fft_data_ready.emit(fft_data)
                
                # Sync with video frame rate
                time.sleep(1.0 / self.video_fps)
                
            wf.close()
            
            # Clean up temp file
            try:
                os.remove(temp_audio)
            except:
                pass
                
        except Exception as e:
            print(f"[VideoDisplay] Error in FFT analysis: {e}")
            
    def _extract_audio_to_temp(self):
        """Extract audio from video to temporary WAV file"""
        try:
            temp_audio = f"/tmp/video_audio_{int(time.time())}.wav"
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', self.video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '2',  # Stereo
                '-hide_banner',
                '-loglevel', 'error',
                temp_audio
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(temp_audio):
                print(f"[VideoDisplay] Audio extracted to: {temp_audio}")
                return temp_audio
            else:
                print(f"[VideoDisplay] FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"[VideoDisplay] Error extracting audio: {e}")
            return None
            
    def cleanup(self, force=False):
        """Clean up all resources"""
        print(f"[VideoDisplay] Cleaning up resources (force={force})")
        self.stop_playback()
        
        # Additional cleanup: force kill any remaining ffplay processes
        self._force_cleanup_ffplay_processes()
        
        # If force cleanup, be more aggressive
        if force:
            print("[VideoDisplay] Force cleanup - killing all ffplay processes immediately")
            try:
                subprocess.run(['pkill', '-9', 'ffplay'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             timeout=1)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
    def _force_cleanup_ffplay_processes(self):
        """Force cleanup of any remaining ffplay processes"""
        try:
            import psutil
            # Find and kill any ffplay processes that might be running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'ffplay' in proc.info['name']:
                        print(f"[VideoDisplay] Found lingering ffplay process: PID {proc.info['pid']}")
                        proc.kill()
                        proc.wait(timeout=1)
                        print(f"[VideoDisplay] Killed ffplay process: PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except ImportError:
            # psutil not available, fallback to killall command
            try:
                subprocess.run(['killall', 'ffplay'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             timeout=2)
                print("[VideoDisplay] Executed killall ffplay as fallback")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
    def __del__(self):
        """Destructor"""
        self.cleanup()