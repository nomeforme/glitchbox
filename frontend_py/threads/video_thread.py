from PySide6.QtCore import QThread, Signal
import cv2
import time
import numpy as np
import sys
import os

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT

class VideoThread(QThread):
    """Thread for handling video file playback"""
    frame_ready = Signal(np.ndarray)
    video_finished = Signal()
    
    def __init__(self, video_path=None, loop=True, fps=None):
        super().__init__()
        self.running = False
        self.video_path = video_path
        self.loop = loop
        self.target_fps = fps
        self.cap = None
        self.total_frames = 0
        self.current_frame_number = 0
        self.video_fps = 30.0  # Default FPS

    def set_video_path(self, video_path):
        """Set the video file path"""
        self.video_path = video_path

    def set_loop(self, loop):
        """Set whether to loop the video"""
        self.loop = loop

    def set_fps(self, fps):
        """Set target playback FPS"""
        self.target_fps = fps

    def get_video_info(self):
        """Get video information"""
        if not self.video_path or not os.path.exists(self.video_path):
            return None
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
            
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info

    def run(self):
        """Main thread loop for reading video frames"""
        if not self.video_path or not os.path.exists(self.video_path):
            print(f"[Video] Video file not found: {self.video_path}")
            return
            
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"[Video] Failed to open video file: {self.video_path}")
                return
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0:
                self.video_fps = 30.0  # Default if FPS is not available
                
            # Set target FPS (use video FPS if not specified)
            target_fps = self.target_fps if self.target_fps else self.video_fps
            frame_delay = 1.0 / target_fps
            
            print(f"[Video] Playing video: {self.video_path}")
            print(f"[Video] Total frames: {self.total_frames}, Video FPS: {self.video_fps:.2f}, Target FPS: {target_fps:.2f}")
            
            self.running = True
            self.current_frame_number = 0
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video reached
                    if self.loop:
                        print("[Video] End of video reached, looping...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_number = 0
                        continue
                    else:
                        print("[Video] End of video reached")
                        self.video_finished.emit()
                        break
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)
                
                self.current_frame_number += 1
                
                # Sleep to maintain target FPS
                time.sleep(frame_delay)
                
        except Exception as e:
            print(f"[Video] Error in video thread: {e}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the video thread"""
        print("[Video] Stopping video thread")
        self.running = False
        
        # Wait for the thread to finish with timeout
        if not self.wait(2000):  # Wait max 2 seconds for thread to finish
            print("[Video] Thread wait timed out, forcing termination")
            self.terminate()
            
        self.cleanup()

    def pause(self):
        """Pause video playback"""
        self.running = False

    def resume(self):
        """Resume video playback"""
        if self.cap and self.cap.isOpened():
            self.running = True

    def seek_to_frame(self, frame_number):
        """Seek to a specific frame number"""
        if self.cap and self.cap.isOpened():
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_number = frame_number

    def seek_to_time(self, seconds):
        """Seek to a specific time in seconds"""
        if self.cap and self.cap.isOpened():
            frame_number = int(seconds * self.video_fps)
            self.seek_to_frame(frame_number)

    def cleanup(self):
        """Clean up video resources"""
        print("[Video] Cleaning up video resources")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.running = False

    def __del__(self):
        """Destructor to ensure video is released"""
        self.cleanup() 