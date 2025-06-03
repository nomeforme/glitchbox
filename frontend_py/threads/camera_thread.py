from PySide6.QtCore import QThread, Signal
import cv2
import time
import numpy as np
import sys
import os

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, CAMERA_DEVICE_INDEX

class CameraThread(QThread):
    """Thread for handling camera capture"""
    frame_ready = Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.device_index = CAMERA_DEVICE_INDEX

    def run(self):
        """Main thread loop for capturing camera frames"""
        try:
            self.camera = cv2.VideoCapture(self.device_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
            if not self.camera.isOpened():
                print("[Camera] Failed to open camera")
                return
                
            self.running = True
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("[Camera] Failed to read frame")
                    break
                    
                # # Resize frame to 640x480
                # frame = cv2.resize(frame, (854, 480))
                    
                # Convert BGR to RGB for display
                # TODO: Remove this once we have a working RGB frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)
                    
                # Small delay to prevent tight loop
                time.sleep(0.01)
                
        except Exception as e:
            print(f"[Camera] Error in camera thread: {e}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the camera thread"""
        print("[Camera] Stopping camera thread")
        self.running = False
        
        # First set running to false to break out of the loop
        self.running = False
        
        # Add a timeout for waiting to prevent hanging
        if not self.wait(2000):  # Wait max 2 seconds for thread to finish
            print("[Camera] Thread wait timed out, forcing termination")
            self.terminate()  # Force terminate if it doesn't finish in time
            
        # Only after the thread is done (or timeout), release the camera
        self.cleanup()

    def cleanup(self):
        """Clean up camera resources"""
        print("[Camera] Cleaning up camera resources")
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        self.running = False

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.cleanup()