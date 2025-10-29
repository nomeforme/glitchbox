from PySide6.QtCore import QThread, Signal
import cv2
import time
import numpy as np
import sys
import os

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DepthCameraThread(QThread):
    """Thread for handling depth camera capture"""
    frame_ready = Signal(np.ndarray)

    def __init__(self, device_index=42):
        super().__init__()
        self.running = False
        self.camera = None
        self.device_index = device_index

    def run(self):
        """Main thread loop for capturing depth camera frames"""
        try:
            self.camera = cv2.VideoCapture(self.device_index)
            if not self.camera.isOpened():
                print(f"[DepthCamera] Failed to open depth camera at index {self.device_index}")
                return

            self.running = True
            print(f"[DepthCamera] Started capturing from device {self.device_index}")
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("[DepthCamera] Failed to read frame")
                    break

                # Convert BGR to RGB for consistency
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)

                # Small delay to prevent tight loop
                time.sleep(0.01)

        except Exception as e:
            print(f"[DepthCamera] Error in depth camera thread: {e}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the depth camera thread"""
        print("[DepthCamera] Stopping depth camera thread")
        self.running = False

        # Add a timeout for waiting to prevent hanging
        if not self.wait(2000):  # Wait max 2 seconds for thread to finish
            print("[DepthCamera] Thread wait timed out, forcing termination")
            self.terminate()  # Force terminate if it doesn't finish in time

        # Only after the thread is done (or timeout), release the camera
        self.cleanup()

    def cleanup(self):
        """Clean up camera resources"""
        print("[DepthCamera] Cleaning up depth camera resources")
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        self.running = False

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.cleanup()
