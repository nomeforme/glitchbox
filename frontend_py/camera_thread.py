from PySide6.QtCore import QThread, Signal
import cv2
import time
import numpy as np

class CameraThread(QThread):
    """Thread for handling camera capture"""
    frame_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None

    def run(self):
        """Main thread loop for capturing camera frames"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("[Camera] Failed to open camera")
                return
                
            self.running = True
            while self.running:
                ret, frame = self.camera.read()
                if ret:
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_ready.emit(rgb_frame)
                else:
                    print("[Camera] Failed to read frame")
                    break
                    
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
        if self.camera is not None:
            self.camera.release()
        self.wait()  # Wait for thread to finish

    def cleanup(self):
        """Clean up camera resources"""
        print("[Camera] Cleaning up camera resources")
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.running = False

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.cleanup()