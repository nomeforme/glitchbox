from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2
import requests
import threading
import sys
import os

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_SCALE

class StreamThread(QThread):
    """Thread for handling MJPEG stream from server"""
    frame_received = Signal(np.ndarray)
    
    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self.running = False

    def run(self):
        """Process the MJPEG stream"""
        try:
            response = requests.get(self.stream_url, stream=True)
            bytes_buffer = bytes()
            self.running = True
            
            while self.running:
                chunk = response.raw.read(1024)
                if not chunk:
                    break
                    
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
                b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    
                    # Decode JPEG to numpy array
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    
                    if frame is not None:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_received.emit(frame_rgb)
                        
        except Exception as e:
            print(f"[Stream] Error processing stream: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the stream thread"""
        self.running = False
        self.wait()

class ProcessedDisplay(QWidget):
    """Widget to display processed image output"""
    
    def __init__(self, min_size=(640, 480)):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(*min_size)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # Stream handling
        self.stream_thread = None

    def start_stream(self, user_id: str, server_uri: str = "http://localhost:7860"):
        """Start receiving the MJPEG stream
        
        Args:
            user_id: The user ID for the stream
            server_uri: The server URI (default: http://localhost:7860)
        """
        if self.stream_thread and self.stream_thread.running:
            self.stop_stream()
            
        # Create and start stream thread
        stream_url = f"{server_uri}/api/stream/{user_id}"
        print(f"[Stream] Starting stream from: {stream_url}")
        self.stream_thread = StreamThread(stream_url)
        self.stream_thread.frame_received.connect(self.update_frame)
        self.stream_thread.start()

    def stop_stream(self):
        """Stop the stream thread"""
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return
            
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            DISPLAY_WIDTH * DISPLAY_SCALE, DISPLAY_HEIGHT * DISPLAY_SCALE, Qt.KeepAspectRatio))
        
        # Update FPS counter in status bar
        main_window = self.window()
        if main_window:
            main_window.status_bar.update_fps()

    def clear_display(self):
        """Clear the display and stop stream"""
        self.stop_stream()
        self.image_label.clear()

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_stream()