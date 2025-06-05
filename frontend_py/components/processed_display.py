from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2
import requests
import threading
import sys
import os
import zmq
import time
from dotenv import load_dotenv

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_SCALE
from .fullscreen_window import FullscreenWindow

# Load environment variables
load_dotenv(override=True)

# Server configuration
DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_ZMQ_PORT = os.getenv("DEFAULT_SERVER_ZMQ_PORT")

print(f"DEFAULT_SERVER_HOST: {DEFAULT_SERVER_HOST}")
print(f"DEFAULT_SERVER_ZMQ_PORT: {DEFAULT_SERVER_ZMQ_PORT}")

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

class ZMQThread(QThread):
    """Thread for handling ZMQ image stream"""
    frame_received = Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        print("[ZMQ] Initializing ZMQ context and socket...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        zmq_address = f"tcp://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_ZMQ_PORT}"
        print(f"[ZMQ] Attempting to connect to {zmq_address}...")
        try:
            self.socket.connect(zmq_address)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
            print("[ZMQ] Socket connected and subscribed")
        except Exception as e:
            print(f"[ZMQ] Failed to connect to ZMQ socket: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Process the ZMQ stream"""
        try:
            print("[ZMQ] Starting ZMQ stream processing...")
            self.running = True
            
            while self.running:
                try:
                    # Receive raw bytes with timeout
                    print("[ZMQ] Waiting for data...")
                    if self.socket.poll(timeout=1000) == 0:  # 1 second timeout
                        print("[ZMQ] No data received within timeout")
                        continue
                        
                    data = self.socket.recv()
                    if not data:
                        print("[ZMQ] Received empty data")
                        continue
                        
                    # Convert bytes to numpy array
                    print(f"[ZMQ] Received data of size: {len(data)} bytes")
                    frame = np.frombuffer(data, dtype=np.uint8)
                    
                    # Calculate expected size based on display dimensions and upscaling
                    expected_size = int(DISPLAY_HEIGHT * DISPLAY_WIDTH * 3 * (DISPLAY_SCALE ** 2))  # 3 channels for RGB, squared scale factor for 2D upscaling
                    if len(frame) != expected_size:
                        print(f"[ZMQ] Warning: Received data size {len(frame)} doesn't match expected size {expected_size}")
                        continue
                        
                    # Reshape to image dimensions accounting for upscaling
                    frame = frame.reshape(int(DISPLAY_HEIGHT * DISPLAY_SCALE), int(DISPLAY_WIDTH * DISPLAY_SCALE), 3)
                    
                    if frame is not None:
                        self.frame_received.emit(frame)
                    else:
                        print("[ZMQ] Failed to reshape frame")
                        
                except zmq.error.Again:
                    print("[ZMQ] ZMQ timeout - no data received")
                    continue
                except Exception as e:
                    print(f"[ZMQ] Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                        
        except Exception as e:
            print(f"[ZMQ] Error in ZMQ thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[ZMQ] Cleaning up ZMQ resources...")
            self.running = False
            self.socket.close()
            self.context.term()
            print("[ZMQ] ZMQ resources cleaned up")

    def stop(self):
        """Stop the ZMQ thread"""
        print("[ZMQ] Stopping ZMQ thread...")
        self.running = False
        self.wait()
        print("[ZMQ] ZMQ thread stopped")

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
        
        # Fullscreen button
        self.button_layout = QHBoxLayout()
        self.fullscreen_button = QPushButton("Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50%);
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 70%);
            }
        """)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.fullscreen_button)
        self.layout.addLayout(self.button_layout)
        
        # Stream handling
        self.stream_thread = None
        self.zmq_thread = None
        
        # Fullscreen window
        self.fullscreen_window = None
        self.is_fullscreen = False
        
        # Black frame mode
        self.black_frame_mode = False
        
        # Mirror mode
        self.mirrored = False
        self.is_mirrored = False

    def toggle_fullscreen(self):
        """Toggle fullscreen window"""
        if not self.is_fullscreen:
            if not self.fullscreen_window:
                self.fullscreen_window = FullscreenWindow()
            self.fullscreen_window.show()
            self.fullscreen_window.showFullScreen()
            self.is_fullscreen = True
            self.fullscreen_button.setText("Exit Fullscreen")
        else:
            if self.fullscreen_window:
                self.fullscreen_window.close()
                self.fullscreen_window = None
            self.is_fullscreen = False
            self.fullscreen_button.setText("Fullscreen")

    def start_stream(self, user_id: str, server_uri: str = "http://localhost:7860"):
        """Start receiving the image stream
        
        Args:
            user_id: The user ID for the stream
            server_uri: The server URI (default: http://localhost:7860)
        """
        if self.stream_thread and self.stream_thread.running:
            self.stop_stream()
            
        # Create and start ZMQ thread
        print("[ZMQ] Starting ZMQ stream")
        self.zmq_thread = ZMQThread()
        self.zmq_thread.frame_received.connect(self.update_frame)
        self.zmq_thread.start()
        
        # NOTE: Required for ZMQ to start
        stream_url = f"{server_uri}/api/stream/{user_id}"
        print(f"[Stream] Starting WebSocket stream from: {stream_url}")
        self.stream_thread = StreamThread(stream_url)
        self.stream_thread.frame_received.connect(self.update_frame)
        self.stream_thread.start()

    def stop_stream(self):
        """Stop the stream threads"""
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None
            
        if self.zmq_thread:
            self.zmq_thread.stop()
            self.zmq_thread = None

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new frame"""
        if frame is None:
            return
        
        # If black frame mode is enabled, show black frame instead
        if self.black_frame_mode:
            # Calculate image dimensions from config
            height = int(DISPLAY_HEIGHT * DISPLAY_SCALE)
            width = int(DISPLAY_WIDTH * DISPLAY_SCALE)
            
            # Create black frame (RGB)
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame = black_frame
        
        # Apply mirroring if enabled
        if self.mirrored:
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip
            
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Get the available size of the label
        available_size = self.image_label.size()
        
        # Scale the pixmap to fit the available space while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
        
        # Update fullscreen window if active
        if self.fullscreen_window and self.is_fullscreen:
            self.fullscreen_window.update_frame(frame)
        
        # Update FPS counter in status bar
        main_window = self.window()
        if main_window:
            main_window.status_bar.update_fps()

    def clear_display(self):
        """Clear the display and stop stream"""
        self.stop_stream()
        self.image_label.clear()
        if self.fullscreen_window:
            self.fullscreen_window.clear_display()

    def clear_zmq_queue(self):
        """Clear any pending messages in the ZMQ queue"""
        if self.zmq_thread and self.zmq_thread.running:
            try:
                # Clear pending messages by receiving all available data without blocking
                while self.zmq_thread.socket.poll(timeout=0) > 0:  # 0 timeout = non-blocking
                    self.zmq_thread.socket.recv(zmq.NOBLOCK)
                    print("[ZMQ] Cleared pending message from queue")
            except zmq.error.Again:
                # No more messages to clear
                pass
            except Exception as e:
                print(f"[ZMQ] Error clearing queue: {e}")

    def display_black_frame(self):
        """Display a black frame of the expected image size"""
        try:
            # Calculate image dimensions from config
            height = int(DISPLAY_HEIGHT * DISPLAY_SCALE)
            width = int(DISPLAY_WIDTH * DISPLAY_SCALE)
            
            # Create black frame (RGB)
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Display the black frame
            self.update_frame(black_frame)
            print(f"[Display] Showing black frame of size {width}x{height}")
            
        except Exception as e:
            print(f"[Display] Error creating black frame: {e}")

    def set_black_frame_mode(self, enabled: bool):
        """Enable or disable black frame mode"""
        self.black_frame_mode = enabled
        if enabled:
            print("[Display] Black frame mode enabled")
        else:
            print("[Display] Black frame mode disabled")

    def set_mirror_mode(self, enabled: bool):
        """Enable or disable mirror mode"""
        self.mirrored = enabled
        self.is_mirrored = enabled
        if enabled:
            print("[Display] Mirror mode enabled")
        else:
            print("[Display] Mirror mode disabled")

    def toggle_mirror(self):
        """Toggle mirror mode"""
        self.set_mirror_mode(not self.mirrored)

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_stream()
        if self.fullscreen_window:
            self.fullscreen_window.close()
            self.fullscreen_window = None