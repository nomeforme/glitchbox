import sys
import cv2
import numpy as np
import os
import asyncio
import argparse
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
from PySide6.QtCore import Qt, QTimer

from components import CameraDisplay
from components import ProcessedDisplay
from components import ControlPanel
from components import StatusBar
from clients import WebSocketClient
from threads import CameraThread, SpeechToTextThread, FFTAnalyzerThread

load_dotenv(override=True)

# Default server configuration
DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_PORT = os.getenv("DEFAULT_SERVER_PORT")

class MainWindow(QMainWindow):
    def __init__(self, server_host, server_port):
        super().__init__()
        self.setWindowTitle("Glitch Machine Engine")
        
        # Store server configuration
        self.server_host = server_host
        self.server_port = server_port
        self.server_ws_uri = f"ws://{server_host}:{server_port}"
        self.server_http_uri = f"http://{server_host}:{server_port}"
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add status bar at the top
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        # Video feeds container
        self.feeds_layout = QHBoxLayout()
        
        # Camera feed container
        self.camera_container = QFrame()
        camera_container_layout = QVBoxLayout(self.camera_container)
        
        # Camera feed
        self.camera_display = CameraDisplay()
        camera_container_layout.addWidget(self.camera_display)
        
        # Camera label
        self.camera_label = QLabel("Input Camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_container_layout.addWidget(self.camera_label)
        
        self.feeds_layout.addWidget(self.camera_container)
        
        # Processed feed container
        self.processed_container = QFrame()
        processed_container_layout = QVBoxLayout(self.processed_container)
        
        # Processed feed
        self.processed_display = ProcessedDisplay()
        processed_container_layout.addWidget(self.processed_display)
        
        # Processed label
        self.processed_label = QLabel("Output Stream")
        self.processed_label.setAlignment(Qt.AlignCenter)
        processed_container_layout.addWidget(self.processed_label)
        
        self.feeds_layout.addWidget(self.processed_container)
        
        layout.addLayout(self.feeds_layout)
        
        # Controls section
        self.controls_container = QFrame()
        controls_layout = QVBoxLayout(self.controls_container)
        
        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.start_button.setEnabled(False)  # Disabled until settings are loaded
        buttons_layout.addWidget(self.start_button)
        
        # STT Toggle button
        self.stt_button = QPushButton("Start Speech Recognition")
        self.stt_button.clicked.connect(self.toggle_stt)
        buttons_layout.addWidget(self.stt_button)
        
        # FFT Toggle button
        self.fft_button = QPushButton("Start Audio FFT")
        self.fft_button.clicked.connect(self.toggle_fft)
        buttons_layout.addWidget(self.fft_button)
        
        # Add buttons layout to controls
        controls_layout.addLayout(buttons_layout)
        
        # Pipeline controls
        self.control_panel = ControlPanel()
        self.control_panel.parameter_changed.connect(self.update_parameter)
        controls_layout.addWidget(self.control_panel)
        
        layout.addWidget(self.controls_container)
        
        # Presentation mode buttons
        presentation_layout = QHBoxLayout()
        
        self.toggle_input_button = QPushButton("Hide Input Feed")
        self.toggle_input_button.clicked.connect(self.toggle_input_feed)
        presentation_layout.addWidget(self.toggle_input_button)
        
        self.toggle_controls_button = QPushButton("Hide Controls")
        self.toggle_controls_button.clicked.connect(self.toggle_controls)
        presentation_layout.addWidget(self.toggle_controls_button)
        
        self.toggle_presentation_button = QPushButton("Enter Presentation Mode")
        self.toggle_presentation_button.clicked.connect(self.toggle_presentation_mode)
        presentation_layout.addWidget(self.toggle_presentation_button)
        
        self.toggle_fullscreen_button = QPushButton("Detach Output")
        self.toggle_fullscreen_button.clicked.connect(self.toggle_fullscreen)
        presentation_layout.addWidget(self.toggle_fullscreen_button)
        
        layout.addLayout(presentation_layout)
        
        # Initialize threads
        self.ws_client = WebSocketClient(uri=self.server_ws_uri, max_retries=10, initial_retry_delay=1.0)
        self.camera_thread = CameraThread()
        
        # Initialize the audio device index for both STT and FFT
        self.audio_device_index_stt = 16
        self.audio_device_index_fft = 16

        # Initialize the STT thread with the audio device index
        self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index_stt)
        self.stt_thread.transcription_updated.connect(self.handle_transcription)
        self.stt_active = False
        
        # Initialize the FFT thread with the audio device index
        self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index_fft)
        self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
        self.fft_active = False
        
        # Connect signals
        self.ws_client.frame_received.connect(self.processed_display.update_frame)
        self.ws_client.connection_error.connect(self.handle_connection_error)
        self.ws_client.settings_received.connect(self.handle_settings)
        self.ws_client.status_changed.connect(self.handle_status_change)
        self.camera_thread.frame_ready.connect(self.handle_camera_frame)
        
        # Frame processing timer
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.setInterval(33)  # ~30 FPS
        
        # State variables
        self.current_frame = None
        self.processing_frame = False
        self.presentation_mode = False

        # Get initial settings
        self.get_initial_settings()

    def get_initial_settings(self):
        """Start WebSocket client to get initial settings"""
        self.status_bar.update_processing_status("Getting settings...")
        self.ws_client.start()

    def handle_settings(self, settings):
        """Handle received pipeline settings"""
        self.control_panel.setup_pipeline_options(settings)
        self.start_button.setEnabled(True)
        self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")

    def handle_status_change(self, status: str):
        """Handle WebSocket status changes"""
        if status.startswith("Retrying connection"):
            self.status_bar.update_processing_status(f"Retrying connection to {self.server_host}:{self.server_port}...")
        elif status == "connected":
            self.status_bar.update_connection_status(True)
            # Update the status bar with server info
            self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
            # Start the MJPEG stream when connected
            self.processed_display.start_stream(self.ws_client.user_id, self.server_http_uri)
        elif status == "disconnected":
            self.status_bar.update_connection_status(False)
            # Clear the server info from status bar
            self.status_bar.update_processing_status("")
            # Stop the stream when disconnected
            self.processed_display.stop_stream()
        elif status == "ready":
            # Reset UI state when camera is stopped
            self.start_button.setText("Start Camera")
            self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
        elif status == "Connection failed after maximum retries":
            self.status_bar.update_connection_status(False)
            self.status_bar.update_processing_status("")

    def handle_camera_frame(self, frame):
        """Handle new frame from camera"""
        self.current_frame = frame
        self.camera_display.update_frame(frame)

    def update_parameter(self, param_id: str, value):
        """Update parameter in WebSocket client"""
        self.ws_client.update_settings({param_id: value})

    def handle_fft_data(self, fft_data):
        """Handle FFT data from the FFT analyzer thread"""
        if fft_data and isinstance(fft_data, dict):
            # Update the acid_settings in the WebSocket client
            self.ws_client.update_settings({"acid_settings": fft_data})
            # Update the UI to show FFT is active
            if "binned_fft" in fft_data:
                bins = fft_data["binned_fft"]
                if isinstance(bins, list) and len(bins) > 0:
                    avg_energy = sum(bins) / len(bins)
                    self.status_bar.update_processing_status(f"FFT Audio Energy: {avg_energy:.2f}")

    def toggle_camera(self):
        """Start/Stop camera and processing"""
        if self.start_button.text() == "Start Camera":
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start camera and processing"""
        self.camera_thread.start()
        self.ws_client.start_camera()
        self.frame_timer.start()
        self.start_button.setText("Stop Camera")
        self.status_bar.update_processing_status("Processing frames...")

    def stop_camera(self):
        """Stop camera and processing"""
        print("[UI] Stopping camera")
        # Stop frame timer first
        self.frame_timer.stop()
        self.processing_frame = False
        
        # Stop camera thread
        self.camera_thread.stop()
        
        # Stop WebSocket processing and wait for ready status
        self.ws_client.stop_camera()
        
        # Clear displays immediately
        self.camera_display.clear_display()
        self.processed_display.clear_display()
        self.status_bar.update_processing_status("Stopping...")

    def process_frame(self):
        """Process current frame through WebSocket"""
        if self.current_frame is not None and not self.processing_frame:
            self.processing_frame = True
            # Store the current frame in the websocket client
            self.ws_client.current_frame = self.current_frame
            # Reset flag after setting the frame
            self.processing_frame = False

    def handle_connection_error(self, error_msg: str):
        """Handle connection errors"""
        self.status_bar.update_processing_status(f"Error: {error_msg}")
        self.status_bar.update_connection_status(False)

    def handle_transcription(self, text):
        """Handle transcribed text from STT and update the prompt"""
        if text and len(text.strip()) > 0:
            # Update the prompt in the WebSocket client
            self.ws_client.update_prompt(text)
            # Update the UI to show the current prompt
            self.status_bar.update_processing_status(f"Prompt: {text}")

    def toggle_stt(self):
        """Toggle speech-to-text processing"""
        if not self.stt_active:
            # Start STT
            self.stt_thread.start()
            self.stt_active = True
            self.stt_button.setText("Stop Speech Recognition")
            self.status_bar.update_processing_status("Speech recognition active")
        else:
            # Stop STT
            self.stt_thread.stop()
            self.stt_active = False
            self.stt_button.setText("Start Speech Recognition")
            self.status_bar.update_processing_status("Speech recognition stopped")

    def toggle_fft(self):
        """Toggle FFT audio analysis"""
        if not self.fft_active:
            # Start FFT
            self.fft_thread.start()
            self.fft_active = True
            self.fft_button.setText("Stop Audio FFT")
            self.status_bar.update_processing_status("FFT audio analysis active")
        else:
            # Stop FFT
            self.fft_thread.stop()
            self.fft_active = False
            self.fft_button.setText("Start Audio FFT")
            self.status_bar.update_processing_status("FFT audio analysis stopped")
            
    def toggle_input_feed(self):
        """Toggle visibility of the input camera feed"""
        if self.camera_container.isVisible():
            self.camera_container.hide()
            self.toggle_input_button.setText("Show Input Feed")
            # Adjust the processed display to take full width
            self.processed_container.setMinimumWidth(self.width() - 40)
            # Resize window to be more compact
            new_width = max(self.width() // 2, 640)  # Don't go smaller than 640px
            self.resize(new_width, self.height())
        else:
            self.camera_container.show()
            self.toggle_input_button.setText("Hide Input Feed")
            # Reset the processed display width
            self.processed_container.setMinimumWidth(0)
            # Restore window width
            self.resize(self.width() * 2, self.height())
            
    def toggle_controls(self):
        """Toggle visibility of the controls panel"""
        if self.controls_container.isVisible():
            self.controls_container.hide()
            self.toggle_controls_button.setText("Show Controls")
            # Adjust window height
            new_height = self.height() - self.controls_container.height()
            self.resize(self.width(), new_height)
        else:
            self.controls_container.show()
            self.toggle_controls_button.setText("Hide Controls")
            # Restore window height
            new_height = self.height() + self.controls_container.height()
            self.resize(self.width(), new_height)
            
    def toggle_presentation_mode(self):
        """Toggle presentation mode (hide input feed and controls)"""
        if not self.presentation_mode:
            # Store original window size and state for restoration
            self.original_size = self.size()
            self.original_pos = self.pos()
            self.original_window_state = self.windowState()
            
            # Enter presentation mode
            self.camera_container.hide()
            self.controls_container.hide()
            self.toggle_input_button.setText("Show Input Feed")
            self.toggle_controls_button.setText("Show Controls")
            self.toggle_presentation_button.setText("Exit Presentation Mode")
            
            # Hide status bar and connection label for cleaner look
            self.status_bar.hide()
            
            # Remove all margins and spacing from layouts
            self.feeds_layout.setContentsMargins(0, 0, 0, 0)
            self.feeds_layout.setSpacing(0)
            self.layout().setContentsMargins(0, 0, 0, 0)
            self.layout().setSpacing(0)
            
            # Make the processed display fill the entire window
            self.processed_container.setContentsMargins(0, 0, 0, 0)
            self.processed_display.setContentsMargins(0, 0, 0, 0)
            self.processed_label.hide()  # Hide the label in presentation mode
            
            # Go fullscreen
            self.showFullScreen()
            
        else:
            # Exit presentation mode
            self.camera_container.show()
            self.controls_container.show()
            self.toggle_input_button.setText("Hide Input Feed")
            self.toggle_controls_button.setText("Hide Controls")
            self.toggle_presentation_button.setText("Enter Presentation Mode")
            
            # Show status bar and connection label
            self.status_bar.show()
            self.processed_label.show()
            
            # Restore original margins and spacing
            self.feeds_layout.setContentsMargins(9, 9, 9, 9)
            self.feeds_layout.setSpacing(6)
            self.layout().setContentsMargins(9, 9, 9, 9)
            self.layout().setSpacing(6)
            
            # Restore processed display margins
            self.processed_container.setContentsMargins(9, 9, 9, 9)
            self.processed_display.setContentsMargins(9, 9, 9, 9)
            
            # Restore original window state
            if hasattr(self, 'original_window_state'):
                if self.original_window_state == Qt.WindowFullScreen:
                    self.showFullScreen()
                else:
                    self.showNormal()
                    self.resize(self.original_size)
                    if hasattr(self, 'original_pos'):
                        self.move(self.original_pos)
            
        self.presentation_mode = not self.presentation_mode

    def toggle_fullscreen(self):
        """Toggle fullscreen mode for the processed display"""
        if hasattr(self, 'processed_display'):
            self.processed_display.toggle_fullscreen()
            if self.processed_display.is_fullscreen:
                self.toggle_fullscreen_button.setText("Close Output Window")
            else:
                self.toggle_fullscreen_button.setText("Detach Output")

    def closeEvent(self, event):
        """Handle window close event - cleanup all threads in proper order"""
        print("[UI] Closing window - cleaning up...")
        try:
            # First stop all active processes
            self.frame_timer.stop()
            
            # Stop the STT thread if it's running
            if hasattr(self, 'stt_thread') and self.stt_thread is not None:
                self.stt_thread.stop()
            
            # Stop the FFT thread if it's running
            if hasattr(self, 'fft_thread') and self.fft_thread is not None:
                self.fft_thread.stop()
            
            # Stop the stream immediately to prevent further network activity
            self.processed_display.stop_stream()
            
            # Force flag changes to prevent new operations
            if self.ws_client is not None:
                self.ws_client.running = False
                self.ws_client.processing = False
            
            # Stop camera thread with short timeout
            if self.camera_thread is not None:
                self.camera_thread.stop()
                if not self.camera_thread.wait(1000):  # 1 second timeout
                    print("[UI] Force terminating camera thread")
                    self.camera_thread.terminate()
            
            # Stop WebSocket client with short timeout to avoid blocking
            if self.ws_client is not None:
                self.ws_client.stop()
                if not self.ws_client.wait(1000):  # 1 second timeout
                    print("[UI] Force terminating WebSocket thread")
                    self.ws_client.terminate()
            
            # Clear displays
            self.camera_display.clear_display()
            self.processed_display.clear_display()
            
            print("[UI] Cleanup completed successfully")
        except Exception as e:
            print(f"[UI] Error during cleanup: {e}")
        finally:
            # Always accept the close event
            event.accept()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glitch Machine Engine Client')
    parser.add_argument('--host', default=DEFAULT_SERVER_HOST, help=f'Server hostname (default: {DEFAULT_SERVER_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_SERVER_PORT, help=f'Server port (default: {DEFAULT_SERVER_PORT})')
    args = parser.parse_args()
    
    # Print server configuration
    print(f"Connecting to server at {args.host}:{args.port}")
    
    app = QApplication(sys.argv)
    window = MainWindow(server_host=args.host, server_port=args.port)
    window.setGeometry(100, 100, 1280, 720)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()