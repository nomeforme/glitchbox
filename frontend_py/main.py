import sys
import cv2
import numpy as np
import asyncio
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, QTimer

from components.camera_display import CameraDisplay
from components.processed_display import ProcessedDisplay
from components.control_panel import ControlPanel
from components.status_bar import StatusBar
from websocket_client import WebSocketClient
from camera_thread import CameraThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glitch Machine Engine")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add status bar at the top
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        # Video feeds container
        feeds_layout = QHBoxLayout()
        
        # Camera feed
        self.camera_display = CameraDisplay()
        feeds_layout.addWidget(self.camera_display)
        
        # Processed feed
        self.processed_display = ProcessedDisplay()
        feeds_layout.addWidget(self.processed_display)
        
        layout.addLayout(feeds_layout)
        
        # Controls section
        controls_layout = QVBoxLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.start_button.setEnabled(False)  # Disabled until settings are loaded
        controls_layout.addWidget(self.start_button)
        
        # Pipeline controls
        self.control_panel = ControlPanel()
        self.control_panel.parameter_changed.connect(self.update_parameter)
        controls_layout.addWidget(self.control_panel)
        
        layout.addLayout(controls_layout)
        
        # Initialize threads
        self.ws_client = WebSocketClient()
        self.camera_thread = CameraThread()
        
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
        self.status_bar.update_processing_status("Ready")

    def handle_status_change(self, status: str):
        """Handle WebSocket status changes"""
        self.status_bar.update_processing_status(status)
        if status == "connected":
            self.status_bar.update_connection_status(True)
            # Start the MJPEG stream when connected
            self.processed_display.start_stream(self.ws_client.user_id)
        elif status == "disconnected":
            self.status_bar.update_connection_status(False)
            # Stop the stream when disconnected
            self.processed_display.stop_stream()
        elif status == "ready":
            # Reset UI state when camera is stopped
            self.start_button.setText("Start Camera")
            self.status_bar.update_processing_status("Ready")

    def handle_camera_frame(self, frame):
        """Handle new frame from camera"""
        self.current_frame = frame
        self.camera_display.update_frame(frame)

    def update_parameter(self, param_id: str, value):
        """Update parameter in WebSocket client"""
        self.ws_client.update_settings({param_id: value})

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

    def closeEvent(self, event):
        """Handle window close event - cleanup all threads in proper order"""
        print("[UI] Closing window - cleaning up...")
        try:
            # First stop all active processes
            self.frame_timer.stop()
            
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 1280, 720)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()