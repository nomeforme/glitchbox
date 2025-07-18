import sys
import cv2
import numpy as np
import os
import asyncio
import argparse
import threading
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QScrollArea, QSpinBox
from PySide6.QtCore import Qt, QTimer, Signal, QObject

# Add imports for device detection
import pyaudio
from typing import List, Tuple

from components import CameraDisplay
from components import ProcessedDisplay
from components import ControlPanel
from components import StatusBar
from clients import WebSocketClient
from threads import CameraThread, SpeechToTextThread, FFTAnalyzerThread
from config import MIC_DEVICE_INDEX, AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE, CAMERA_DEVICE_INDEX, CURATION_INDEX_AUTO_UPDATE, CURATION_INDEX_UPDATE_TIME, CURATION_INDEX_MAX
load_dotenv(override=True)

# Default server configuration
DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_PORT = os.getenv("DEFAULT_SERVER_PORT")

class CurationUpdateSignalHandler(QObject):
    """Signal handler for curation index updates"""
    update_completed = Signal(bool, str)  # success, message

def detect_cameras() -> List[Tuple[int, str, str]]:
    """Detect available camera indices with device names and info"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Get basic camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Try to get device name from sysfs (Linux)
                    device_name = f"Camera {i}"
                    try:
                        sysfs_path = f"/sys/class/video4linux/video{i}/name"
                        if os.path.exists(sysfs_path):
                            with open(sysfs_path, 'r') as f:
                                device_name = f.read().strip()
                    except Exception:
                        pass
                    
                    # Create info string with resolution and FPS
                    info = f"{width}x{height}"
                    if fps > 0:
                        info += f" @ {fps:.1f}fps"
                    
                    available_cameras.append((i, device_name, info))
                cap.release()
            else:
                cap.release()
        except Exception:
            pass
    return available_cameras

def detect_microphones() -> List[Tuple[int, str]]:
    """Detect available microphone devices"""
    available_mics = []
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                # Only include devices with input channels
                if device_info['maxInputChannels'] > 0:
                    name = device_info['name']
                    available_mics.append((i, name))
            except Exception:
                pass
        p.terminate()
    except Exception:
        pass
    return available_mics

class MainWindow(QMainWindow):
    def __init__(self, server_host, server_port):
        super().__init__()
        self.setWindowTitle("Glitch Machine Engine")
        
        # Store server configuration
        self.server_host = server_host
        self.server_port = server_port
        self.server_ws_uri = f"ws://{server_host}:{server_port}"
        self.server_http_uri = f"http://{server_host}:{server_port}"

        # Initialize device indices from config
        self.camera_device_index = CAMERA_DEVICE_INDEX
        self.audio_device_index = MIC_DEVICE_INDEX
        
        # Detect available devices
        self.available_cameras = detect_cameras()
        self.available_microphones = detect_microphones()
        
        print(f"[UI] Detected cameras: {[(idx, name, info) for idx, name, info in self.available_cameras]}")
        print(f"[UI] Detected microphones: {[f'{idx}: {name}' for idx, name in self.available_microphones]}")
        
        # Import and set UI behavior config
        self.auto_disable_black_frame_after_curation_update = AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE
        
        # Create scroll area as the central widget
        self.scroll_area = QScrollArea()
        self.setCentralWidget(self.scroll_area)
        
        # Configure scroll area properties
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Enable smooth scrolling and configure for cross-platform compatibility
        self.scroll_area.verticalScrollBar().setSingleStep(20)  # Smooth scrolling step
        self.scroll_area.horizontalScrollBar().setSingleStep(20)
        
        # Set scroll bar styling for better appearance on both Windows and Linux
        scroll_style = """
        QScrollBar:vertical {
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            background: #f0f0f0;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #c0c0c0;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
        """
        self.scroll_area.setStyleSheet(scroll_style)
        
        # Create the main content widget that will be scrollable
        self.main_content_widget = QWidget()
        self.scroll_area.setWidget(self.main_content_widget)
        
        # Create layout for the main content
        layout = QVBoxLayout(self.main_content_widget)
        
        # Set minimum size for the content widget to ensure proper scrolling
        self.main_content_widget.setMinimumSize(800, 600)
        
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
        
        # Device Index Controls - arranged horizontally
        device_controls_layout = QHBoxLayout()
        
        # Curation Index Control
        curation_layout = QHBoxLayout()
        curation_label = QLabel("Curation Index:")
        self.curation_spinbox = QSpinBox()
        self.curation_spinbox.setMinimum(0)
        self.curation_spinbox.setMaximum(CURATION_INDEX_MAX)
        self.curation_spinbox.setValue(0)  # Default value
        self.curation_update_button = QPushButton("Update")
        self.curation_update_button.clicked.connect(self.update_curation_index)
        
        # Connect value change signal to validate the input
        self.curation_spinbox.valueChanged.connect(self.validate_curation_index)
        
        curation_layout.addWidget(curation_label)
        curation_layout.addWidget(self.curation_spinbox)
        curation_layout.addWidget(self.curation_update_button)
        
        device_controls_layout.addLayout(curation_layout)
        
        # Camera Index Control
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera Index:")
        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setMinimum(0)
        self.camera_spinbox.setMaximum(20)  # Allow higher range in case detection missed some
        self.camera_spinbox.setValue(self.camera_device_index)
        
        # Set range and tooltip based on detected cameras
        if self.available_cameras:
            camera_list = [f"{idx}: {name} ({info})" for idx, name, info in self.available_cameras]
            self.camera_spinbox.setToolTip("Available cameras:\n" + "\n".join(camera_list))
        else:
            self.camera_spinbox.setToolTip("No cameras detected, but you can still try different indices")
            
        self.camera_update_button = QPushButton("Update")
        self.camera_update_button.clicked.connect(self.update_camera_index)
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_spinbox)
        camera_layout.addWidget(self.camera_update_button)
        
        device_controls_layout.addLayout(camera_layout)
        
        # Microphone Index Control
        mic_layout = QHBoxLayout()
        mic_label = QLabel("Microphone Index:")
        self.mic_spinbox = QSpinBox()
        self.mic_spinbox.setMinimum(0)
        self.mic_spinbox.setMaximum(50)  # Audio devices can have higher indices
        self.mic_spinbox.setValue(self.audio_device_index)
        
        # Set tooltip with detected microphones
        if self.available_microphones:
            mic_list = [f"{idx}: {name[:30]}..." if len(name) > 30 else f"{idx}: {name}" 
                       for idx, name in self.available_microphones]
            self.mic_spinbox.setToolTip("Available microphones:\n" + "\n".join(mic_list))
        else:
            self.mic_spinbox.setToolTip("No microphones detected, but you can still try different indices")
            
        self.mic_update_button = QPushButton("Update")
        self.mic_update_button.clicked.connect(self.update_mic_index)
        
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_spinbox)
        mic_layout.addWidget(self.mic_update_button)
        
        device_controls_layout.addLayout(mic_layout)
        
        # Refresh Devices Control
        refresh_layout = QHBoxLayout()
        refresh_label = QLabel("Device Lists:")
        self.refresh_devices_button = QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self.refresh_device_lists)
        self.refresh_devices_button.setToolTip("Re-scan for available cameras and microphones")
        
        refresh_layout.addWidget(refresh_label)
        refresh_layout.addWidget(self.refresh_devices_button)
        
        device_controls_layout.addLayout(refresh_layout)
        
        # Add stretch to push everything to the left
        device_controls_layout.addStretch()
        
        controls_layout.addLayout(device_controls_layout)
        
        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        
        # Connect to Server button
        self.connect_button = QPushButton("Connect to Server")
        self.connect_button.clicked.connect(self.connect_to_server)
        buttons_layout.addWidget(self.connect_button)
        
        # Disconnect from Server button
        self.disconnect_button = QPushButton("Disconnect from Server")
        self.disconnect_button.clicked.connect(self.disconnect_from_server)
        self.disconnect_button.setEnabled(False)  # Only enabled when connected
        buttons_layout.addWidget(self.disconnect_button)
        
        # Reconnect button (renamed for clarity)
        self.reconnect_button = QPushButton("Reconnect to Server")
        self.reconnect_button.clicked.connect(self.reconnect_to_server)
        self.reconnect_button.setEnabled(False)  # Only enabled when connected
        buttons_layout.addWidget(self.reconnect_button)
        
        # Start/Stop Camera button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        # Camera button is now always enabled
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
        
        self.toggle_black_frame_button = QPushButton("Enable Black Frame")
        self.toggle_black_frame_button.clicked.connect(self.toggle_black_frame)
        presentation_layout.addWidget(self.toggle_black_frame_button)

        self.toggle_mirror_button = QPushButton("Enable Mirror")
        self.toggle_mirror_button.clicked.connect(self.toggle_mirror)
        presentation_layout.addWidget(self.toggle_mirror_button)
        
        self.toggle_presentation_button = QPushButton("Enter Presentation Mode")
        self.toggle_presentation_button.clicked.connect(self.toggle_presentation_mode)
        presentation_layout.addWidget(self.toggle_presentation_button)
        
        self.toggle_fullscreen_button = QPushButton("Detach Output")
        self.toggle_fullscreen_button.clicked.connect(self.toggle_fullscreen)
        presentation_layout.addWidget(self.toggle_fullscreen_button)
        
        # Add automatic curation update toggle button
        self.toggle_auto_curation_button = QPushButton("Start Auto Curation Updates")
        self.toggle_auto_curation_button.clicked.connect(self.toggle_automatic_curation_updates)
        self.toggle_auto_curation_button.setToolTip(f"Toggle automatic curation index updates (every {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
        presentation_layout.addWidget(self.toggle_auto_curation_button)
        
        # Add force terminate button
        self.force_terminate_button = QPushButton("Exit")
        self.force_terminate_button.setStyleSheet("background-color: #ff4444; color: white;")
        self.force_terminate_button.clicked.connect(self.force_terminate)
        presentation_layout.addWidget(self.force_terminate_button)
        
        layout.addLayout(presentation_layout)
        
        # Initialize threads
        self.ws_client = WebSocketClient(uri=self.server_ws_uri, max_retries=10, initial_retry_delay=1.0)
        self.camera_thread = CameraThread()
        # Set the camera device index
        self.camera_thread.device_index = self.camera_device_index

        # Initialize the STT thread with the audio device index
        self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
        self.stt_thread.transcription_updated.connect(self.handle_transcription)
        self.stt_active = False
        
        # Initialize the FFT thread with the audio device index
        self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
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
        
        # Automatic curation index update timer
        self.curation_auto_timer = QTimer()
        self.curation_auto_timer.timeout.connect(self.perform_automatic_curation_update)
        self.curation_auto_timer.setInterval(CURATION_INDEX_UPDATE_TIME * 1000)  # Convert seconds to milliseconds
        
        # State variables
        self.current_frame = None
        self.processing_frame = False
        self.presentation_mode = False
        self.black_frame_enabled = False
        
        # Add separate state tracking for camera and server connection
        self.camera_running = False
        self.server_connected = False

        # Create signal handler for curation updates
        self.curation_signal_handler = CurationUpdateSignalHandler()
        self.curation_signal_handler.update_completed.connect(self._handle_curation_update_result)

        # Don't automatically connect to server - user will click Connect button
        # self.get_initial_settings()
        
        # Set initial status message
        self.status_bar.update_processing_status("Ready - Click 'Connect to Server' to connect")
        
        # Log automatic curation update configuration
        if CURATION_INDEX_AUTO_UPDATE:
            print(f"[UI] Automatic curation updates enabled (interval: {CURATION_INDEX_UPDATE_TIME} seconds)")
            print(f"[UI] Curation index range: 0 to {CURATION_INDEX_MAX}")
            print(f"[UI] Timer will start automatically when connected to server")
        else:
            print("[UI] Automatic curation updates disabled in config")
        print(f"[UI] Curation index range: 0 to {CURATION_INDEX_MAX}")

    def get_initial_settings(self):
        """Start WebSocket client to get initial settings"""
        print("[UI] get_initial_settings called - starting WebSocket client...")
        self.status_bar.update_processing_status("Connecting to server...")
        # Disable all connection buttons during connection
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.reconnect_button.setEnabled(False)
        
        try:
            print(f"[UI] Starting WebSocket client with URI: {self.ws_client.uri}")
            print(f"[UI] WebSocket client user_id: {self.ws_client.user_id}")
            self.ws_client.start()
            print("[UI] WebSocket client start() called successfully")
        except Exception as e:
            print(f"[UI] Error starting WebSocket client: {e}")
            self.status_bar.update_processing_status(f"Failed to start connection: {e}")
            # Re-enable buttons on error - allow connect and reconnect attempts
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.reconnect_button.setEnabled(True)  # Enable reconnect on connection error

    def connect_to_server(self):
        """Connect to server"""
        if not self.server_connected:
            self.get_initial_settings()
        else:
            self.status_bar.update_processing_status("Already connected to server")

    def disconnect_from_server(self):
        """Disconnect from server with proper cleanup"""
        if not self.server_connected:
            self.status_bar.update_processing_status("Not connected to server")
            return
            
        print("[UI] Disconnecting from server...")
        self.status_bar.update_processing_status("Disconnecting from server...")
        
        # Disable buttons during disconnection
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        self.reconnect_button.setEnabled(False)
        
        # Immediately update connection state to prevent further operations
        self.server_connected = False
        self.status_bar.update_connection_status(False)
        
        # Stop frame processing immediately
        print("[UI] Stopping frame processing...")
        self.frame_timer.stop()
        self.processing_frame = False
        
        # Stop the processed display stream immediately
        print("[UI] Stopping processed display stream...")
        # Clear ZMQ queue first to prevent blocking on pending messages
        self.processed_display.clear_zmq_queue()
        self.processed_display.stop_stream()
        self.processed_display.clear_display()
        
        # Stop automatic curation update timer
        if self.curation_auto_timer.isActive():
            print("[UI] Stopping automatic curation update timer")
            self.curation_auto_timer.stop()
            # Update button text to reflect inactive state
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Start Auto Curation Updates")
        
        # Use QTimer to perform cleanup asynchronously to avoid blocking UI
        def perform_async_cleanup():
            try:
                print("[UI] Starting async cleanup...")
                
                # Stop streaming if camera was running
                if self.camera_running:
                    print("[UI] Stopping camera streaming to server...")
                    try:
                        self.ws_client.stop_camera()
                    except Exception as e:
                        print(f"[UI] Error stopping camera streaming: {e}")
                
                # Gracefully close the WebSocket connection
                print("[UI] Gracefully closing WebSocket connection...")
                try:
                    self.ws_client.close()
                except Exception as e:
                    print(f"[UI] Error during close: {e}")
                
                print("[UI] Async cleanup completed")
                
            except Exception as e:
                print(f"[UI] Error during async cleanup: {e}")
            
            # Schedule the final cleanup after a short delay
            QTimer.singleShot(200, perform_final_cleanup)
        
        def perform_final_cleanup():
            try:
                print("[UI] Starting final cleanup...")
                
                # Stop the WebSocket client thread (non-blocking approach)
                if hasattr(self, 'ws_client'):
                    print("[UI] Stopping WebSocket client thread...")
                    self.ws_client.stop()
                
                # Restart frame timer if camera is still running
                if self.camera_running:
                    print("[UI] Restarting frame timer for local camera display...")
                    self.frame_timer.start()
                
                print("[UI] Server disconnection completed successfully")
                
            except Exception as e:
                print(f"[UI] Error during final cleanup: {e}")
                # Try to restart frame timer if camera is running, even after error
                if self.camera_running and not self.frame_timer.isActive():
                    self.frame_timer.start()
            finally:
                # Always update UI state regardless of errors
                self.connect_button.setEnabled(True)
                self.disconnect_button.setEnabled(False)
                self.reconnect_button.setEnabled(True)  # Enable reconnect when disconnected
                
                # Update status based on camera state
                if self.camera_running:
                    self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
                else:
                    self.status_bar.update_processing_status("Disconnected from server")
        
        # Start the async cleanup with a small delay to let the UI update
        QTimer.singleShot(50, perform_async_cleanup)

    def handle_settings(self, settings):
        """Handle received pipeline settings"""
        self.control_panel.setup_pipeline_options(settings)
        self.server_connected = True
        
        # Update UI state - when connected, disable connect and reconnect buttons
        self.connect_button.setEnabled(False)  # Disable connect when already connected
        self.disconnect_button.setEnabled(True)  # Enable disconnect button when connected
        self.reconnect_button.setEnabled(False)  # Disable reconnect when already connected
        self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
        
        # Ensure ProcessedDisplay streaming is started (important for reconnection)
        print("[UI] Starting ProcessedDisplay streaming...")
        try:
            # Stop any existing streams first
            self.processed_display.stop_stream()
            # Start fresh streaming
            self.processed_display.start_stream(self.ws_client.user_id, self.server_http_uri)
            print("[UI] ProcessedDisplay streaming started successfully")
        except Exception as e:
            print(f"[UI] Error starting ProcessedDisplay streaming: {e}")
        
        # If camera is already running, start streaming automatically
        if self.camera_running:
            self.ws_client.start_camera()
            self.status_bar.update_processing_status("Streaming frames to server...")
        
        # Restart streaming components
        self._restart_streaming_components()
        
        # Update curation index from server settings
        current_curation_index = settings.get('current_curation_index', 0)
        self.curation_spinbox.setValue(current_curation_index)
        print(f"[UI] Set curation index to {current_curation_index} from server settings")
        
        # Start automatic curation update timer if enabled
        if CURATION_INDEX_AUTO_UPDATE:
            print(f"[UI] Starting automatic curation update timer (interval: {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
            self.curation_auto_timer.start()
            print(f"[UI] Automatic curation update timer is now active")
            # Update button text to reflect active state
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Stop Auto Curation Updates")
        else:
            print("[UI] Automatic curation updates disabled in config")
            # Update button text to reflect inactive state
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Start Auto Curation Updates")

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
            # Update buttons on successful connection - disable connect and reconnect when connected
            self.connect_button.setEnabled(False)  # Disable connect when already connected
            self.disconnect_button.setEnabled(True)  # Enable disconnect when connected
            self.reconnect_button.setEnabled(False)  # Disable reconnect when already connected
        elif status == "disconnected":
            self.server_connected = False
            self.status_bar.update_connection_status(False)
            # Update button states
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.reconnect_button.setEnabled(True)  # Enable reconnect when disconnected
            # Clear the server info from status bar
            if self.camera_running:
                self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
            else:
                self.status_bar.update_processing_status("Disconnected from server")
            # Stop the stream when disconnected
            self.processed_display.stop_stream()
            # Stop automatic curation update timer
            if self.curation_auto_timer.isActive():
                print("[UI] Stopping automatic curation update timer due to disconnection")
                self.curation_auto_timer.stop()
                # Update button text to reflect inactive state
                if hasattr(self, 'toggle_auto_curation_button'):
                    self.toggle_auto_curation_button.setText("Start Auto Curation Updates")
        elif status == "ready":
            # This status comes when camera streaming is stopped on server side
            if self.camera_running and self.server_connected:
                self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
            elif self.camera_running:
                self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
        elif status == "Connection failed after maximum retries":
            self.server_connected = False
            self.status_bar.update_connection_status(False)
            self.status_bar.update_processing_status("Connection failed - click Connect or Reconnect to retry")
            # Ensure buttons are in correct state when connection fails
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.reconnect_button.setEnabled(True)  # Enable reconnect when connection fails

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
        """Start/Stop camera (local display only)"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start camera and local display"""
        self.camera_thread.start()
        self.frame_timer.start()
        self.camera_running = True
        self.start_button.setText("Stop Camera")
        
        # Update status based on connection state
        if self.server_connected:
            # Start server streaming automatically
            self.ws_client.start_camera()
            self.status_bar.update_processing_status("Camera running - streaming to server")
        else:
            self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")

    def stop_camera(self):
        """Stop camera and processing"""
        print("[UI] Stopping camera")
        # Stop frame timer first
        self.frame_timer.stop()
        self.processing_frame = False
        
        # Update state immediately
        self.camera_running = False
        self.start_button.setText("Start Camera")
        
        # Clear displays immediately
        self.camera_display.clear_display()
        self.processed_display.clear_display()
        
        # Use async cleanup to avoid blocking UI
        def perform_camera_cleanup():
            try:
                print("[UI] Starting camera cleanup...")
                
                # Stop camera thread
                self.camera_thread.stop()
                
                # Stop server streaming if connected (non-blocking)
                if self.server_connected:
                    try:
                        self.ws_client.stop_camera()
                    except Exception as e:
                        print(f"[UI] Error stopping server streaming: {e}")
                
                print("[UI] Camera cleanup completed")
                
            except Exception as e:
                print(f"[UI] Error during camera cleanup: {e}")
            finally:
                # Update status based on connection state
                if self.server_connected:
                    self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
                else:
                    self.status_bar.update_processing_status("Camera stopped")
        
        # Start cleanup asynchronously
        QTimer.singleShot(50, perform_camera_cleanup)

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
        self.server_connected = False
        self.status_bar.update_processing_status(f"Error: {error_msg}")
        self.status_bar.update_connection_status(False)
        
        # Update button states
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.reconnect_button.setEnabled(True)  # Enable reconnect after connection error
        
        # Update status message based on camera state
        if self.camera_running:
            self.status_bar.update_processing_status("Camera running (not streaming - connection error)")
        else:
            self.status_bar.update_processing_status(f"Connection error: {error_msg}")

    def reconnect_to_server(self):
        """Reconnect to server by completely recreating all network components"""
        print("[UI] Manual reconnection initiated...")
        self.status_bar.update_processing_status("Reconnecting to server...")
        # Disable all connection buttons during reconnection
        self.reconnect_button.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(False)
        
        # Store camera state to restore later
        was_camera_running = self.camera_running
        
        try:
            # Phase 1: Complete shutdown of all network components
            print("[UI] Phase 1: Shutting down all network components...")
            
            # Stop frame processing immediately
            self.frame_timer.stop()
            self.processing_frame = False
            
            # Update connection state immediately
            self.server_connected = False
            self.status_bar.update_connection_status(False)
            
            # Stop and cleanup processed display completely
            print("[UI] Stopping processed display...")
            self.processed_display.clear_zmq_queue()
            self.processed_display.stop_stream()
            self.processed_display.clear_display()
            
            # Stop and recreate WebSocket client completely
            print("[UI] Stopping WebSocket client...")
            if hasattr(self, 'ws_client') and self.ws_client is not None:
                # Force stop the WebSocket client
                self.ws_client.running = False
                self.ws_client.processing = False
                self.ws_client.close()
                # Give it a moment to stop
                if self.ws_client.isRunning():
                    self.ws_client.terminate()
                    self.ws_client.wait(500)  # Wait up to 500ms
                
                # Disconnect all signals to prevent issues
                try:
                    self.ws_client.frame_received.disconnect()
                    self.ws_client.connection_error.disconnect()
                    self.ws_client.settings_received.disconnect()
                    self.ws_client.status_changed.disconnect()
                except:
                    pass  # Ignore if signals weren't connected
            
            print("[UI] Phase 1 complete - all components stopped")
            
            # Phase 2: Recreate all components from scratch
            def recreate_components():
                try:
                    print("[UI] Phase 2: Recreating all network components...")
                    
                    # Import the WebSocket client class
                    from clients import WebSocketClient
                    
                    # Create completely new WebSocket client
                    print("[UI] Creating new WebSocket client...")
                    self.ws_client = WebSocketClient(uri=self.server_ws_uri, max_retries=10, initial_retry_delay=1.0)
                    
                    # Connect signals for the new client
                    self.ws_client.frame_received.connect(self.processed_display.update_frame)
                    self.ws_client.connection_error.connect(self.handle_connection_error)
                    self.ws_client.settings_received.connect(self.handle_settings)
                    self.ws_client.status_changed.connect(self.handle_status_change)
                    
                    print("[UI] New WebSocket client created and connected")
                    
                    # Update status
                    if was_camera_running:
                        self.status_bar.update_processing_status("Camera running (not streaming - reconnecting)")
                    else:
                        self.status_bar.update_processing_status("Reconnecting to server...")
                    
                    # Phase 3: Start the connection process
                    def start_connection():
                        try:
                            print("[UI] Phase 3: Starting connection process...")
                            print(f"[UI] Server URI: {self.server_ws_uri}")
                            print(f"[UI] WebSocket client user_id: {self.ws_client.user_id}")
                            self.get_initial_settings()
                            print("[UI] Reconnection process initiated successfully")
                        except Exception as e:
                            print(f"[UI] Error starting connection: {e}")
                            self.status_bar.update_processing_status(f"Reconnection failed: {e}")
                            self._update_button_states_after_reconnect()
                    
                    # Start connection with a small delay
                    QTimer.singleShot(200, start_connection)
                    
                except Exception as e:
                    print(f"[UI] Error recreating components: {e}")
                    self.status_bar.update_processing_status(f"Reconnection failed: {e}")
                    self._update_button_states_after_reconnect()
            
            # Start recreation process with a small delay to ensure cleanup is complete
            QTimer.singleShot(300, recreate_components)
            
        except Exception as e:
            print(f"[UI] Error during reconnection: {e}")
            self.status_bar.update_processing_status(f"Reconnection failed: {e}")
            self.server_connected = False
            # Re-enable buttons on error
            QTimer.singleShot(1000, lambda: self._update_button_states_after_reconnect())
        
        # Fallback: Re-enable buttons after reasonable time regardless
        QTimer.singleShot(10000, lambda: self._update_button_states_after_reconnect())

    def _restart_streaming_components(self):
        """Restart all streaming components for reconnection"""
        print("[UI] Restarting streaming components...")
        
        # The ProcessedDisplay will automatically restart its streaming when start_stream is called
        # This happens in handle_settings when the server connection is established
        # For now, just ensure it's ready to receive new streams
        
        # Restart frame timer if camera is running
        if self.camera_running:
            print("[UI] Restarting frame timer for camera...")
            if not self.frame_timer.isActive():
                self.frame_timer.start()
        
        print("[UI] Streaming components restart completed")

    def _update_button_states_after_reconnect(self):
        """Update button states after reconnection attempt"""
        try:
            if self.server_connected:
                self.connect_button.setEnabled(False)  # Disable connect when connected
                self.disconnect_button.setEnabled(True)  # Enable disconnect when connected
                self.reconnect_button.setEnabled(False)  # Disable reconnect when connected
                print("[UI] Reconnection successful - only disconnect button enabled")
            else:
                self.connect_button.setEnabled(True)  # Enable connect when disconnected
                self.disconnect_button.setEnabled(False)  # Disable disconnect when disconnected
                self.reconnect_button.setEnabled(True)  # Enable reconnect when disconnected
                print("[UI] Reconnection failed - connect and reconnect buttons enabled")
        except Exception as e:
            print(f"[UI] Error updating button states after reconnect: {e}")
            # Fallback: enable connect and reconnect buttons when disconnected
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.reconnect_button.setEnabled(True)

    def _stop_all_threads(self):
        """Stop all active threads cleanly"""
        print("[UI] Stopping all threads for reconnection...")
        
        # Stop frame timer
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
        self.processing_frame = False
        
        # Reset state variables
        self.camera_running = False
        self.server_connected = False
        
        # Stop STT thread if running
        if hasattr(self, 'stt_thread') and self.stt_thread is not None:
            print("[UI] Stopping STT thread...")
            self.stt_thread.stop()
            if not self.stt_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating STT thread")
                self.stt_thread.terminate()
        
        # Stop FFT thread if running
        if hasattr(self, 'fft_thread') and self.fft_thread is not None:
            print("[UI] Stopping FFT thread...")
            self.fft_thread.stop()
            if not self.fft_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating FFT thread")
                self.fft_thread.terminate()
        
        # Stop camera thread
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            print("[UI] Stopping camera thread...")
            self.camera_thread.stop()
            if not self.camera_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating camera thread")
                self.camera_thread.terminate()
        
        # Stop WebSocket client
        if hasattr(self, 'ws_client') and self.ws_client is not None:
            print("[UI] Stopping WebSocket client...")
            # Force stop flags
            self.ws_client.running = False
            self.ws_client.processing = False
            
            # Stop the stream immediately
            if hasattr(self, 'processed_display'):
                self.processed_display.stop_stream()
            
            # Stop WebSocket thread
            self.ws_client.stop()
            if not self.ws_client.wait(3000):  # 3 second timeout
                print("[UI] Force terminating WebSocket thread")
                self.ws_client.terminate()
        
        # Recreate threads (except WebSocket which is recreated in reconnect_to_server)
        print("[UI] Recreating threads...")
        self.camera_thread = CameraThread()
        self.camera_thread.device_index = self.camera_device_index
        self.camera_thread.frame_ready.connect(self.handle_camera_frame)
        
        # Recreate STT thread
        self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
        self.stt_thread.transcription_updated.connect(self.handle_transcription)
        
        # Recreate FFT thread
        self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
        self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
        
        # Reset UI button states for disconnected state
        self.start_button.setText("Start Camera")
        self.connect_button.setEnabled(True)  # Enable connect when disconnected
        self.disconnect_button.setEnabled(False)  # Disable disconnect when disconnected
        self.reconnect_button.setEnabled(True)  # Enable reconnect when disconnected
        self.stt_active = False
        self.stt_button.setText("Start Speech Recognition")
        self.fft_active = False
        self.fft_button.setText("Start Audio FFT")
        
        print("[UI] All threads stopped and recreated successfully")

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
            
            # Recreate the STT thread for next use (QThread cannot be restarted)
            self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
            self.stt_thread.transcription_updated.connect(self.handle_transcription)

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
            
            # Recreate the FFT thread for next use (QThread cannot be restarted)
            self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
            self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
            
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
            # Keep black frame button text as-is since it's independent of presentation mode
            self.toggle_presentation_button.setText("Exit Presentation Mode")
            
            # Hide status bar and connection label for cleaner look
            self.status_bar.hide()
            
            # Hide scroll bars in presentation mode
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            # Remove all margins and spacing from layouts
            self.feeds_layout.setContentsMargins(0, 0, 0, 0)
            self.feeds_layout.setSpacing(0)
            self.main_content_widget.layout().setContentsMargins(0, 0, 0, 0)
            self.main_content_widget.layout().setSpacing(0)
            
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
            # Keep black frame button text as-is since it's independent of presentation mode
            self.toggle_presentation_button.setText("Enter Presentation Mode")
            
            # Show status bar and connection label
            self.status_bar.show()
            self.processed_label.show()
            
            # Restore scroll bars
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Restore original margins and spacing
            self.feeds_layout.setContentsMargins(9, 9, 9, 9)
            self.feeds_layout.setSpacing(6)
            self.main_content_widget.layout().setContentsMargins(9, 9, 9, 9)
            self.main_content_widget.layout().setSpacing(6)
            
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

    def toggle_black_frame(self):
        """Toggle black frame mode for the processed display"""
        self.black_frame_enabled = not self.black_frame_enabled
        
        # Update the processed display
        self.processed_display.set_black_frame_mode(self.black_frame_enabled)
        
        # Update button text
        if self.black_frame_enabled:
            self.toggle_black_frame_button.setText("Disable Black Frame")
            self.status_bar.update_processing_status("Black frame mode enabled")
        else:
            self.toggle_black_frame_button.setText("Enable Black Frame")
            self.status_bar.update_processing_status("Black frame mode disabled")

    def toggle_mirror(self):
        """Toggle mirror mode for the processed display"""
        if hasattr(self, 'processed_display'):
            self.processed_display.toggle_mirror()
            if self.processed_display.is_mirrored:
                self.toggle_mirror_button.setText("Disable Mirror")
            else:
                self.toggle_mirror_button.setText("Enable Mirror")

    def scroll_to_top(self):
        """Scroll to the top of the content"""
        self.scroll_area.verticalScrollBar().setValue(0)
        
    def scroll_to_bottom(self):
        """Scroll to the bottom of the content"""
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
    def scroll_to_controls(self):
        """Scroll to the controls section"""
        if hasattr(self, 'controls_container'):
            # Get the position of the controls container
            controls_pos = self.controls_container.mapTo(self.main_content_widget, self.controls_container.rect().topLeft())
            # Scroll to show the controls
            self.scroll_area.ensureVisible(controls_pos.x(), controls_pos.y(), 0, 0)
            
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for scrolling"""
        # Handle common scroll shortcuts that work on both Windows and Linux
        if event.key() == Qt.Key_Home:
            self.scroll_to_top()
        elif event.key() == Qt.Key_End:
            self.scroll_to_bottom()
        elif event.key() == Qt.Key_PageUp:
            # Scroll up by one page
            current_value = self.scroll_area.verticalScrollBar().value()
            page_step = self.scroll_area.verticalScrollBar().pageStep()
            self.scroll_area.verticalScrollBar().setValue(current_value - page_step)
        elif event.key() == Qt.Key_PageDown:
            # Scroll down by one page
            current_value = self.scroll_area.verticalScrollBar().value()
            page_step = self.scroll_area.verticalScrollBar().pageStep()
            self.scroll_area.verticalScrollBar().setValue(current_value + page_step)
        else:
            # Pass other key events to the parent
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event - cleanup all threads in proper order"""
        print("[UI] Closing window - cleaning up...")
        
        # Accept the close event immediately to prevent freezing
        event.accept()
        
        try:
            # Stop all active processes immediately
            self.frame_timer.stop()
            
            # Stop automatic curation update timer
            if hasattr(self, 'curation_auto_timer'):
                self.curation_auto_timer.stop()
            
            # Reset state variables
            self.camera_running = False
            self.server_connected = False
            self.processing_frame = False
            
            # Clear displays immediately
            if hasattr(self, 'camera_display'):
                self.camera_display.clear_display()
            if hasattr(self, 'processed_display'):
                print("[UI] Cleaning up processed display...")
                # Clear ZMQ queue first to prevent blocking
                self.processed_display.clear_zmq_queue()
                self.processed_display.cleanup()
            
            # Force terminate all threads immediately to prevent core dumps
            threads_to_terminate = []
            
            if hasattr(self, 'ws_client') and self.ws_client is not None:
                self.ws_client.running = False
                self.ws_client.processing = False
                self.ws_client.close()
                threads_to_terminate.append(('WebSocket', self.ws_client))
            
            if hasattr(self, 'stt_thread') and self.stt_thread is not None:
                self.stt_thread.stop()
                threads_to_terminate.append(('STT', self.stt_thread))
            
            if hasattr(self, 'fft_thread') and self.fft_thread is not None:
                self.fft_thread.stop()
                threads_to_terminate.append(('FFT', self.fft_thread))
                
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                self.camera_thread.stop()
                threads_to_terminate.append(('Camera', self.camera_thread))
            
            # Give threads a brief moment to stop gracefully, then force terminate
            def force_terminate_remaining():
                for name, thread in threads_to_terminate:
                    if thread and thread.isRunning():
                        print(f"[UI] Force terminating {name} thread...")
                        thread.terminate()
                
                # Final cleanup after all threads are terminated
                def final_cleanup():
                    print("[UI] All threads terminated - final cleanup...")
                    QApplication.quit()
                
                QTimer.singleShot(200, final_cleanup)  # Final cleanup after 200ms
            
            QTimer.singleShot(300, force_terminate_remaining)  # Force terminate after 300ms
            
            print("[UI] Cleanup signals sent - threads will be force terminated if needed")
            
        except Exception as e:
            print(f"[UI] Error during cleanup: {e}")
            # Force immediate exit on error
            def emergency_exit():
                import os
                print("[UI] Emergency exit due to cleanup error")
                try:
                    os._exit(1)
                except:
                    import sys
                    sys.exit(1)
            
            QTimer.singleShot(100, emergency_exit)
            
        # Use QTimer to force termination if app doesn't quit quickly
        def force_exit():
            import os
            import sys
            print("[UI] Force terminating application")
            try:
                os._exit(0)
            except:
                sys.exit(0)
        
        QTimer.singleShot(2000, force_exit)  # Force exit after 2 seconds

    def update_curation_index(self):
        """Update the curation index on the server (manual button press)"""
        index = self.curation_spinbox.value()
        self._perform_curation_index_update(index, is_automatic=False)

    def _perform_curation_index_update(self, index, is_automatic=False):
        """Internal method to perform curation index update"""
        # Enable black frame mode before updating curation index
        if not self.black_frame_enabled:
            print(f"[UI] Enabling black frame mode for curation index update")
            self.black_frame_enabled = True
            self.processed_display.set_black_frame_mode(True)
            self.toggle_black_frame_button.setText("Disable Black Frame")
        
        # Clear ZMQ queue and display black frame immediately
        print(f"[UI] Clearing ZMQ queue and displaying black frame for curation index update")
        self.processed_display.clear_zmq_queue()
        
        # Handle button state only for manual updates
        if not is_automatic:
            # Disable the button during update
            self.curation_update_button.setEnabled(False)
            self.curation_update_button.setText("Updating...")
        
        update_type = "Automatic" if is_automatic else "Manual"
        self.status_bar.update_processing_status(f"{update_type} curation index update to {index}...")
        print(f"[UI] {update_type.lower()} curation index update: {index}")

        
        # Use QTimer to perform the update without blocking the UI
        def perform_update():
            try:
                # This will be called in the main thread but we'll use the WebSocket client's
                # built-in async handling which should be non-blocking
                print(f"[UI] Sending curation index update request for index {index}")
                
                # Store the update parameters for the success/failure callbacks
                self._pending_curation_index = index
                self._pending_curation_is_automatic = is_automatic
                
                # Call the WebSocket client's update method (this should be non-blocking)
                # We'll assume the WebSocket client handles this asynchronously
                import asyncio
                import threading
                
                def run_async_update():
                    success = False
                    message = "Unknown error"
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success, message = loop.run_until_complete(
                            self.ws_client.update_curation_index(index)
                        )
                        loop.close()
                        print(f"[UI] Async update completed: success={success}, message={message}")
                        
                    except Exception as e:
                        success = False
                        message = str(e)
                        print(f"[UI] Async update failed with exception: {e}")
                    finally:
                        # Use Qt signal to communicate back to main thread (this works from any thread)
                        print(f"[UI] Emitting signal with success={success}, message={message}")
                        self.curation_signal_handler.update_completed.emit(success, message)
                
                # Run the async operation in a separate thread
                thread = threading.Thread(target=run_async_update)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                print(f"[UI] Error in perform_update: {e}")
                # Use signal for error case too
                self.curation_signal_handler.update_completed.emit(False, str(e))
        
        # Use QTimer with 0 delay to run the update in the next event loop iteration
        QTimer.singleShot(50, perform_update)  # Small delay to ensure UI updates are processed
    
    def _handle_curation_update_result(self, success, message):
        """Handle the result of curation index update (called from main thread)"""
        try:
            print(f"[UI] Handling curation update result: success={success}, message={message}")
            index = getattr(self, '_pending_curation_index', 'unknown')
            is_automatic = getattr(self, '_pending_curation_is_automatic', False)
            
            update_type = "Automatic" if is_automatic else "Manual"
            
            if success:
                self.status_bar.update_processing_status(f"{update_type} curation index updated to {index}: {message}")
                print(f"[UI] Successfully updated curation index to {index} ({update_type.lower()})")
                
                # Automatically disable black frame mode on successful update (if configured)
                if self.black_frame_enabled and self.auto_disable_black_frame_after_curation_update:
                    print(f"[UI] Automatically disabling black frame mode after successful curation index update")
                    self.black_frame_enabled = False
                    self.processed_display.set_black_frame_mode(False)
                    self.toggle_black_frame_button.setText("Enable Black Frame")
            else:
                self.status_bar.update_processing_status(f"Failed to update curation index ({update_type.lower()}): {message}")
                print(f"[UI] Failed to update curation index ({update_type.lower()}): {message}")
                
        except Exception as e:
            self.status_bar.update_processing_status(f"Error updating curation index: {str(e)}")
            print(f"[UI] Error in _handle_curation_update_result: {e}")
        finally:
            # Re-enable the button only for manual updates
            try:
                if not getattr(self, '_pending_curation_is_automatic', False):
                    print(f"[UI] Re-enabling curation update button")
                    self.curation_update_button.setEnabled(True)
                    self.curation_update_button.setText("Update Curation Index")
                
                # Clean up the pending data
                if hasattr(self, '_pending_curation_index'):
                    delattr(self, '_pending_curation_index')
                if hasattr(self, '_pending_curation_is_automatic'):
                    delattr(self, '_pending_curation_is_automatic')
                print(f"[UI] Cleaned up pending curation data")
            except Exception as e:
                print(f"[UI] Error cleaning up: {e}")

    def perform_automatic_curation_update(self):
        """Perform automatic curation index update (called by timer)"""
        print("[UI] Automatic curation update timer triggered")
        
        if not CURATION_INDEX_AUTO_UPDATE:
            print("[UI] Skipping automatic curation update - disabled in config")
            return
            
        if not self.server_connected:
            print("[UI] Skipping automatic curation update - not connected to server")
            return
            
        print("[UI] Performing automatic curation index update...")
        
        # Get current curation index and increment it
        current_index = self.curation_spinbox.value()
        
        # Calculate next index (cycle through available indices)
        next_index = (current_index + 1) % (CURATION_INDEX_MAX + 1)
        
        # Update the spinbox value
        self.curation_spinbox.setValue(next_index)
        
        # Validate that the value is within range
        if next_index > CURATION_INDEX_MAX:
            print(f"[UI] Warning: Curation index {next_index} exceeds maximum {CURATION_INDEX_MAX}, clamping to maximum")
            next_index = CURATION_INDEX_MAX
            self.curation_spinbox.setValue(next_index)
        
        # Simulate the manual update process
        print(f"[UI] Automatic curation update: {current_index} -> {next_index}")
        self.status_bar.update_processing_status(f"Automatic curation update: {current_index} -> {next_index}")
        
        # Perform the actual update (similar to manual update but without button interference)
        self._perform_curation_index_update(next_index, is_automatic=True)

    def validate_curation_index(self, value):
        """Validate that the curation index is within the configured range"""
        if value > CURATION_INDEX_MAX:
            print(f"[UI] Warning: Manual curation index {value} exceeds maximum {CURATION_INDEX_MAX}, clamping to maximum")
            self.curation_spinbox.setValue(CURATION_INDEX_MAX)
        elif value < 0:
            print(f"[UI] Warning: Manual curation index {value} is negative, setting to 0")
            self.curation_spinbox.setValue(0)

    def toggle_automatic_curation_updates(self):
        """Toggle automatic curation index updates"""
        if self.curation_auto_timer.isActive():
            # Stop automatic updates
            self.curation_auto_timer.stop()
            self.toggle_auto_curation_button.setText("Start Auto Curation Updates")
            self.status_bar.update_processing_status("Automatic curation updates stopped")
            print("[UI] Automatic curation updates stopped")
        else:
            # Start automatic updates (only if connected to server)
            if self.server_connected:
                self.curation_auto_timer.start()
                self.toggle_auto_curation_button.setText("Stop Auto Curation Updates")
                self.status_bar.update_processing_status(f"Automatic curation updates started (every {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
                print(f"[UI] Automatic curation updates started (interval: {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
            else:
                self.status_bar.update_processing_status("Cannot start automatic updates - not connected to server")
                print("[UI] Cannot start automatic updates - not connected to server")

    def update_camera_index(self):
        """Update the camera device index"""
        new_index = self.camera_spinbox.value()
        
        if new_index == self.camera_device_index:
            self.status_bar.update_processing_status(f"Camera already using index {new_index}")
            return
            
        print(f"[UI] Updating camera index from {self.camera_device_index} to {new_index}")
        self.camera_update_button.setEnabled(False)
        self.camera_update_button.setText("Updating...")
        self.status_bar.update_processing_status(f"Updating camera to index {new_index}...")
        
        # Store the old state
        was_camera_running = self.camera_running
        
        try:
            # Stop camera if it's running
            if was_camera_running:
                self.stop_camera()
            
            # Update the index
            self.camera_device_index = new_index
            
            # Recreate camera thread with new index
            if hasattr(self, 'camera_thread'):
                self.camera_thread.stop()
                self.camera_thread.wait(1000)
                if self.camera_thread.isRunning():
                    self.camera_thread.terminate()
            
            self.camera_thread = CameraThread()
            self.camera_thread.device_index = self.camera_device_index
            self.camera_thread.frame_ready.connect(self.handle_camera_frame)
            
            # Restart camera if it was running
            if was_camera_running:
                self.start_camera()
                self.status_bar.update_processing_status(f"Camera updated to index {new_index} and restarted")
            else:
                self.status_bar.update_processing_status(f"Camera updated to index {new_index}")
                
            print(f"[UI] Successfully updated camera to index {new_index}")
            
        except Exception as e:
            print(f"[UI] Error updating camera index: {e}")
            self.status_bar.update_processing_status(f"Error updating camera: {e}")
        finally:
            self.camera_update_button.setEnabled(True)
            self.camera_update_button.setText("Update Camera")

    def update_mic_index(self):
        """Update the microphone device index"""
        new_index = self.mic_spinbox.value()
        
        if new_index == self.audio_device_index:
            self.status_bar.update_processing_status(f"Microphone already using index {new_index}")
            return
            
        print(f"[UI] Updating microphone index from {self.audio_device_index} to {new_index}")
        self.mic_update_button.setEnabled(False)
        self.mic_update_button.setText("Updating...")
        self.status_bar.update_processing_status(f"Updating microphone to index {new_index}...")
        
        # Store the old states
        was_stt_running = self.stt_active
        was_fft_running = self.fft_active
        
        try:
            # Stop audio threads if they're running
            if was_stt_running:
                self.toggle_stt()  # This will stop and recreate the thread
            if was_fft_running:
                self.toggle_fft()  # This will stop and recreate the thread
            
            # Update the index
            self.audio_device_index = new_index
            
            # Recreate audio threads with new index
            self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
            self.stt_thread.transcription_updated.connect(self.handle_transcription)
            
            self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
            self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
            
            # Restart audio threads if they were running
            if was_stt_running:
                self.toggle_stt()  # This will start the new thread
            if was_fft_running:
                self.toggle_fft()  # This will start the new thread
                
            if was_stt_running or was_fft_running:
                self.status_bar.update_processing_status(f"Microphone updated to index {new_index} and audio processing restarted")
            else:
                self.status_bar.update_processing_status(f"Microphone updated to index {new_index}")
                
            print(f"[UI] Successfully updated microphone to index {new_index}")
            
        except Exception as e:
            print(f"[UI] Error updating microphone index: {e}")
            self.status_bar.update_processing_status(f"Error updating microphone: {e}")
        finally:
            self.mic_update_button.setEnabled(True)
            self.mic_update_button.setText("Update Microphone")

    def refresh_device_lists(self):
        """Refresh the lists of available cameras and microphones"""
        print("[UI] Refreshing device lists...")
        self.refresh_devices_button.setEnabled(False)
        self.refresh_devices_button.setText("Refreshing...")
        
        try:
            self.available_cameras = detect_cameras()
            self.available_microphones = detect_microphones()
            
            print(f"[UI] Detected cameras: {[(idx, name, info) for idx, name, info in self.available_cameras]}")
            print(f"[UI] Detected microphones: {[f'{idx}: {name}' for idx, name in self.available_microphones]}")
            
            # Update camera spinbox tooltip
            if self.available_cameras:
                camera_list = [f"{idx}: {name} ({info})" for idx, name, info in self.available_cameras]
                self.camera_spinbox.setToolTip("Available cameras:\n" + "\n".join(camera_list))
            else:
                self.camera_spinbox.setToolTip("No cameras detected, but you can still try different indices")
            
            # Update microphone spinbox tooltip
            if self.available_microphones:
                mic_list = [f"{idx}: {name[:30]}..." if len(name) > 30 else f"{idx}: {name}" 
                           for idx, name in self.available_microphones]
                self.mic_spinbox.setToolTip("Available microphones:\n" + "\n".join(mic_list))
            else:
                self.mic_spinbox.setToolTip("No microphones detected, but you can still try different indices")
            
            self.status_bar.update_processing_status(f"Device refresh complete: {len(self.available_cameras)} cameras, {len(self.available_microphones)} microphones")
            print("[UI] Device lists refreshed successfully")
            
        except Exception as e:
            print(f"[UI] Error refreshing device lists: {e}")
            self.status_bar.update_processing_status(f"Error refreshing devices: {e}")
        finally:
            self.refresh_devices_button.setEnabled(True)
            self.refresh_devices_button.setText("Refresh Device Lists")

    def force_terminate(self):
        """Force terminate the application at the OS level"""
        print("[UI] Force terminating application...")
        try:
            # Get the current process ID
            pid = os.getpid()
            # Use os.kill to forcefully terminate the process
            os.kill(pid, 9)  # SIGKILL signal
        except Exception as e:
            print(f"[UI] Error during force termination: {e}")
            # If os.kill fails, try sys.exit as a fallback
            sys.exit(1)

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
