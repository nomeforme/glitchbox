import sys
import cv2
import numpy as np
from PIL import Image
import io
import json
import asyncio
import websockets
import uuid
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QCheckBox, QLabel)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QImage, QPixmap

class WebSocketClient(QThread):
    frame_received = Signal(np.ndarray)
    connection_error = Signal(str)
    
    def __init__(self, uri="ws://localhost:7860/api/ws"):
        super().__init__()
        self.uri = uri
        self.websocket = None
        self.running = False
        self.user_id = str(uuid.uuid4())
        # Base parameters for image generation
        self.params = {
            "prompt": "",
            "acid_settings": {
                "acid_strength": 0.4,
                "zoom_factor": 1.0,
                "do_acid_tracers": False,
                "do_acid_wobblers": False,
                "do_human_seg": True
            }
        }

    def update_settings(self, settings):
        self.params["acid_settings"].update(settings)

    async def _connect(self):
        try:
            full_uri = f"{self.uri}/{self.user_id}"
            self.websocket = await websockets.connect(full_uri)
            # Wait for initial messages from server
            msg = await self.websocket.recv()
            print(f"Connected: {msg}")
            msg = await self.websocket.recv()
            print(f"Status: {msg}")
            msg = await self.websocket.recv()
            print(f"Ready: {msg}")
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    async def send_frame(self, frame):
        if not self.websocket:
            return False

        try:
            # Convert frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return False

            # Send frame metadata and status
            await self.websocket.send(json.dumps({
                "status": "next_frame"
            }))
            
            # Send parameters with acid settings
            await self.websocket.send(json.dumps(self.params))
            
            # Send the actual frame data
            await self.websocket.send(buffer.tobytes())
            return True

        except Exception as e:
            self.connection_error.emit(str(e))
            self.websocket = None
            return False

    async def receive_processed_frame(self):
        if not self.websocket:
            print("[WebSocket] No active websocket connection")
            return None
            
        try:
            # Wait for server response
            print("[WebSocket] Waiting for server response...")
            msg = await self.websocket.recv()
            print(f"[WebSocket] Received message: {msg[:200]}...") # Truncate long messages
            
            try:
                data = json.loads(msg)
                print(f"[WebSocket] Parsed message status: {data.get('status')}")
                
                if data.get("status") == "send_frame":
                    print("[WebSocket] Server ready for next frame")
                    return None
                    
                # Expect binary frame data next
                print("[WebSocket] Waiting for frame data...")
                frame_data = await self.websocket.recv()
                
                if isinstance(frame_data, bytes):
                    print(f"[WebSocket] Received binary frame data: {len(frame_data)} bytes")
                    # Convert bytes to numpy array
                    np_img = np.frombuffer(frame_data, dtype=np.uint8)
                    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if img is not None:
                        print(f"[WebSocket] Successfully decoded frame: {img.shape}")
                        return img
                    else:
                        print("[WebSocket] Failed to decode frame data")
                else:
                    print(f"[WebSocket] Unexpected frame data type: {type(frame_data)}")
                
                return None
                    
            except json.JSONDecodeError as e:
                print(f"[WebSocket] Failed to parse message as JSON: {e}")
                return None
                
        except Exception as e:
            print(f"[WebSocket] Error receiving frame: {str(e)}")
            self.connection_error.emit(str(e))
            return None

    async def main_loop(self):
        if not await self._connect():
            self.connection_error.emit("Failed to connect to server")
            return
            
        while self.running:
            try:
                print("[WebSocket] Running main loop...")
                # Receive messages from server
                msg = await self.websocket.recv()
                try:
                    data = json.loads(msg)
                    print(f"[WebSocket] Received message: {data}")
                    if data.get("status") == "send_frame":
                        # Server is ready for next frame - but we don't have one to send yet
                        # The main thread will handle this when it calls send_frame
                        print("[WebSocket] Got send_frame status. Passing...")
                        pass
                    elif data.get("status") == "wait":
                        # Server is processing, continue waiting
                        print("[WebSocket] Got wait status. Passing...")
                        pass
                except json.JSONDecodeError:
                    # Message might be binary frame data
                    print(f"[WebSocket] Error - received non-JSON message...")
                    pass
                    
            except websockets.exceptions.ConnectionClosed:
                print("[WebSocket] Connection closed")
                self.websocket = None
                await asyncio.sleep(1)
                # Try to reconnect
                if not await self._connect():
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"[WebSocket] Error: {str(e)}")
                self.connection_error.emit(str(e))
                self.websocket = None
                await asyncio.sleep(1)

    def run(self):
        self.running = True
        asyncio.run(self.main_loop())

    def stop(self):
        self.running = False
        if self.websocket:
            asyncio.run(self.websocket.close())

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None

    def run(self):
        self.camera = cv2.VideoCapture(0)
        self.running = True

        while self.running:
            ret, frame = self.camera.read()
            if ret:
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)
            # Small delay to not flood the system
            time.sleep(0.03)  # ~30 FPS

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glitch Machine")
        self.setup_ui()
        
        # Initialize camera and websocket threads
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        
        self.ws_client = WebSocketClient()
        self.ws_client.frame_received.connect(self.update_processed_feed)
        self.ws_client.connection_error.connect(self.handle_connection_error)

        # Timer for frame updates
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.setInterval(100)  # 10 FPS for processing

        self.current_frame = None
        self.processing_frame = False

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Video feeds
        feeds_layout = QHBoxLayout()
        self.camera_label = QLabel()
        self.processed_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.processed_label.setMinimumSize(640, 480)
        feeds_layout.addWidget(self.camera_label)
        feeds_layout.addWidget(self.processed_label)
        layout.addLayout(feeds_layout)

        # Controls
        controls_layout = QVBoxLayout()

        # Start/Stop button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.start_button)

        # Sliders
        slider_layout = QVBoxLayout()
        
        # Acid Strength slider
        acid_layout = QHBoxLayout()
        acid_layout.addWidget(QLabel("Acid Strength:"))
        self.acid_slider = QSlider(Qt.Horizontal)
        self.acid_slider.setRange(0, 100)
        self.acid_slider.setValue(40)
        self.acid_slider.valueChanged.connect(self.update_settings)
        acid_layout.addWidget(self.acid_slider)
        slider_layout.addLayout(acid_layout)

        # Zoom Factor slider
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom Factor:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.update_settings)
        zoom_layout.addWidget(self.zoom_slider)
        slider_layout.addLayout(zoom_layout)

        controls_layout.addLayout(slider_layout)

        # Checkboxes
        checkbox_layout = QHBoxLayout()
        self.tracers_checkbox = QCheckBox("Acid Tracers")
        self.wobblers_checkbox = QCheckBox("Acid Wobblers")
        self.human_seg_checkbox = QCheckBox("Human Segmentation")
        self.human_seg_checkbox.setChecked(True)

        for checkbox in [self.tracers_checkbox, self.wobblers_checkbox, self.human_seg_checkbox]:
            checkbox.stateChanged.connect(self.update_settings)
            checkbox_layout.addWidget(checkbox)

        controls_layout.addLayout(checkbox_layout)
        layout.addLayout(controls_layout)

    def toggle_camera(self):
        if self.start_button.text() == "Start Camera":
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.camera_thread.start()
        self.ws_client.start()
        self.frame_timer.start()
        self.start_button.setText("Stop Camera")

    def stop_camera(self):
        self.frame_timer.stop()
        self.camera_thread.stop()
        self.ws_client.stop()
        self.camera_thread.wait()
        self.ws_client.wait()
        self.start_button.setText("Start Camera")
        self.camera_label.clear()
        self.processed_label.clear()

    def update_camera_feed(self, frame):
        self.current_frame = frame
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            640, 480, Qt.KeepAspectRatio))

    def update_processed_feed(self, frame):
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.processed_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            640, 480, Qt.KeepAspectRatio))
        self.processing_frame = False

    def process_frame(self):
        if self.current_frame is not None and not self.processing_frame:
            self.processing_frame = True
            asyncio.run(self.ws_client.send_frame(self.current_frame))

    def update_settings(self):
        settings = {
            "acid_strength": self.acid_slider.value() / 100,
            "zoom_factor": self.zoom_slider.value() / 100,
            "do_acid_tracers": self.tracers_checkbox.isChecked(),
            "do_acid_wobblers": self.wobblers_checkbox.isChecked(),
            "do_human_seg": self.human_seg_checkbox.isChecked()
        }
        self.ws_client.update_settings(settings)

    def handle_connection_error(self, error_msg):
        print(f"Connection error: {error_msg}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

# Need to import time for the short delay in the camera thread
import time

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 1280, 720)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()