from PySide6.QtCore import QThread, Signal
import asyncio
import websockets
import json
import numpy as np
import cv2
import uuid

class WebSocketClient(QThread):
    frame_received = Signal(np.ndarray)
    connection_error = Signal(str)
    settings_received = Signal(dict)
    status_changed = Signal(str)
    
    def __init__(self, uri="ws://localhost:7860"):
        super().__init__()
        self.uri = uri
        self.websocket = None
        self.running = False
        self.processing = False
        self.user_id = str(uuid.uuid4())
        self.settings = None
        self.current_frame = None
        # Initialize with empty parameters - will be populated from settings
        self.params = {}

    async def fetch_settings(self):
        """Fetch pipeline settings"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.uri}/api/settings") as response:
                    if response.status == 200:
                        self.settings = await response.json()
                        # Initialize parameters with default values from settings
                        self.initialize_parameters()
                        self.settings_received.emit(self.settings)
                        return True
                    else:
                        return False
        except Exception as e:
            self.connection_error.emit(f"Settings error: {e}")
            return False

    def initialize_parameters(self):
        """Initialize parameters with default values from settings"""
        if not self.settings:
            return

        params = self.settings.get('input_params', {}).get('properties', {})
        self.params = {
            param_id: param.get('default', 0)
            for param_id, param in params.items()
        }
        # Add acid_settings if they exist in defaults
        if any(key.startswith('acid_') for key in self.params):
            acid_params = {
                key: value 
                for key, value in self.params.items() 
                if key.startswith('acid_')
            }
            # Remove acid_ parameters from main params and put them in acid_settings
            for key in acid_params:
                del self.params[key]
            self.params['acid_settings'] = acid_params

        print("[WebSocket] Initialized parameters:", self.params)

    def update_settings(self, settings):
        """Update processing parameters"""
        if isinstance(settings, dict):
            param_id = next(iter(settings))  # Get the parameter ID
            value = settings[param_id]
            
            # Special handling for the acid_settings dictionary
            if param_id == 'acid_settings':
                # Directly assign the acid_settings dictionary, don't nest it again
                self.params['acid_settings'] = value
            # Check if this is an acid parameter
            elif param_id.startswith('acid_'):
                if 'acid_settings' not in self.params:
                    self.params['acid_settings'] = {}
                self.params['acid_settings'][param_id] = value
            else:
                self.params[param_id] = value
            print(f"[WebSocket] Updated parameter {param_id}: {value}")
            print("[WebSocket] Current parameters:", self.params)

    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            full_uri = f"{self.uri}/api/ws/{self.user_id}"
            self.websocket = await websockets.connect(full_uri)
            
            # Handle initial messages
            msg = await self.websocket.recv()
            data = json.loads(msg)
            if data.get('status') == 'connected':
                self.status_changed.emit('connected')
                return True
                
            return False
            
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    async def send_frame(self, frame):
        """Send a frame for processing"""
        if not self.websocket or not self.processing:
            return False

        try:
            if not self.processing:
                print("[WebSocket] Processing stopped, not sending frame")
                return False

            # Convert frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return False

            # Send next_frame signal
            await self.websocket.send(json.dumps({
                "status": "next_frame"
            }))
            
            # Send parameters
            await self.websocket.send(json.dumps(self.params))
            
            # Send frame data
            await self.websocket.send(buffer.tobytes())
            self.processing_frame = False
            return True

        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    async def receive_processed_frame(self):
        """Receive and process server response"""
        if not self.websocket:
            return None
            
        try:
            msg = await self.websocket.recv()
            data = json.loads(msg)
            status = data.get('status')
            self.status_changed.emit(status)
                
            if status == 'frame':
                # Get binary frame data
                frame_data = await self.websocket.recv()
                if isinstance(frame_data, bytes):
                    np_img = np.frombuffer(frame_data, dtype=np.uint8)
                    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                
            elif status == 'send_frame':
                # Server ready for next frame
                if self.current_frame is not None and self.processing:
                    await self.send_frame(self.current_frame)
                    
            elif status == 'error':
                self.connection_error.emit(data.get('message', 'Unknown error'))
                
            return None
                    
        except Exception as e:
            self.connection_error.emit(str(e))
            return None

    async def main_loop(self):
        """Main event loop"""
        try:
            # Fetch settings first
            if not await self.fetch_settings():
                self.connection_error.emit("Failed to get pipeline settings")
                return

            # Then establish WebSocket connection            
            if not await self._connect():
                self.connection_error.emit("Failed to connect to server")
                return
                
            while self.running:
                try:
                    if self.current_frame is not None and self.processing:
                        processed_frame = await self.receive_processed_frame()
                        if processed_frame is not None:
                            self.frame_received.emit(processed_frame)
                            
                    await asyncio.sleep(0.01)
                        
                except websockets.exceptions.ConnectionClosed:
                    if not self.running:  # If we're shutting down, don't try to reconnect
                        break
                    self.status_changed.emit("disconnected")
                    self.websocket = None
                    if not await self._connect():
                        await asyncio.sleep(2)
                except Exception as e:
                    if not self.running:  # If we're shutting down, don't report errors
                        break
                    self.connection_error.emit(str(e))
                    self.websocket = None
                    await asyncio.sleep(1)
        finally:
            # Cleanup if the loop exits for any reason
            if self.websocket and self.websocket.open:
                await self.websocket.close()
                self.websocket = None

    def run(self):
        """Start the WebSocket client thread"""
        self.running = True
        asyncio.run(self.main_loop())

    def start_camera(self):
        """Start processing frames"""
        self.processing = True
        self.processing_frame = False

    def stop_camera(self):
        """Stop camera and frame processing"""
        print("[WebSocket] Stopping camera")
        self.processing = False
        self.processing_frame = False
        self.current_frame = None
        
        # Use a timeout to prevent hanging if the server doesn't respond
        # Create event loop for stop signal with a timeout
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Send stop signal to server with timeout
            loop.run_until_complete(
                asyncio.wait_for(self._send_stop_signal(), timeout=2.0)
            )
        except asyncio.TimeoutError:
            print("[WebSocket] Stop signal timed out")
        except Exception as e:
            print(f"[WebSocket] Error sending stop signal: {e}")
        finally:
            loop.close()
            # Emit status change to update UI
            self.status_changed.emit("ready")

    async def _send_stop_signal(self):
        """Send stop signal to server"""
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(json.dumps({"status": "stop"}))
            except websockets.exceptions.ConnectionClosed:
                print("[WebSocket] Connection already closed")
            except Exception as e:
                print(f"[WebSocket] Error sending stop signal: {e}")

    def stop(self):
        """Stop the WebSocket client and cleanup resources"""
        print("[WebSocket] Stopping WebSocket client")
        if not self.running:
            print("[WebSocket] Already stopped")
            return
            
        self.running = False
        self.processing = False
        
        # Create event loop in this thread for cleanup with a timeout
        try:
            # Signal the main loop to stop
            self.running = False
            
            # Wait with timeout for the thread to finish
            if not self.wait(3000):  # 3 second timeout
                print("[WebSocket] Thread wait timed out, forcing termination")
                self.terminate()
            
            # Now we can safely close the websocket
            if self.websocket:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Set a timeout for the close operation
                    loop.run_until_complete(
                        asyncio.wait_for(
                            self._close_websocket(), 
                            timeout=2.0
                        )
                    )
                except asyncio.TimeoutError:
                    print("[WebSocket] Close operation timed out")
                except Exception as e:
                    print(f"[WebSocket] Error during close: {e}")
                finally:
                    loop.close()
                    self.websocket = None
        except Exception as e:
            print(f"[WebSocket] Error during cleanup: {e}")
            
    async def _close_websocket(self):
        """Safely close the websocket connection"""
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"[WebSocket] Error closing websocket: {e}")
        self.websocket = None