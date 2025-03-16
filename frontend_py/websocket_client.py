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

    async def fetch_settings(self):
        """Fetch pipeline settings"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.uri}/api/settings") as response:
                    if response.status == 200:
                        self.settings = await response.json()
                        self.settings_received.emit(self.settings)
                        return True
                    else:
                        return False
        except Exception as e:
            self.connection_error.emit(f"Settings error: {e}")
            return False

    def update_settings(self, settings):
        """Update processing parameters"""
        if isinstance(settings, dict):
            self.params["acid_settings"].update(settings)

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
            self.websocket = None
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

    def stop_camera(self):
        """Stop processing frames"""
        self.processing = False

    def stop(self):
        """Stop the WebSocket client and cleanup resources"""
        print("[WebSocket] Stopping WebSocket client")
        self.running = False
        self.processing = False
        
        # Create event loop in this thread for cleanup
        try:
            # Signal the main loop to stop
            self.running = False
            # Wait for the thread to finish its current work
            self.wait()
            
            # Now we can safely close the websocket
            if self.websocket:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Set a timeout for the close operation
                    loop.run_until_complete(asyncio.wait_for(self.websocket.close(), timeout=2.0))
                except asyncio.TimeoutError:
                    print("[WebSocket] Close operation timed out")
                except Exception as e:
                    print(f"[WebSocket] Error during close: {e}")
                finally:
                    loop.stop()
                    loop.close()
                    self.websocket = None
        except Exception as e:
            print(f"[WebSocket] Error during cleanup: {e}")