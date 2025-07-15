from PySide6.QtCore import QThread, Signal
import asyncio
import websockets
import json
import numpy as np
import cv2
import uuid
import os
import time

IS_WINDOWS = os.name == 'nt'

if not IS_WINDOWS:
    import uvloop

class WebSocketClient(QThread):
    frame_received = Signal(np.ndarray)
    connection_error = Signal(str)
    settings_received = Signal(dict)
    status_changed = Signal(str)
    
    def __init__(self, uri="ws://localhost:7860", max_retries=10, initial_retry_delay=1.0):
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
        # Polling configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.retry_count = 0
        self.retry_delay = initial_retry_delay
        self.connection_successful = False

    def reset_connection_state(self):
        """Reset connection state for clean reconnection"""
        print("[WebSocket] Resetting connection state...")
        self.websocket = None
        self.running = False
        self.processing = False
        self.user_id = str(uuid.uuid4())  # Generate new user ID
        self.settings = None
        self.current_frame = None
        self.params = {}
        self.retry_count = 0
        self.retry_delay = self.initial_retry_delay
        self.connection_successful = False
        print(f"[WebSocket] New user ID: {self.user_id}")

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

    def update_prompt(self, text):
        """Specifically update the prompt parameter with text from STT"""
        if 'prompt' in self.params:
            self.params['prompt'] = text
            print(f"[WebSocket] Updated prompt: {text}")
        else:
            print("[WebSocket] Warning: 'prompt' parameter not found in initialized parameters")

    async def update_curation_index(self, curation_index):
        """Update the curation index on the server"""
        import aiohttp
        try:
            # Extract base URL from WebSocket URI
            base_url = self.uri.replace('ws://', 'http://').replace('wss://', 'https://')
            url = f"{base_url}/api/update_curation_index"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"curation_index": curation_index}) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"[WebSocket] Curation index updated successfully: {result}")
                        return True, result.get('message', 'Success')
                    else:
                        error_text = await response.text()
                        print(f"[WebSocket] Failed to update curation index: {error_text}")
                        return False, f"Server error: {response.status}"
        except Exception as e:
            print(f"[WebSocket] Error updating curation index: {e}")
            return False, f"Network error: {str(e)}"

    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            full_uri = f"{self.uri}/api/ws/{self.user_id}"
            print(f"[WebSocket] Attempting to connect to {full_uri}")
            self.websocket = await websockets.connect(full_uri)
            
            # Handle initial messages
            msg = await self.websocket.recv()
            data = json.loads(msg)
            if data.get('status') == 'connected':
                self.status_changed.emit('connected')
                self.connection_successful = True
                self.retry_count = 0  # Reset retry count on successful connection
                self.retry_delay = self.initial_retry_delay  # Reset retry delay
                print("[WebSocket] Successfully connected and received connected status")
                return True
                
            print("[WebSocket] Connection attempt failed - did not receive connected status")
            return False
            
        except Exception as e:
            self.connection_error.emit(str(e))
            print(f"[WebSocket] Connection attempt failed with error: {e}")
            return False

    async def _poll_connection(self):
        """Poll for connection with exponential backoff"""
        while self.running and not self.connection_successful and self.retry_count < self.max_retries:
            self.status_changed.emit(f"Retrying connection (attempt {self.retry_count + 1}/{self.max_retries})...")
            print(f"[WebSocket] Retrying connection (attempt {self.retry_count + 1}/{self.max_retries})...")
            
            # Try to connect
            if await self._connect():
                print("[WebSocket] Connection successful")
                return True
                
            # Increment retry count and calculate next delay with exponential backoff
            self.retry_count += 1
            self.retry_delay = min(self.initial_retry_delay * (2 ** (self.retry_count - 1)), 30)  # Cap at 30 seconds
            
            # Wait before next retry
            print(f"[WebSocket] Waiting {self.retry_delay} seconds before next attempt...")
            await asyncio.sleep(self.retry_delay)
            
        if not self.connection_successful:
            self.status_changed.emit("Connection failed after maximum retries")
            self.connection_error.emit("Connection failed after maximum retries")
            return False
            
        return True

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
            # Fetch settings first with polling
            settings_success = False
            while self.running and not settings_success:
                settings_success = await self.fetch_settings()
                if not settings_success:
                    self.status_changed.emit("Retrying to fetch settings...")
                    await asyncio.sleep(self.initial_retry_delay)
            
            if not settings_success:
                self.connection_error.emit("Failed to get pipeline settings after multiple attempts")
                return

            # Then establish WebSocket connection with polling
            if not await self._poll_connection():
                return
                
            while self.running:
                try:
                    # Check running state more frequently
                    if not self.running:
                        print("[WebSocket] Main loop stopping - running flag is False")
                        break
                        
                    if self.current_frame is not None and self.processing:
                        processed_frame = await self.receive_processed_frame()
                        if processed_frame is not None and self.running:  # Check again after processing
                            self.frame_received.emit(processed_frame)
                    
                    # Short sleep with running check
                    await asyncio.sleep(0.01)
                        
                except websockets.exceptions.ConnectionClosed:
                    if not self.running:  # If we're shutting down, don't try to reconnect
                        print("[WebSocket] Connection closed during shutdown - exiting gracefully")
                        break
                    print("[WebSocket] Connection closed unexpectedly - attempting reconnect")
                    self.status_changed.emit("disconnected")
                    self.websocket = None
                    self.connection_successful = False
                    # Reset retry count and delay for fresh polling attempt
                    self.retry_count = 0
                    self.retry_delay = self.initial_retry_delay
                    # Try to reconnect with polling
                    if not await self._poll_connection():
                        break
                except Exception as e:
                    if not self.running:  # If we're shutting down, don't report errors
                        print(f"[WebSocket] Exception during shutdown (expected): {e}")
                        break
                    print(f"[WebSocket] Unexpected error in main loop: {e}")
                    self.connection_error.emit(str(e))
                    self.websocket = None
                    self.connection_successful = False
                    # Reset retry count and delay for fresh polling attempt
                    self.retry_count = 0
                    self.retry_delay = self.initial_retry_delay
                    # Try to reconnect with polling
                    if not await self._poll_connection():
                        break
        finally:
            # Cleanup if the loop exits for any reason
            if self.websocket:
                try:
                    if hasattr(self.websocket, 'open') and self.websocket.open:
                        await self.websocket.close()
                except Exception as e:
                    print(f"[WebSocket] Error during cleanup close: {e}")
                finally:
                    self.websocket = None

    def run(self):
        """Start the WebSocket client thread"""
        self.running = True
        if IS_WINDOWS:
            asyncio.run(self.main_loop())
        else:
            uvloop.install()
            uvloop.run(self.main_loop())            

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
        
        # Emit status change immediately
        self.status_changed.emit("ready")
        
        # Try to send stop signal but don't block if it fails
        try:
            # Only attempt to send stop signal if we have an active connection
            if self.websocket and hasattr(self.websocket, 'open') and self.websocket.open:
                # Create a brief event loop with very short timeout
                if IS_WINDOWS:
                    loop = asyncio.new_event_loop()
                else:
                    loop = uvloop.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Send stop signal to server with very short timeout
                    loop.run_until_complete(
                        asyncio.wait_for(self._send_stop_signal(), timeout=0.5)
                    )
                    print("[WebSocket] Stop signal sent successfully")
                except asyncio.TimeoutError:
                    print("[WebSocket] Stop signal timed out (non-critical)")
                except Exception as e:
                    print(f"[WebSocket] Error sending stop signal (non-critical): {e}")
                finally:
                    loop.close()
            else:
                print("[WebSocket] No active connection, skip sending stop signal")
        except Exception as e:
            print(f"[WebSocket] Error during stop camera cleanup (non-critical): {e}")

    async def _send_stop_signal(self):
        """Send stop signal to server"""
        if self.websocket:
            try:
                # Check if websocket is open using the correct method
                if hasattr(self.websocket, 'open') and self.websocket.open:
                    await self.websocket.send(json.dumps({"status": "stop"}))
                else:
                    print("[WebSocket] Connection is not open, cannot send stop signal")
            except websockets.exceptions.ConnectionClosed:
                print("[WebSocket] Connection already closed")
            except AttributeError as e:
                print(f"[WebSocket] Websocket object missing expected attribute: {e}")
            except Exception as e:
                print(f"[WebSocket] Error sending stop signal: {e}")

    def close(self):
        """Close the WebSocket connection gracefully (called from main thread)"""
        print("[WebSocket] Closing WebSocket connection gracefully")
        
        # Signal to stop processing and running
        self.running = False
        self.processing = False
        
        # Clear current frame to prevent new processing
        self.current_frame = None
        
        print("[WebSocket] Graceful close initiated - signals set")

    def stop(self):
        """Stop the WebSocket client and cleanup resources"""
        print("[WebSocket] Stopping WebSocket client")
        if not self.running:
            print("[WebSocket] Already stopped")
            return
            
        self.running = False
        self.processing = False
        self.current_frame = None
        
        # Emit disconnected status immediately
        self.status_changed.emit("disconnected")
        
        # Create event loop in this thread for cleanup with a timeout
        try:
            # Wait with shorter timeout for the thread to finish gracefully
            if not self.wait(1000):  # 1 second timeout
                print("[WebSocket] Thread wait timed out, forcing termination")
                self.terminate()
                # Give termination a moment to complete
                self.wait(500)
            else:
                print("[WebSocket] Thread stopped gracefully")
            
            # Now we can safely close the websocket if it wasn't closed already
            if self.websocket:
                try:
                    # Try a quick close operation with minimal timeout
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                self._close_websocket(), 
                                timeout=1.0  # Shorter timeout
                            )
                        )
                    except asyncio.TimeoutError:
                        print("[WebSocket] Close operation timed out - forcing close")
                    except Exception as e:
                        print(f"[WebSocket] Error during close: {e}")
                    finally:
                        loop.close()
                        self.websocket = None
                except Exception as e:
                    print(f"[WebSocket] Error creating event loop for close: {e}")
                    self.websocket = None
        except Exception as e:
            print(f"[WebSocket] Error during cleanup: {e}")
        finally:
            # Ensure websocket is None regardless of errors
            self.websocket = None
            print("[WebSocket] WebSocket client stopped and cleaned up")
            
    async def _close_websocket(self):
        """Safely close the websocket connection"""
        if self.websocket:
            try:
                if hasattr(self.websocket, 'open') and self.websocket.open:
                    await self.websocket.close()
                else:
                    print("[WebSocket] Connection already closed or not open")
            except Exception as e:
                print(f"[WebSocket] Error closing websocket: {e}")
            finally:
                self.websocket = None