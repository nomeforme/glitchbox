from typing import Dict, Union
from uuid import UUID
import asyncio
from fastapi import WebSocket
from starlette.websockets import WebSocketState
import logging
import time
from types import SimpleNamespace

# Configure logging to include filename
logging.basicConfig(
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

Connections = Dict[UUID, Dict[str, Union[WebSocket, asyncio.Queue]]]


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(
        self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0
    ):
        start_time = time.time()
        await websocket.accept()
        user_count = self.get_user_count()
        logging.info(f"User count: {user_count}")
        if max_queue_size > 0 and user_count >= max_queue_size:
            logging.warning("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            raise ServerFullException("Server is full")
        logging.info(f"New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
        }
        await websocket.send_json(
            {"status": "connected", "message": "Connected"},
        )
        await websocket.send_json({"status": "wait"})
        await websocket.send_json({"status": "send_frame"})
        end_time = time.time()
        logging.info(f"Connection setup completed in {end_time - start_time:.2f} seconds")

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data: SimpleNamespace):
        start_time = time.time()
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            await queue.put(new_data)
            end_time = time.time()
            logging.info(f"Data update completed in {end_time - start_time:.2f} seconds")

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        start_time = time.time()
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                data = await queue.get()
                end_time = time.time()
                logging.info(f"Data retrieval completed in {end_time - start_time:.2f} seconds")
                return data
            except asyncio.QueueEmpty:
                end_time = time.time()
                logging.info(f"Empty queue check completed in {end_time - start_time:.2f} seconds")
                return None

    def delete_user(self, user_id: UUID):
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            if websocket.client_state == WebSocketState.CONNECTED:
                return user_session["websocket"]
        return None

    async def disconnect(self, user_id: UUID):
        websocket = self.get_websocket(user_id)
        if websocket:
            await websocket.close()
        self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: Dict):
        start_time = time.time()
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.send_json(data)
                end_time = time.time()
                logging.info(f"JSON send completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            end_time = time.time()
            logging.error(f"Error: Send json: {e} (took {end_time - start_time:.2f} seconds)")

    async def receive_json(self, user_id: UUID) -> Dict:
        start_time = time.time()
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                data = await websocket.receive_json()
                end_time = time.time()
                logging.info(f"JSON receive completed in {end_time - start_time:.2f} seconds")
                return data
        except Exception as e:
            end_time = time.time()
            logging.error(f"Error: Receive json: {e} (took {end_time - start_time:.2f} seconds)")

    async def receive_bytes(self, user_id: UUID) -> bytes:
        start_time = time.time()
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                data = await websocket.receive_bytes()
                end_time = time.time()
                logging.info(f"Bytes receive completed in {end_time - start_time:.2f} seconds")
                return data
        except Exception as e:
            end_time = time.time()
            logging.error(f"Error: Receive bytes: {e} (took {end_time - start_time:.2f} seconds)")
