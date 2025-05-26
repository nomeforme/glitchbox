from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional, List, Any
import torch
import time
import asyncio
import json
from pydantic import BaseModel
from modules.prompt_travel.prompt_travel import PromptTravel

# Create a FastAPI router
router = APIRouter()

class PromptTravelRequest(BaseModel):
    user_id: str
    prompt: str
    target_prompt: str
    factor: float

class EmbeddingsServiceManager:
    """
    Asynchronous embeddings service for prompt travel.
    
    This service handles prompt travel operations using async methods,
    providing a more efficient and FastAPI-compatible approach than multiprocessing.
    """
    
    def __init__(self):
        self.prompt_travel = None
        self.embeddings_cache = {}
        self.pending_requests = {}
        self.cache_lock = asyncio.Lock()
        self.initialized = False
        self.device = None
        
    async def initialize(self, text_encoder, tokenizer, device="cuda"):
        """Initialize the prompt travel module with the given models."""
        if self.initialized:
            return
            
        self.device = device
        self.prompt_travel = PromptTravel(
            text_encoder=text_encoder,
            tokenizer=tokenizer
        )
        self.initialized = True
        print("Embeddings service initialized")
    
    async def process_prompt_travel(self, user_id: str, prompt: str, target_prompt: str, factor: float):
        """
        Process a prompt travel request asynchronously.
        
        Args:
            user_id: User identifier
            prompt: Source prompt
            target_prompt: Target prompt
            factor: Interpolation factor between 0 and 1
        """
        if not self.initialized or not self.prompt_travel:
            print("Error: Embeddings service not initialized")
            print(f"Prompt travel: {self.prompt_travel}")
            print(f"Initialized: {self.initialized}")
            return
            
        if not prompt or not target_prompt:
            print("Error: Missing prompt or target_prompt")
            return
            
        # Add to pending requests
        async with self.cache_lock:
            self.pending_requests[user_id] = {
                "prompt": prompt,
                "target_prompt": target_prompt,
                "factor": factor,
                "timestamp": time.time()
            }
        
        # Process in the background, without waiting
        asyncio.create_task(self._calculate_embeddings(user_id))
    
    async def _calculate_embeddings(self, user_id: str):
        """Calculate embeddings for a pending request"""
        async with self.cache_lock:
            if user_id not in self.pending_requests:
                return
            
            request = self.pending_requests.pop(user_id)
            
        try:
            # Run the embedding calculation in a threadpool to avoid blocking
            # Using loop.run_in_executor which is available in older Python versions
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._calculate_embeddings_sync,
                request["prompt"],
                request["target_prompt"],
                request["factor"]
            )
            
            if result:
                prompt_embeds, negative_prompt_embeds = result
                
                # Store the results in the cache
                async with self.cache_lock:
                    self.embeddings_cache[user_id] = {
                        "prompt_embeds": prompt_embeds.detach().cpu(),
                        "negative_prompt_embeds": negative_prompt_embeds.detach().cpu(),
                        "timestamp": time.time()
                    }
        except Exception as e:
            print(f"Error calculating embeddings: {e}")
    
    def _calculate_embeddings_sync(self, prompt: str, target_prompt: str, factor: float):
        """
        Synchronous method to calculate embeddings, to be run in a thread pool.
        """
        try:
            # Get prompt embeddings for both source and target prompts
            embeds_from, neg_embeds = self.prompt_travel.encode_prompt(
                prompt=prompt,
                device=torch.device(self.device),
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            
            embeds_to, _ = self.prompt_travel.encode_prompt(
                prompt=target_prompt,
                device=torch.device(self.device),
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            
            # Interpolate between the embeddings using the factor
            interpolated_embeds = self.prompt_travel.interpolate_embeddings(
                embeds_from=embeds_from,
                embeds_to=embeds_to,
                factor=factor,
            )
            
            return interpolated_embeds, neg_embeds
        except Exception as e:
            print(f"Error in sync embedding calculation: {e}")
            return None
    
    async def get_embeddings(self, user_id: str):
        """Get cached embeddings for a user if available"""
        async with self.cache_lock:
            if user_id in self.embeddings_cache:
                data = self.embeddings_cache[user_id]
                return data.get("prompt_embeds"), data.get("negative_prompt_embeds")
        return None
    
    async def clear_cache(self, older_than_seconds=3600):
        """Clear old entries from the cache"""
        current_time = time.time()
        async with self.cache_lock:
            # Identify keys to remove
            keys_to_remove = []
            for user_id, data in self.embeddings_cache.items():
                if current_time - data.get("timestamp", 0) > older_than_seconds:
                    keys_to_remove.append(user_id)
            
            # Remove the identified keys
            for key in keys_to_remove:
                del self.embeddings_cache[key]


# Create a singleton instance of the service
embeddings_service = EmbeddingsServiceManager()

# WebSocket handlers for embeddings service
connected_clients = {}

@router.websocket("/ws/prompt-travel/{user_id}")
async def websocket_prompt_travel(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for prompt travel requests and notifications"""
    await websocket.accept()
    connected_clients[user_id] = websocket
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "prompt_travel_request":
                # Process a prompt travel request
                request_data = message.get("data", {})
                await embeddings_service.process_prompt_travel(
                    user_id=user_id,
                    prompt=request_data.get("prompt", ""),
                    target_prompt=request_data.get("target_prompt", ""),
                    factor=request_data.get("factor", 0.0)
                )
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "acknowledgement",
                    "message": "Processing prompt travel request"
                })
                
            elif message.get("type") == "get_embeddings":
                # Fetch embeddings for the user
                embeddings = await embeddings_service.get_embeddings(user_id)
                
                if embeddings:
                    # Convert tensors to base64 or another serializable format
                    # Here we just send a notification that embeddings are ready
                    await websocket.send_json({
                        "type": "embeddings_ready",
                        "message": "Embeddings are ready for use"
                    })
                else:
                    await websocket.send_json({
                        "type": "no_embeddings",
                        "message": "No embeddings available"
                    })
    
    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {user_id}")
    except Exception as e:
        print(f"Error in embeddings service websocket: {e}")
    finally:
        if user_id in connected_clients:
            del connected_clients[user_id]

# REST API endpoints for prompt travel
@router.post("/api/prompt-travel")
async def request_prompt_travel(request: PromptTravelRequest):
    """HTTP endpoint to request prompt travel embeddings generation"""
    await embeddings_service.process_prompt_travel(
        user_id=request.user_id,
        prompt=request.prompt,
        target_prompt=request.target_prompt,
        factor=request.factor
    )
    return {"status": "processing", "message": "Prompt travel request is being processed"}

# Background task for cleaning the cache
async def periodic_cache_cleaner():
    """Background task to periodically clean old entries from the cache"""
    while True:
        try:
            # Wait for 30 minutes
            await asyncio.sleep(30 * 60)
            # Clean entries older than 1 hour
            await embeddings_service.clear_cache(older_than_seconds=60 * 60)
            print("Cleaned old entries from embeddings cache")
        except Exception as e:
            print(f"Error in cache cleaner: {e}")
            await asyncio.sleep(60)  # Wait a bit and retry

# Create a function to start background tasks that will be called from main.py
async def start_background_tasks():
    """Start background tasks for the embeddings service"""
    asyncio.create_task(periodic_cache_cleaner())

# Add this router to the main app in main.py with:
# app.include_router(embeddings_service_router, prefix="/embeddings") 