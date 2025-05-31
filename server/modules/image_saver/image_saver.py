import os
import asyncio
import time
from datetime import datetime
from typing import Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageSaver:
    """
    Asynchronous image saver that queues images for saving without blocking the main thread.
    Creates timestamped subdirectories for each session.
    """
    
    def __init__(
        self,
        base_dir: str = "saved_images",
        image_format: str = "png",
        quality: int = 95,
        queue_size: int = 100,
        enabled: bool = True,
        debug: bool = False
    ):
        """
        Initialize the image saver.
        
        Args:
            base_dir: Base directory for saving images
            image_format: Image format (png, jpg, jpeg)
            quality: JPEG quality (1-100)
            queue_size: Maximum queue size
            enabled: Whether image saving is enabled
            debug: Enable debug logging
        """
        # Resolve the base directory path
        if os.path.isabs(base_dir):
            self.base_dir = base_dir
        else:
            # For relative paths, resolve relative to the current working directory
            self.base_dir = os.path.abspath(base_dir)
        
        self.image_format = image_format.lower()
        self.quality = quality
        self.queue_size = queue_size
        self.enabled = enabled
        self.debug = debug
        
        # Create session directory with timestamp
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.base_dir, f"session_{self.session_timestamp}")
        
        # Image counter for this session
        self.image_counter = 0
        
        # Async queue for images to save
        self.save_queue = None
        self.save_task = None
        
        # Ensure directory exists
        if self.enabled:
            try:
                os.makedirs(self.session_dir, exist_ok=True)
                if self.debug:
                    print(f"[ImageSaver] Session directory created: {self.session_dir}")
                    print(f"[ImageSaver] Resolved base directory: {self.base_dir}")
            except Exception as e:
                logger.error(f"Failed to create session directory {self.session_dir}: {e}")
                if self.debug:
                    print(f"[ImageSaver] Error creating directory: {e}")
                self.enabled = False
    
    async def start(self):
        """Start the image saver background task."""
        if not self.enabled:
            return
            
        self.save_queue = asyncio.Queue(maxsize=self.queue_size)
        self.save_task = asyncio.create_task(self._save_worker())
        
        if self.debug:
            print(f"[ImageSaver] Started with queue size {self.queue_size}")
    
    async def stop(self):
        """Stop the image saver and wait for remaining images to be saved."""
        if not self.enabled or not self.save_task:
            return
            
        # Signal the worker to stop
        await self.save_queue.put(None)
        
        # Wait for the worker to finish
        await self.save_task
        
        if self.debug:
            print(f"[ImageSaver] Stopped. Total images saved: {self.image_counter}")
    
    async def save_image(self, image: Image.Image, metadata: Optional[dict] = None):
        """
        Queue an image for saving.
        
        Args:
            image: PIL Image to save
            metadata: Optional metadata to include in filename or save separately
        """
        if not self.enabled or not self.save_queue:
            return
            
        try:
            # Put image in queue without blocking
            self.save_queue.put_nowait((image.copy(), metadata, time.time()))
            
            if self.debug:
                print(f"[ImageSaver] Queued image for saving (queue size: {self.save_queue.qsize()})")
                
        except asyncio.QueueFull:
            if self.debug:
                print(f"[ImageSaver] Queue full, dropping image")
            logger.warning("Image save queue is full, dropping image")
    
    async def _save_worker(self):
        """Background worker that saves images from the queue."""
        while True:
            try:
                # Get next item from queue
                item = await self.save_queue.get()
                
                # Check for stop signal
                if item is None:
                    break
                
                image, metadata, timestamp = item
                
                # Generate filename
                self.image_counter += 1
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H%M%S_%f")[:-3]  # milliseconds
                filename = f"image_{self.image_counter:06d}_{timestamp_str}.{self.image_format}"
                filepath = os.path.join(self.session_dir, filename)
                
                # Save the image
                await self._save_image_to_disk(image, filepath, metadata)
                
                # Mark task as done
                self.save_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in image save worker: {e}")
                if self.debug:
                    print(f"[ImageSaver] Error saving image: {e}")
    
    async def _save_image_to_disk(self, image: Image.Image, filepath: str, metadata: Optional[dict]):
        """Save image to disk in a thread to avoid blocking."""
        def _save():
            try:
                if self.image_format in ["jpg", "jpeg"]:
                    # Convert RGBA to RGB for JPEG
                    if image.mode == "RGBA":
                        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                        rgb_image.paste(image, mask=image.split()[-1])
                        rgb_image.save(filepath, format="JPEG", quality=self.quality)
                    else:
                        image.save(filepath, format="JPEG", quality=self.quality)
                else:
                    image.save(filepath, format="PNG")
                
                if self.debug:
                    print(f"[ImageSaver] Saved: {filepath}")
                    
                # Save metadata if provided
                if metadata:
                    metadata_path = filepath.replace(f".{self.image_format}", "_metadata.txt")
                    with open(metadata_path, "w") as f:
                        for key, value in metadata.items():
                            f.write(f"{key}: {value}\n")
                            
            except Exception as e:
                logger.error(f"Error saving image to {filepath}: {e}")
                if self.debug:
                    print(f"[ImageSaver] Error saving to {filepath}: {e}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _save)
    
    def get_session_info(self) -> dict:
        """Get information about the current session."""
        return {
            "session_dir": self.session_dir,
            "session_timestamp": self.session_timestamp,
            "images_saved": self.image_counter,
            "queue_size": self.save_queue.qsize() if self.save_queue else 0,
            "enabled": self.enabled
        }


# Factory function to create image saver instance
def get_image_saver(
    base_dir: str = "server/output",
    image_format: str = "png", 
    quality: int = 95,
    queue_size: int = 100,
    enabled: bool = True,
    debug: bool = False
) -> ImageSaver:
    """
    Factory function to create an ImageSaver instance.
    
    Args:
        base_dir: Base directory for saving images
        image_format: Image format (png, jpg, jpeg)
        quality: JPEG quality (1-100)
        queue_size: Maximum queue size
        enabled: Whether image saving is enabled
        debug: Enable debug logging
        
    Returns:
        ImageSaver instance
    """
    return ImageSaver(
        base_dir=base_dir,
        image_format=image_format,
        quality=quality,
        queue_size=queue_size,
        enabled=enabled,
        debug=debug
    ) 