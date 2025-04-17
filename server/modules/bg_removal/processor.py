import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .models.modnet import MODNet


class BackgroundRemovalProcessor:
    """
    A processor class for background removal using MODNet.
    This class handles loading the model, preprocessing, and postprocessing of images.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the background removal processor.
        
        Args:
            model_path (str, optional): Path to the pretrained MODNet model.
                If None, will use the default path.
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                If None, will automatically detect.
        """
        # Set default model path if not provided
        if model_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'pretrained', 'modnet_webcam_portrait_matting.ckpt')
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # Load model
        self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the MODNet model from the specified path."""
        print(f'Loading MODNet from {model_path}...')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Initialize model
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        
        # Load weights
        if self.device == 'cuda':
            self.model = self.model.cuda()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set to evaluation mode
        self.model.eval()
        print(f'MODNet loaded successfully on {self.device}')
        
    def process_image(self, pil_image, return_alpha=False):
        """
        Process a PIL image to remove the background.
        
        Args:
            pil_image (PIL.Image): Input image as PIL Image
            return_alpha (bool): If True, return the alpha matte instead of the foreground
            
        Returns:
            PIL.Image: Processed image with background removed
        """
        # Convert PIL to numpy array
        frame_np = np.array(pil_image)
        
        # Resize image to a size that's divisible by 32
        h, w = frame_np.shape[:2]
        if w >= h:
            rh = 512
            rw = int(w / h * 512)
        else:
            rw = 512
            rh = int(h / w * 512)
        rh = rh - rh % 32
        rw = rw - rw % 32
        
        # Resize image
        frame_np = np.array(Image.fromarray(frame_np).resize((rw, rh), Image.LANCZOS))
        
        # Convert to PIL for transforms
        frame_PIL = Image.fromarray(frame_np)
        
        # Apply transforms
        frame_tensor = self.transforms(frame_PIL)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        if self.device == 'cuda':
            frame_tensor = frame_tensor.cuda()
        
        # Process with model
        with torch.no_grad():
            _, _, matte_tensor = self.model(frame_tensor, True)
        
        # Convert matte to numpy
        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
        
        # Generate output
        if return_alpha:
            # Return alpha matte
            alpha_np = matte_np * np.full(frame_np.shape, 255.0)
            result_np = alpha_np.astype(np.uint8)
        else:
            # Return foreground with white background
            fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            result_np = fg_np.astype(np.uint8)
        
        # Resize back to original size
        result_np = np.array(Image.fromarray(result_np).resize((w, h), Image.LANCZOS))
        
        # Convert back to PIL
        return Image.fromarray(result_np)
    
    def process_batch(self, pil_images, return_alpha=False):
        """
        Process a batch of PIL images to remove the background.
        
        Args:
            pil_images (list): List of PIL Images
            return_alpha (bool): If True, return the alpha matte instead of the foreground
            
        Returns:
            list: List of processed PIL Images with background removed
        """
        results = []
        for img in pil_images:
            results.append(self.process_image(img, return_alpha))
        return results


# Create a singleton instance for easy import
_processor = None

def get_processor(model_path=None, device=None):
    """
    Get or create a BackgroundRemovalProcessor instance.
    
    Args:
        model_path (str, optional): Path to the pretrained MODNet model.
        device (str, optional): Device to run the model on ('cuda' or 'cpu').
        
    Returns:
        BackgroundRemovalProcessor: The processor instance
    """
    global _processor
    if _processor is None:
        _processor = BackgroundRemovalProcessor(model_path, device)
    return _processor 