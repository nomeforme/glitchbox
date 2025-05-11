import os
import time
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

# Import the Generator from the Fast-SRGAN module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..models.fast_srgan.model import Generator

class FastSRGANProcessor:
    """
    A processor class for upscaling images using Fast-SRGAN.
    This class handles upscaling images with the Fast-SRGAN model.
    """
    
    def __init__(self, device=None, model_path=None, config_path=None):
        """
        Initialize the Fast-SRGAN upscaler processor.
        
        Args:
            device (str, optional): Device to run the processor on ('cuda', 'mps', or 'cpu').
                If None, will automatically select the best available device.
            model_path (str, optional): Path to the model weights file.
                If None, will use the default model path in the Fast-SRGAN module.
            config_path (str, optional): Path to the model configuration file.
                If None, will use the default config path in the Fast-SRGAN module.
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Get the directory of the Fast-SRGAN module
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoints_dir = os.path.join(current_dir, "checkpoints", "fast_srgan")
        
        # Set default paths if not provided
        if config_path is None:
            config_path = os.path.join(checkpoints_dir, "config.yaml")
        if model_path is None:
            model_path = os.path.join(checkpoints_dir, "model.pt")
            
        # Load model configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = Generator(self.config.generator)
        
        # Load model weights
        weights = torch.load(model_path, map_location="cpu")
        new_weights = {}
        for k, v in weights.items():
            new_weights[k.replace("_orig_mod.", "")] = v
        self.model.load_state_dict(new_weights)
        
        # Compile model for better performance
        try:
            self.model = torch.compile(self.model)
            print("Successfully compiled model with torch.compile()")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Default scale factor (Fast-SRGAN is designed for 2x upscaling)
        self.scale_factor = 2
        
    def set_scale_factor(self, factor):
        """
        Set the scale factor for upscaling.
        
        Args:
            factor (int or float): Scale factor to use for upscaling
        """
        if factor > 0:
            self.scale_factor = factor
        else:
            print(f"Invalid scale factor: {factor}. Using default: 2")
            self.scale_factor = 2
            
    def set_resample_method(self, method):
        """
        Set the resampling method to use for upscaling.
        Note: Fast-SRGAN uses a neural network for upscaling, so this method
        is kept for API consistency but doesn't actually change the behavior.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        # Fast-SRGAN uses a neural network for upscaling, so this method
        # doesn't actually change the behavior
        print(f"Note: Fast-SRGAN uses a neural network for upscaling. "
              f"Resampling method '{method}' is ignored.")
            
    def process_image(self, pil_image, scale_factor=None):
        """
        Upscale a PIL image using Fast-SRGAN.
        
        Args:
            pil_image (PIL.Image): Input image to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: Fast-SRGAN is designed for 2x upscaling.
                
        Returns:
            PIL.Image: The upscaled image
        """
        # Use provided parameters or defaults
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        # Fast-SRGAN is designed for 2x upscaling
        if scale != 2:
            print(f"Warning: Fast-SRGAN is designed for 2x upscaling. "
                  f"Requested factor {scale} will be treated as 2x.")
            scale = 2
        
        # Convert PIL image to tensor
        lr_image = np.array(pil_image)
        lr_image = (torch.from_numpy(lr_image) / 127.5) - 1.0
        lr_image = lr_image.permute(2, 0, 1).unsqueeze(dim=0).to(self.device)
        
        # Upscale the image
        with torch.no_grad():
            t1 = time.time()
            sr_image = self.model(lr_image)
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            sr_image = sr_image.cpu()
            sr_image = (sr_image + 1.0) / 2.0
            sr_image = sr_image.permute(0, 2, 3, 1).squeeze()
            sr_image = (sr_image * 255).numpy().astype(np.uint8)
        
        # Convert back to PIL image
        return Image.fromarray(sr_image)
    
    def process_batch(self, pil_images, scale_factor=None):
        """
        Upscale a batch of PIL images using Fast-SRGAN.
        
        Args:
            pil_images (list): List of PIL images to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: Fast-SRGAN is designed for 2x upscaling.
                
        Returns:
            list: List of upscaled PIL images
        """
        return [self.process_image(img, scale_factor) for img in pil_images]


def get_processor(device=None, model_path=None, config_path=None):
    """
    Factory function to create a FastSRGANProcessor instance.
    
    Args:
        device (str, optional): Device to run the processor on ('cuda', 'mps', or 'cpu').
            If None, will automatically select the best available device.
        model_path (str, optional): Path to the model weights file.
            If None, will use the default model path in the Fast-SRGAN module.
        config_path (str, optional): Path to the model configuration file.
            If None, will use the default config path in the Fast-SRGAN module.
            
    Returns:
        FastSRGANProcessor: An instance of the Fast-SRGAN upscaler processor
    """
    return FastSRGANProcessor(device=device, model_path=model_path, config_path=config_path) 