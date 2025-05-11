import os
import time
import numpy as np
from PIL import Image

class PilUpscalerProcessor:
    """
    A processor class for upscaling images using PIL.
    This class handles upscaling images with various resampling methods.
    """
    
    def __init__(self, device=None):
        """
        Initialize the PIL upscaler processor.
        
        Args:
            device (str, optional): Device to run the processor on ('cuda' or 'cpu').
                Not used for PIL but kept for consistency with other processors.
        """
        # Store device for consistency with other processors
        self.device = device
        
        # Default resampling method
        self.resample_method = Image.LANCZOS
        
        # Default scale factor
        self.scale_factor = 2
        
    def set_resample_method(self, method):
        """
        Set the resampling method to use for upscaling.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        method_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        if method.lower() in method_map:
            self.resample_method = method_map[method.lower()]
        else:
            print(f"Unknown resampling method: {method}. Using default: LANCZOS")
            self.resample_method = Image.LANCZOS
            
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
            
    def process_image(self, pil_image, scale_factor=None, resample_method=None):
        """
        Upscale a PIL image.
        
        Args:
            pil_image (PIL.Image): Input image to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
            resample_method (str, optional): Resampling method to use.
                If None, uses the default set in the processor.
                
        Returns:
            PIL.Image: The upscaled image
        """
        # Use provided parameters or defaults
        scale = scale_factor if scale_factor is not None else self.scale_factor
        resample = resample_method if resample_method is not None else self.resample_method
        
        # Calculate new dimensions
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        
        # Upscale the image
        upscaled_img = pil_image.resize((new_width, new_height), resample=resample)
        
        return upscaled_img
    
    def process_batch(self, pil_images, scale_factor=None, resample_method=None):
        """
        Upscale a batch of PIL images.
        
        Args:
            pil_images (list): List of PIL images to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
            resample_method (str, optional): Resampling method to use.
                If None, uses the default set in the processor.
                
        Returns:
            list: List of upscaled PIL images
        """
        return [self.process_image(img, scale_factor, resample_method) for img in pil_images]


def get_processor(device=None):
    """
    Factory function to create a PilUpscalerProcessor instance.
    
    Args:
        device (str, optional): Device to run the processor on ('cuda' or 'cpu').
            Not used for PIL but kept for consistency with other processors.
            
    Returns:
        PilUpscalerProcessor: An instance of the PIL upscaler processor
    """
    return PilUpscalerProcessor(device=device) 