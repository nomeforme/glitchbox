import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from ..models.rsvr.inference_arch import RVSR

class RvsrUpscalerProcessor:
    """
    A processor class for upscaling images using the RVSR model.
    This class handles upscaling images using the RVSR deep learning model.
    """
    
    def __init__(self, device=None):
        """
        Initialize the RVSR upscaler processor.
        Args:
            device (str, optional): Device to run the processor on ('cuda' or 'cpu').
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        # Default scale factor
        self.scale_factor = 4  # RVSR is trained for x4 upscaling
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                              std=[0.5, 0.5, 0.5])  # RVSR uses [-1, 1] normalization
        ])
        
    def _load_model(self):
        model = RVSR(sr_rate=4, N=16)  # Using default parameters from the notebook
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'checkpoints/rvsr/RVSR_rep.pth'
        )
        checkpoint_path = os.path.abspath(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        
        # Convert model to float16
        model = model.half()
        
        try:
            print("Compiling RVSR model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            print("RVSR model compilation complete!")
        except Exception as e:
            print(f"Failed to compile RVSR model: {str(e)}")
            print("Falling back to uncompiled model")

        return model
        
    def set_scale_factor(self, factor):
        if factor == 4:  # RVSR only supports x4 upscaling
            self.scale_factor = factor
        else:
            print(f"RVSR only supports x4 upscaling. Using default: 4")
            self.scale_factor = 4
            
    def set_resample_method(self, method):
        """
        Set the resampling method to use for upscaling.
        Note: RVSR uses a neural network for upscaling, so this method
        is kept for API consistency but doesn't actually change the behavior.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        # RVSR uses a neural network for upscaling, so this method
        # doesn't actually change the behavior
        print(f"Note: RVSR uses a neural network for upscaling. "
              f"Resampling method '{method}' is ignored.")
            
    def process_image(self, pil_image, scale_factor=None):
        """
        Process a single image using RVSR.
        Args:
            pil_image (PIL.Image): Input image to upscale
            scale_factor (int, optional): Scale factor (must be 4 for RVSR)
        Returns:
            PIL.Image: Upscaled image
        """
        if scale_factor is not None and scale_factor != 4:
            print("Warning: RVSR only supports x4 upscaling. Ignoring provided scale factor.")
        
        # Preprocess image
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            t1 = time.time()
            output = self.model(img_tensor)
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            
        # Postprocess output
        output = output.squeeze(0).cpu()
        output = output * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]
        output = torch.clamp(output, 0, 1)
        output = transforms.ToPILImage()(output)
        
        return output
    
    def process_batch(self, pil_images, scale_factor=None):
        """
        Process a batch of images using RVSR.
        Args:
            pil_images (list): List of PIL.Image objects to upscale
            scale_factor (int, optional): Scale factor (must be 4 for RVSR)
        Returns:
            list: List of upscaled PIL.Image objects
        """
        return [self.process_image(img, scale_factor) for img in pil_images]

    def process_tensor(self, image_tensor):
        """
        Process a tensor directly using RVSR without PIL conversion.
        Args:
            image_tensor (torch.Tensor): Input tensor of shape [B, C, H, W] in range [-1, 1]
        Returns:
            torch.Tensor: Upscaled tensor of shape [B, C, H*4, W*4] in range [-1, 1]
        """
        
        # Perform inference
        with torch.no_grad():
            t1 = time.time()
            output = self.model(image_tensor.to(device=self.device))
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            
        return output

def get_processor(device=None):
    """
    Factory function to create a RvsrUpscalerProcessor instance.
    Args:
        device (str, optional): Device to run the processor on ('cuda' or 'cpu').
    Returns:
        RvsrUpscalerProcessor: An instance of the RVSR upscaler processor
    """
    return RvsrUpscalerProcessor(device=device) 