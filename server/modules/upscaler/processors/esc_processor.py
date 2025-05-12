import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from ..models.esc.basicsr.models import build_model
from ..models.esc.basicsr.utils.img_utils import img2tensor, tensor2img, imwrite
from ..models.esc.basicsr.utils.options import yaml_load
# from ..models.esc import esc
# from ..models.esc.esc import archs, models, data

class ESCProcessor:
    """
    A processor class for upscaling images using the ESC model.
    This class handles upscaling images using a deep learning model.
    """
    
    def __init__(self, device=None, model_path=None, config_path=None, scale_factor=4):
        """
        Initialize the ESC upscaler processor.
        
        Args:
            device (str, optional): Device to run the processor on ('cuda' or 'cpu').
                If None, will automatically select the best available device.
            model_path (str, optional): Path to the model weights file.
                If None, will use the default model path.
            config_path (str, optional): Path to the model configuration file.
                If None, will use the default config path.
            scale_factor (int, optional): Scale factor for upscaling (2 or 4).
                Defaults to 4.
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Set scale factor
        if scale_factor not in [2, 4]:
            print(f"Warning: ESC only supports 2x or 4x upscaling. Using 4x.")
            scale_factor = 4
        self.scale_factor = scale_factor
        
        # Set default paths if not provided
        base_checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'checkpoints/esc'
        )
        scale_folder = f"{scale_factor}x"
        
        if model_path is None:
            model_path = os.path.join(
                base_checkpoint_dir,
                scale_folder,
                f'ESC_light_X{scale_factor}.pth'
            )
        if config_path is None:
            config_path = os.path.join(
                base_checkpoint_dir,
                scale_folder,
                f'ESC_light_X{scale_factor}.yaml'
            )
            
        # Load configuration
        opt = yaml_load(config_path)
        
        # Modify Opt for Inference
        opt['is_train'] = False
        opt['dist'] = False
        opt['rank'] = 0
        opt['num_gpu'] = 1 if torch.cuda.is_available() else 0
        opt['gpu_ids'] = [0] if torch.cuda.is_available() else []
        
        # Build model
        self.model = build_model(opt)
        self.model.net_g = self.model.net_g.to(self.device)
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            load_net = torch.load(model_path, map_location=self.device)
            if 'params' in load_net:
                self.model.net_g.load_state_dict(load_net['params'], strict=True)
            elif 'state_dict' in load_net:
                self.model.net_g.load_state_dict(load_net['state_dict'], strict=True)
            else:
                self.model.net_g.load_state_dict(load_net, strict=True)
        
        # Convert to half precision
        self.model.net_g = self.model.net_g.half()
        
        # Attempt to compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile'):
            pt_version = torch.__version__
            major_minor = tuple(map(int, pt_version.split('.')[:2]))
            if major_minor >= (2, 0):
                try:
                    self.model.net_g = torch.compile(self.model.net_g)
                    print(f"Model compiled successfully with torch.compile (PyTorch {pt_version})")
                except Exception as e:
                    print(f"WARNING: torch.compile failed. Using uncompiled model. Error: {e}")
            else:
                print(f"INFO: torch.compile present but PyTorch version ({pt_version}) is < 2.0. Skipping compilation.")
        else:
            print("INFO: torch.compile not available (PyTorch < 2.0). Using uncompiled model.")
        
        # Set model to evaluation mode
        self.model.net_g.eval()

    def set_scale_factor(self, factor):
        """
        Set the scale factor for upscaling.
        
        Args:
            factor (int or float): Scale factor to use for upscaling
        """
        if factor > 0:
            self.scale_factor = factor
        else:
            print(f"Invalid scale factor: {factor}. Using default: 4")
            self.scale_factor = 4
            
    def set_resample_method(self, method):
        """
        Set the resampling method to use for upscaling.
        Note: ESC uses a neural network for upscaling, so this method
        is kept for API consistency but doesn't actually change the behavior.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        print(f"Note: ESC uses a neural network for upscaling. "
              f"Resampling method '{method}' is ignored.")
            
    def process_image(self, pil_image, scale_factor=None):
        """
        Upscale a PIL image using ESC.
        
        Args:
            pil_image (PIL.Image): Input image to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: ESC supports 2x or 4x upscaling.
                
        Returns:
            PIL.Image: The upscaled image
        """
        # Use provided parameters or defaults
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        # Validate scale factor
        if scale not in [2, 4]:
            print(f"Warning: ESC only supports 2x or 4x upscaling. Using {self.scale_factor}x.")
            scale = self.scale_factor
        
        # Convert PIL image to tensor
        img_np = np.array(pil_image)
        img_tensor = img2tensor(img_np, bgr2rgb=True, float32=True)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Upscale the image
        with torch.no_grad():
            t1 = time.time()
            self.model.lq = img_tensor
            self.model.test()
            output = self.model.get_current_visuals()['result']
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            
        # Convert output tensor back to PIL image
        output_img = tensor2img(output, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
        output = Image.fromarray(output_img)
        
        return output
    
    def process_batch(self, pil_images, scale_factor=None):
        """
        Upscale a batch of PIL images using ESC.
        
        Args:
            pil_images (list): List of PIL images to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: ESC supports 2x or 4x upscaling.
                
        Returns:
            list: List of upscaled PIL images
        """
        return [self.process_image(img, scale_factor) for img in pil_images]

    def process_tensor(self, image_tensor):
        """
        Process a tensor directly using ESC without PIL conversion.
        
        Args:
            image_tensor (torch.Tensor): Input tensor of shape [B, C, H, W] in range [0, 1]
            
        Returns:
            torch.Tensor: Upscaled tensor of shape [B, C, H*4, W*4] in range [0, 1]
        """
        # Ensure tensor is on the correct device and convert to half precision
        image_tensor = image_tensor.to(self.device).half()
        
        # Process through model
        with torch.no_grad():
            t1 = time.time()
            self.model.lq = image_tensor
            self.model.test()
            output = self.model.get_current_visuals()['result']
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1:.2f} seconds")
            
        return output

def get_processor(device=None, model_path=None, config_path=None, scale_factor=4):
    """
    Factory function to create an ESCProcessor instance.
    
    Args:
        device (str, optional): Device to run the processor on ('cuda' or 'cpu').
            If None, will automatically select the best available device.
        model_path (str, optional): Path to the model weights file.
            If None, will use the default model path.
        config_path (str, optional): Path to the model configuration file.
            If None, will use the default config path.
        scale_factor (int, optional): Scale factor for upscaling (2 or 4).
            Defaults to 4.
            
    Returns:
        ESCProcessor: An instance of the ESC upscaler processor
    """
    return ESCProcessor(device=device, model_path=model_path, config_path=config_path, scale_factor=scale_factor) 