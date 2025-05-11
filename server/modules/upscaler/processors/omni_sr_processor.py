import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from ..models.omni_sr.OmniSR import OmniSR
from ..models.omni_sr.utilities import tensor2img

class OmniSRProcessor:
    """
    A processor class for upscaling images using the Omni-SR model.
    This class handles upscaling images using a deep learning model.
    """
    
    def __init__(self, device=None, model_path=None, config_path=None):
        """
        Initialize the Omni-SR upscaler processor.
        
        Args:
            device (str, optional): Device to run the processor on ('cuda' or 'cpu').
                If None, will automatically select the best available device.
            model_path (str, optional): Path to the model weights file.
                If None, will use the default model path.
            config_path (str, optional): Path to the model configuration file.
                If None, will use the default config path.
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'checkpoints/omni_sr/epoch994_OmniSR.pth'
            )
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'checkpoints/omni_sr/model_config.json'
            )
            
        # Load configuration
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Initialize model with correct parameters
        model_kwargs = {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": config.get('feature_num', 64),
            **config.get('module_params', {})
        }
        
        self.model = OmniSR(**model_kwargs)
        
        # Load model weights
        load_net = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in load_net:
            state_dict = load_net['state_dict']
        elif 'params' in load_net:
            state_dict = load_net['params']
        elif 'model_state_dict' in load_net:
            state_dict = load_net['model_state_dict']
        else:
            state_dict = load_net
            
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            key_no_module = k[7:] if k.startswith('module.') else k
            if key_no_module in model_state_dict:
                if model_state_dict[key_no_module].shape == v.shape:
                    filtered_state_dict[key_no_module] = v
                    
        self.model.load_state_dict(filtered_state_dict, strict=True)
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        # Attempt to compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile'):
            pt_version = torch.__version__
            major_minor = tuple(map(int, pt_version.split('.')[:2]))
            if major_minor >= (2, 0):
                try:
                    # Using default options for broader compatibility
                    self.model = torch.compile(self.model)
                    print(f"Model compiled successfully with torch.compile (PyTorch {pt_version})")
                except Exception as e:
                    print(f"WARNING: torch.compile failed. Using uncompiled model. Error: {e}")
            else:
                print(f"INFO: torch.compile present but PyTorch version ({pt_version}) is < 2.0. Skipping compilation.")
        else:
            print("INFO: torch.compile not available (PyTorch < 2.0). Using uncompiled model.")
        
        # Default scale factor (Omni-SR is designed for 4x upscaling)
        self.scale_factor = model_kwargs.get('upsampling', 4)
        
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
        Note: Omni-SR uses a neural network for upscaling, so this method
        is kept for API consistency but doesn't actually change the behavior.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        # Omni-SR uses a neural network for upscaling, so this method
        # doesn't actually change the behavior
        print(f"Note: Omni-SR uses a neural network for upscaling. "
              f"Resampling method '{method}' is ignored.")
            
    def process_image(self, pil_image, scale_factor=None):
        """
        Upscale a PIL image using Omni-SR.
        
        Args:
            pil_image (PIL.Image): Input image to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: Omni-SR is designed for 4x upscaling.
                
        Returns:
            PIL.Image: The upscaled image
        """
        # Use provided parameters or defaults
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        # Omni-SR is designed for 4x upscaling
        if scale != 4:
            print(f"Warning: Omni-SR is designed for 4x upscaling. "
                  f"Requested factor {scale} will be treated as 4x.")
            scale = 4
        
        # Convert PIL image to tensor
        img_np = np.array(pil_image)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
        img_tensor = (img_tensor - 0.5) * 2.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Upscale the image
        with torch.no_grad():
            t1 = time.time()
            output = self.model(img_tensor)
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            
        # Convert output tensor back to PIL image
        img_np_rgb_float = tensor2img(output.cpu())[0]
        img_np_rgb_uint8 = np.clip(img_np_rgb_float, 0, 255).astype(np.uint8)
        output = Image.fromarray(img_np_rgb_uint8)
        
        return output
    
    def process_batch(self, pil_images, scale_factor=None):
        """
        Upscale a batch of PIL images using Omni-SR.
        
        Args:
            pil_images (list): List of PIL images to upscale
            scale_factor (int or float, optional): Scale factor to use for upscaling.
                If None, uses the default set in the processor.
                Note: Omni-SR is designed for 4x upscaling.
                
        Returns:
            list: List of upscaled PIL images
        """
        return [self.process_image(img, scale_factor) for img in pil_images]


def get_processor(device=None, model_path=None, config_path=None):
    """
    Factory function to create an OmniSRProcessor instance.
    
    Args:
        device (str, optional): Device to run the processor on ('cuda' or 'cpu').
            If None, will automatically select the best available device.
        model_path (str, optional): Path to the model weights file.
            If None, will use the default model path.
        config_path (str, optional): Path to the model configuration file.
            If None, will use the default config path.
            
    Returns:
        OmniSRProcessor: An instance of the Omni-SR upscaler processor
    """
    return OmniSRProcessor(device=device, model_path=model_path, config_path=config_path) 