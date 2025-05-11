import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from ..models.dscf_sr.team23_DSCF import DSCF
from ..models.dscf_sr.team00_EFDN import EFDN
# from ..models.dscf_sr.utils_image import uint2tensor4, tensor2uint

# convert uint (HxWxn_channels) to 4-dimensional torch tensor
def uint2tensor4(img, data_range):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255./data_range).unsqueeze(0)


# convert uint (HxWxn_channels) to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# convert torch tensor to uint
def tensor2uint(img, data_range):
    img = img.data.squeeze().float().clamp_(0, 1*data_range).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0/data_range).round())

class DscfEfdnUpscalerProcessor:
    """
    A processor class for upscaling images using the DSCF-SR or EFDN model.
    This class handles upscaling images using a deep learning model.
    """
    
    def __init__(self, device=None, model_type='efdn'):
        """
        Initialize the upscaler processor.
        Args:
            device (str, optional): Device to run the processor on ('cuda' or 'cpu').
            model_type (str): 'dscf' or 'efdn'.
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = "efdn" #model_type.lower()
        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        # Try to compile the model if torch.compile is available
        print(f"Checking for torch.compile support...")
        if hasattr(torch, 'compile'):
            print(f"torch.compile is available, attempting to compile model...")
            try:
                self.model = torch.compile(self.model)
                print(f"Successfully compiled {self.model_type} model")
            except Exception as e:
                print(f"Failed to compile model: {e}")
                print("Continuing with uncompiled model")
        else:
            print(f"torch.compile is not available, using uncompiled model")
        # Default scale factor
        self.scale_factor = 4  # Changed to 4 to match notebook
        # Data range for normalization
        self.data_range = 1.0  # Both models expect [0,1] input and output
        
    def _load_model(self):
        if self.model_type == 'dscf':
            model = DSCF(
                num_in_ch=3,
                num_out_ch=3,
                feature_channels=26,
                upscale=4,
                bias=True,
                img_range=1.0,  # Since we normalize to [0,1]
                rgb_mean=(0.485, 0.456, 0.406)
            )
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '../checkpoints/dscf_sr/team23_DSCF.pth'
            )
        elif self.model_type == 'efdn':
            model = EFDN(scale=4, in_channels=3, n_feats=48, out_channels=3)
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '../checkpoints/dscf_sr/team00_EFDN.pth'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'dscf' or 'efdn'.")
        checkpoint_path = os.path.abspath(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        return model
        
    def set_scale_factor(self, factor):
        if factor > 0:
            self.scale_factor = factor
        else:
            print(f"Invalid scale factor: {factor}. Using default: 4")
            self.scale_factor = 4
            
    def set_resample_method(self, method):
        """
        Set the resampling method to use for upscaling.
        Note: DSCF-SR uses a neural network for upscaling, so this method
        is kept for API consistency but doesn't actually change the behavior.
        
        Args:
            method (str): Resampling method to use. Options are:
                'nearest', 'bilinear', 'bicubic', 'lanczos'
        """
        # DSCF-SR uses a neural network for upscaling, so this method
        # doesn't actually change the behavior
        print(f"Note: DSCF-SR uses a neural network for upscaling. "
              f"Resampling method '{method}' is ignored.")
            
    def process_image(self, pil_image, scale_factor=None):
        """
        Process a single PIL image through the model.
        
        Args:
            pil_image (PIL.Image): Input image to process
            scale_factor (int, optional): Scale factor to use. If None, uses default.
            
        Returns:
            PIL.Image: Processed image
        """
        
        # Convert PIL image to numpy array (HWC, uint8, RGB)
        img_numpy_uint8 = np.array(pil_image)
        
        # Convert to tensor using utils_image function
        img_tensor = uint2tensor4(img_numpy_uint8, data_range=self.data_range)
        img_tensor = img_tensor.to(self.device)
        
        # Process through model
        with torch.no_grad():
            t1 = time.time()
            output = self.model(img_tensor)
            t2 = time.time()
            print(f"Super Resolution - Time taken: {t2 - t1} seconds")
            
        # Convert back to PIL image using utils_image function
        output_numpy = tensor2uint(output, data_range=self.data_range)
        output_pil = Image.fromarray(output_numpy)
        print(f"Super Resolution - Output shape: {output_pil.size}")
        
        return output_pil
    
    def process_batch(self, pil_images, scale_factor=None):
        """
        Process a batch of PIL images through the model.
        
        Args:
            pil_images (list): List of PIL.Image objects to process
            scale_factor (int, optional): Scale factor to use. If None, uses default.
            
        Returns:
            list: List of processed PIL.Image objects
        """
        return [self.process_image(img, scale_factor) for img in pil_images]


def get_processor(device=None, model_type='dscf'):
    """
    Factory function to create a DscfEfdnUpscalerProcessor instance.
    Args:
        device (str, optional): Device to run the processor on ('cuda' or 'cpu').
        model_type (str): 'dscf' or 'efdn'.
    Returns:
        DscfEfdnUpscalerProcessor: An instance of the upscaler processor
    """
    return DscfEfdnUpscalerProcessor(device=device, model_type=model_type) 