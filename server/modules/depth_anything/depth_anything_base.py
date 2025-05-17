import os
import cv2
import torch
import numpy as np
from PIL import Image
from .util.transform import load_image
from .depth_anything_v2.dpt import DepthAnythingV2

class DepthAnything:
    """
    Base implementation of the Depth Anything model using PyTorch.
    This class provides a get_depth() method that can be used to generate depth maps
    from input images without TensorRT optimization.
    """
    
    def __init__(self, encoder='vits', device="cuda", grayscale=False, compile_model=True):
        """
        Initialize the Depth Anything model.
        
        Args:
            encoder (str): Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            device (str): Device to run inference on (cuda or cpu)
            grayscale (bool): Whether to return grayscale depth maps
            compile_model (bool): Whether to use torch.compile for optimization (requires PyTorch 2.0+)
        """
        self.device = device
        self.grayscale = grayscale
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        if encoder not in model_configs:
            raise ValueError(f"Encoder must be one of {list(model_configs.keys())}")
        
        # Initialize model
        self.model = DepthAnythingV2(**model_configs[encoder])
        
        # Load checkpoint
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoint_path = os.path.join(root_dir, 'modules', 'depth_anything', 'depth_anything_v2', 
                                     'checkpoints', f'depth_anything_{encoder}14.pth')
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(device).eval()
        
        # Apply torch.compile if available and requested
        if compile_model:
            try:
                if hasattr(torch, 'compile'):
                    print("Using torch.compile for model optimization...")
                    self.model = torch.compile(
                        self.model,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    print("Model compilation successful")
                else:
                    print("torch.compile not available - requires PyTorch 2.0+. Using standard model.")
            except Exception as e:
                print(f"Model compilation failed: {str(e)}. Using standard model.")
        
        print(f"Depth Anything model initialized with encoder: {encoder}")
    
    @torch.no_grad()
    def get_depth(self, image):
        """
        Get depth map from an image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Depth map as a PIL image
        """
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
            image = np.array(image)
        else:
            image = np.array(image)
            orig_h, orig_w = image.shape[:2]
        
        # Use the model's infer_image method which handles preprocessing
        depth = self.model.infer_image(image)
        
        # Normalize depth values
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Convert to PIL Image
        if self.grayscale:
            depth_pil = Image.fromarray(depth)
        else:
            colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            depth_pil = Image.fromarray(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
        
        return depth_pil
    
    def __call__(self, image):
        """
        Callable interface for the model.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            dict: Dictionary containing the depth map
        """
        depth_map = self.get_depth(image)
        return {"depth": depth_map} 