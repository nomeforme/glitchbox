import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
import argparse

class SimpleUpscaler(nn.Module):
    def __init__(self, scale_factor=2):
        super(SimpleUpscaler, self).__init__()
        self.scale_factor = scale_factor
        
        # Simple convolutional layers for upscaling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

def load_image(image_path):
    """Load an image and convert it to a PyTorch tensor."""
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image onto the background using alpha channel as mask
        background.paste(img, mask=img.split()[3])
        img = background
    
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = np.array(img)
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def save_image(tensor, output_path):
    """Save a PyTorch tensor as an image."""
    # Move tensor to CPU if it's on CUDA
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()  # CHW to HWC
    img = np.clip(img, 0, 1)  # Clip to [0, 1]
    img = (img * 255).astype(np.uint8)  # Scale to [0, 255]
    img = Image.fromarray(img)
    img.save(output_path)

def upscale_image(image_path, output_path=None, scale_factor=2, device='cpu'):
    """
    Upscale an image using a simple PyTorch model.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image (if None, returns the tensor)
        scale_factor: Upscaling factor (default: 2)
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        The upscaled image tensor if output_path is None, otherwise None
    """
    # Load the image
    img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Create and load the model
    model = SimpleUpscaler(scale_factor=scale_factor).to(device)
    
    # Start timing
    start_time = time.time()
    
    # Upscale the image
    with torch.no_grad():
        upscaled_tensor = model(img_tensor)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Upscaling took {processing_time:.2f} seconds")
    
    # Save or return the result
    if output_path:
        save_image(upscaled_tensor, output_path)
        return None
    else:
        return upscaled_tensor

def main():
    parser = argparse.ArgumentParser(description='Upscale an image using PyTorch')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output image')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (default: 2)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run the model on (default: cpu)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    upscale_image(args.input, args.output, args.scale, args.device)

if __name__ == "__main__":
    main() 