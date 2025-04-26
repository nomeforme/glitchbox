import os
import argparse
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_upscaler import upscale_image
from pil_upscaler import upscale_image_pil, Image

def load_image(image_path):
    """Load an image using PIL."""
    if image_path.startswith(('http://', 'https://')):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
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
    
    return img

def compare_upscalers(image_path, output_dir='.', scale_factor=2, device='cpu'):
    """
    Compare PyTorch and PIL upscalers in terms of speed and quality.
    
    Args:
        image_path: Path to the input image or URL
        output_dir: Directory to save output images
        scale_factor: Upscaling factor (default: 2)
        device: Device to run PyTorch model on ('cpu' or 'cuda')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    if image_path.startswith(('http://', 'https://')):
        base_name = 'downloaded_image'
    else:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load the original image
    original_img = load_image(image_path)
    
    # Save original image
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    original_img.save(original_path)
    
    # Upscale with PIL Lanczos (best quality)
    pil_path = os.path.join(output_dir, f"{base_name}_pil_lanczos_{scale_factor}x.png")
    print("\nUpscaling with PIL Lanczos...")
    pil_start = time.time()
    pil_img = upscale_image_pil(image_path, pil_path, scale_factor=scale_factor, resample=Image.LANCZOS)
    pil_time = time.time() - pil_start
    print(f"PIL Lanczos upscaling took {pil_time:.4f} seconds")
    
    # Upscale with PyTorch
    pytorch_path = os.path.join(output_dir, f"{base_name}_pytorch_{scale_factor}x.png")
    print("\nUpscaling with PyTorch...")
    pytorch_start = time.time()
    upscale_image(image_path, pytorch_path, scale_factor=scale_factor, device=device)
    pytorch_time = time.time() - pytorch_start
    print(f"PyTorch upscaling took {pytorch_time:.4f} seconds")
    
    # Calculate speedup
    speedup = pil_time / pytorch_time if pytorch_time > 0 else float('inf')
    print(f"\nPyTorch is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PIL Lanczos")
    
    # Create a comparison image
    try:
        # Load the upscaled images
        pil_upscaled = Image.open(pil_path)
        pytorch_upscaled = Image.open(pytorch_path)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(np.array(original_img))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # PIL upscaled
        axes[0, 1].imshow(np.array(pil_upscaled))
        axes[0, 1].set_title(f'PIL Lanczos ({pil_time:.4f}s)')
        axes[0, 1].axis('off')
        
        # PyTorch upscaled
        axes[1, 0].imshow(np.array(pytorch_upscaled))
        axes[1, 0].set_title(f'PyTorch ({pytorch_time:.4f}s)')
        axes[1, 0].axis('off')
        
        # Difference (exaggerated for visibility)
        diff = np.array(pil_upscaled).astype(np.float32) - np.array(pytorch_upscaled).astype(np.float32)
        diff = np.abs(diff) * 5  # Exaggerate differences
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title('Difference (x5)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison_{scale_factor}x.png")
        plt.savefig(comparison_path)
        print(f"\nComparison image saved to {comparison_path}")
        
    except Exception as e:
        print(f"Error creating comparison image: {e}")
    
    print("\nComparison complete. Output files:")
    print(f"Original: {original_path}")
    print(f"PIL Lanczos: {pil_path}")
    print(f"PyTorch: {pytorch_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch and PIL upscalers')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input image or URL')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output images (default: current directory)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Upscaling factor (default: 2)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run PyTorch model on (default: cpu)')
    
    args = parser.parse_args()
    
    compare_upscalers(args.input, args.output_dir, args.scale, args.device)

if __name__ == "__main__":
    main() 