import os
import argparse
import requests
from PIL import Image
from io import BytesIO
import time
import numpy as np

def load_image(image_path):
    """Load an image using PIL."""
    if image_path.startswith(('http://', 'https://')):
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

def upscale_image_pil(image_path, output_path=None, scale_factor=2, resample=Image.LANCZOS):
    """
    Upscale an image using PIL's best resizing method.
    
    Args:
        image_path: Path to the input image or URL
        output_path: Path to save the output image (if None, returns the PIL Image)
        scale_factor: Upscaling factor (default: 2)
        resample: Resampling filter to use (default: Image.LANCZOS)
        
    Returns:
        The upscaled PIL Image if output_path is None, otherwise None
    """
    # Load the image
    img = load_image(image_path)
    
    # Calculate new dimensions
    new_width = img.width * scale_factor
    new_height = img.height * scale_factor
    
    # Start timing
    start_time = time.time()
    
    # Upscale the image
    upscaled_img = img.resize((new_width, new_height), resample=resample)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"PIL upscaling took {processing_time:.2f} seconds")
    
    # Save or return the result
    if output_path:
        upscaled_img.save(output_path)
        return None
    else:
        return upscaled_img

def compare_methods(image_path, output_dir='.'):
    """
    Compare different upscaling methods.
    
    Args:
        image_path: Path to the input image or URL
        output_dir: Directory to save output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    if image_path.startswith(('http://', 'https://')):
        base_name = 'downloaded_image'
    else:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Upscale with PIL using different methods
    methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    results = {}
    
    for method_name, resample in methods.items():
        output_path = os.path.join(output_dir, f"{base_name}_pil_{method_name}_2x.png")
        print(f"\nUpscaling with PIL {method_name}...")
        upscale_image_pil(image_path, output_path, scale_factor=2, resample=resample)
        results[f"PIL_{method_name}"] = output_path
    
    # Try to import and use PyTorch upscaler if available
    try:
        from pytorch_upscaler import upscale_image
        output_path = os.path.join(output_dir, f"{base_name}_pytorch_2x.png")
        print("\nUpscaling with PyTorch...")
        upscale_image(image_path, output_path, scale_factor=2, device='cpu')
        results["PyTorch"] = output_path
    except ImportError:
        print("\nPyTorch upscaler not available. Skipping comparison.")
    
    print("\nComparison complete. Output files:")
    for method, path in results.items():
        print(f"{method}: {path}")

def main():
    # Default URL from upscale.py
    default_url = 'https://upload.wikimedia.org/wikipedia/commons/2/25/Blisk-logo-512-512-background-transparent.png?20160517154140'
    
    parser = argparse.ArgumentParser(description='Upscale an image using PIL')
    parser.add_argument('--input', type=str, default=default_url, 
                        help=f'Path to input image or URL (default: {default_url})')
    parser.add_argument('--output', type=str, default='scaled_2x_pil.png', 
                        help='Path to output image (default: scaled_2x_pil.png)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Upscaling factor (default: 2)')
    parser.add_argument('--method', type=str, default='lanczos', 
                        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
                        help='Resampling method (default: lanczos)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different upscaling methods')
    
    args = parser.parse_args()
    
    # Map method name to PIL resample constant
    method_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    if args.compare:
        compare_methods(args.input)
    else:
        resample = method_map[args.method]
        upscale_image_pil(args.input, args.output, args.scale, resample)

if __name__ == "__main__":
    main() 