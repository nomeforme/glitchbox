import os
import argparse
import requests
from PIL import Image
from io import BytesIO
import torch
from pytorch_upscaler import upscale_image

def main():
    # Default URL from upscale.py
    default_url = 'https://upload.wikimedia.org/wikipedia/commons/2/25/Blisk-logo-512-512-background-transparent.png?20160517154140'
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    default_device = 'cuda' if cuda_available else 'cpu'
    
    parser = argparse.ArgumentParser(description='Example of using the PyTorch upscaler')
    parser.add_argument('--input', type=str, default=default_url, 
                        help=f'Path to input image or URL (default: {default_url})')
    parser.add_argument('--output', type=str, default='scaled_2x.png', 
                        help='Path to output image (default: scaled_2x.png)')
    parser.add_argument('--scale', type=int, default=2, 
                        help='Upscaling factor (default: 2)')
    parser.add_argument('--device', type=str, default=default_device, choices=['cuda', 'cpu'], 
                        help=f'Device to run the model on (default: {default_device})')
    
    args = parser.parse_args()
    
    # Check if input is a URL or a local file
    if args.input.startswith(('http://', 'https://')):
        print(f"Downloading image from {args.input}...")
        try:
            response = requests.get(args.input)
            img = Image.open(BytesIO(response.content))
            # Save temporarily
            temp_path = 'temp_input.png'
            img.save(temp_path)
            input_path = temp_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            return
    else:
        # Check if local file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist.")
            return
        input_path = args.input
    
    print(f"Upscaling image by a factor of {args.scale}x using {args.device}...")
    upscale_image(input_path, args.output, args.scale, args.device)
    print(f"Upscaled image saved to {args.output}")
    
    # Clean up temporary file if it was downloaded
    if args.input.startswith(('http://', 'https://')) and os.path.exists('temp_input.png'):
        os.remove('temp_input.png')

if __name__ == "__main__":
    main() 