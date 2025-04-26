import os
import argparse
from PIL import Image

from .factory import create_upscaler

def main():
    parser = argparse.ArgumentParser(description="Upscale an image using different methods")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    parser.add_argument("--upscaler", default="pil", choices=["pil", "fast_srgan"], 
                        help="Upscaler to use (default: pil)")
    parser.add_argument("--device", default=None, help="Device to use (cuda, mps, or cpu)")
    parser.add_argument("--scale", type=float, default=2.0, help="Scale factor (default: 2.0)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load input image
    input_image = Image.open(args.input)
    print(f"Input image size: {input_image.size}")
    
    # Create upscaler
    upscaler = create_upscaler(
        upscaler_type=args.upscaler,
        device=args.device
    )
    
    # Set scale factor
    upscaler.set_scale_factor(args.scale)
    
    # Process image
    output_image = upscaler.process_image(input_image)
    print(f"Output image size: {output_image.size}")
    
    # Save output image
    output_image.save(args.output)
    print(f"Saved upscaled image to {args.output}")

if __name__ == "__main__":
    main() 