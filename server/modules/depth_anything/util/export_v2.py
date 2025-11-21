import sys
import os

# Add parent directories to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import cv2
import glob
import numpy as np
import torch
import torch.onnx

from modules.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load_from', type=str, help='Path to the checkpoint file')

    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model. Please run on a machine with GPU support.")
    
    # we are undergoing company review procedures to release Depth-Anything-Giant checkpoint
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    
    # Use the provided checkpoint path if specified, otherwise use the default path
    if args.load_from:
        checkpoint_path = args.load_from
    else:
        checkpoint_path = os.path.join(ROOT_DIR, 'modules', 'depth_anything', 'depth_anything_v2', 'checkpoints', f'depth_anything_{args.encoder}14.pth')
    
    # Load model and convert to float16
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
    depth_anything = depth_anything.to('cuda').half().eval()

    # Define dummy input data in float16
    dummy_input = torch.ones((3, args.input_size, args.input_size), dtype=torch.float16).unsqueeze(0).cuda()

    # Provide an example input to the model, this is necessary for exporting to ONNX
    with torch.no_grad():
        example_output = depth_anything.forward(dummy_input)

    onnx_path = f'depth_anything_v2_{args.encoder}.onnx'

    # Export the PyTorch model to ONNX format
    torch.onnx.export(
        depth_anything,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    main()