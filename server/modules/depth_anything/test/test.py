import os
import argparse
from server.modules.depth_anything.util.trt_infer import run
import time

def batch_inference(args):

    print(args)
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Get the base image name without extension
    base_img_name = os.path.splitext(os.path.basename(args.img))[0]
    
    # Run inference for the specified number of iterations
    for i in range(args.num_iterations):
        # Create a copy of args to modify for each iteration
        current_args = argparse.Namespace(**vars(args))
        
        # Run inference
        print(f"Processing iteration {i+1}/{args.num_iterations}")
        start_time = time.time()
        run(current_args)
        end_time = time.time()
        print(f"Iteration {i+1} completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, default='assets/mountain.png', help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory for the depth maps')
    parser.add_argument('--engine', type=str, default='modules/depth_anything/models/depth_anything_v2_vits.trt', help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations to run')
    
    args = parser.parse_args()
    batch_inference(args) 