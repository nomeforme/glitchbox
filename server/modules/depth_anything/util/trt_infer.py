import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from depth_anything.util.transform import load_image

def run(args):
    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    input_image, (orig_h, orig_w) = load_image(args.img)

    # Create logger and load the TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # Create execution context
    context = engine.create_execution_context()
    
    # Get input and output tensor names
    input_name = engine.get_tensor_name(0)  # First tensor is input
    output_name = engine.get_tensor_name(1)  # Second tensor is output
    
    # Get tensor shapes
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Allocate host and device memory
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Copy the input image to the pagelocked memory
    np.copyto(h_input, input_image.ravel())
    
    # Set input tensor
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Start timing
    start_time = time.time()
    
    # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
    # End timing
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    depth = h_output
    
    # Process the depth output
    depth = np.reshape(depth, output_shape[2:])
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = cv2.resize(depth, (orig_w, orig_h))
    
    # Save the depth map
    img_name = os.path.basename(args.img)
    if args.grayscale:
        cv2.imwrite(f'{args.outdir}/{img_name[:img_name.rfind(".")]}_depth.png', depth)
    else:
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f'{args.outdir}/{img_name[:img_name.rfind(".")]}_depth.png', colored_depth)
    
    # Log the inference time
    print(f"Inference time: {inference_time:.2f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    
    args = parser.parse_args()
    run(args)
