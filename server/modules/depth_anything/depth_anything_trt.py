import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from PIL import Image
from .util.transform import load_image

class DepthAnythingTRT:
    """
    Optimized TensorRT implementation of the Depth Anything model.
    This class provides a get_depth() method that can be used to set params.control_image
    for use in the pipeline within generate() in main.py.
    """
    
    def __init__(self, engine_path, device="cuda", grayscale=False):
        """
        Initialize the Depth Anything TensorRT model.
        
        Args:
            engine_path (str): Path to the TensorRT engine file
            device (str): Device to run inference on (cuda or cpu)
            grayscale (bool): Whether to return grayscale depth maps
        """
        self.engine_path = engine_path
        self.device = device
        self.grayscale = grayscale
        
        # Create logger and load the TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get input and output tensor names
        self.input_name = self.engine.get_tensor_name(0)  # First tensor is input
        self.output_name = self.engine.get_tensor_name(1)  # Second tensor is output
        
        # Get tensor shapes
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        # Allocate host and device memory
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        # Set input tensor
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        print(f"Depth Anything TensorRT model initialized with engine: {engine_path}")
    
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
            # Get original dimensions
            orig_w, orig_h = image.size
            
            # Convert to numpy array using the load_image function
            input_image, _ = load_image(image)
        else:
            # Assume image is already a numpy array
            input_image = image
            orig_h, orig_w = input_image.shape[:2]
        
        # Copy the input image to the pagelocked memory
        np.copyto(self.h_input, input_image.ravel())
        
        # Start timing
        start_time = time.time()
        
        # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        # End timing
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        depth = self.h_output
        
        # Process the depth output
        depth = np.reshape(depth, self.output_shape[2:])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (orig_w, orig_h))
        
        # Convert to PIL Image
        if self.grayscale:
            depth_pil = Image.fromarray(depth)
        else:
            colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            depth_pil = Image.fromarray(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
        
        # Log the inference time
        print(f"Depth estimation inference time: {inference_time:.2f} ms")
        
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