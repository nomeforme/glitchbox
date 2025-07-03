import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from PIL import Image
from .util.transform import load_image
from typing import Optional

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
        
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        self.stream = cuda.Stream()
        
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        print(f"Depth Anything TensorRT model initialized with engine: {engine_path}")
    
    def get_depth(self, image, normalized_distance_threshold: Optional[float] = 0.225, absolute_min: Optional[float] = 0, absolute_max: Optional[float] = 18):
        """
        Get depth map from an image with advanced thresholding.
        
        Args:
            image (PIL.Image): Input image.
            normalized_distance_threshold (float, optional): Threshold from 0.0 (closest) to 1.0 (farthest)
                for background removal. Defaults to None.
            absolute_min (float, optional): The absolute minimum raw depth value for normalization.
                Providing this and absolute_max ensures consistent depth mapping across images.
                If None, the minimum is calculated from the current image. Defaults to None.
            absolute_max (float, optional): The absolute maximum raw depth value for normalization.
                If None, the maximum is calculated from the current image. Defaults to None.
            
        Returns:
            PIL.Image: Depth map as a PIL image.
        """
        # --- Image Preprocessing ---
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
            input_image, _ = load_image(image)
        else:
            input_image = image
            orig_h, orig_w = input_image.shape[:2]
        
        np.copyto(self.h_input, input_image.ravel())
        
        # --- TensorRT Inference ---
        start_time = time.time()
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        inference_time = (time.time() - start_time) * 1000
        
        # --- Post-processing ---
        try:
            depth = np.reshape(self.h_output, self.output_shape[1:]) # remove batch dim
        except ValueError as e:
            print(f"Error reshaping depth output: {e}")
            raise
        
        # --- Determine Normalization Range (The Core of the New Logic) ---
        if absolute_min is not None and absolute_max is not None:
            depth_min, depth_max = absolute_min, absolute_max
            print(f"Using absolute depth range for normalization. Min: {depth_min:.2f}, Max: {depth_max:.2f}")
            # Clip the depth values to the specified absolute range for safety
            depth = np.clip(depth, depth_min, depth_max)
        else:
            depth_min, depth_max = depth.min(), depth.max()
            print(f"Using per-image depth range for normalization. Min: {depth_min:.2f}, Max: {depth_max:.2f}")

        # --- Apply Normalized Distance Threshold ---
        if normalized_distance_threshold is not None and 0.0 < normalized_distance_threshold < 1.0:
            print(f"Applying normalized distance threshold: {normalized_distance_threshold}")
            if depth_max > depth_min:
                # Calculate the threshold in the absolute scale of the depth map
                cutoff_value = depth_min + (depth_max - depth_min) * normalized_distance_threshold
                # Set all values beyond the cutoff to the farthest value (depth_max)
                depth[depth < cutoff_value] = depth_min
        elif normalized_distance_threshold is not None:
             print(f"Warning: normalized_distance_threshold must be between 0.0 and 1.0. Got {normalized_distance_threshold}. Skipping background removal.")

        # --- Normalize and Convert to Image ---
        if depth_max - depth_min > 0:
            # Normalize using the determined min/max (either absolute or per-image)
            depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
            
        depth_normalized = depth_normalized.astype(np.uint8)
        
        depth_resized = cv2.resize(depth_normalized, (orig_w, orig_h))
        
        if self.grayscale:
            depth_pil = Image.fromarray(depth_resized)
        else:
            colored_depth = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)
            depth_pil = Image.fromarray(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
        
        print(f"Depth estimation inference time: {inference_time:.2f} ms")
        
        return depth_pil
    
    def __call__(self, image, normalized_distance_threshold: Optional[float] = None, absolute_min: Optional[float] = None, absolute_max: Optional[float] = None):
        """
        Callable interface for the model.
        
        Args:
            image (PIL.Image): Input image
            normalized_distance_threshold (float, optional): See get_depth() for details.
            absolute_min (float, optional): See get_depth() for details.
            absolute_max (float, optional): See get_depth() for details.
            
        Returns:
            dict: Dictionary containing the depth map
        """
        depth_map = self.get_depth(image, normalized_distance_threshold, absolute_min, absolute_max)
        return {"depth": depth_map}