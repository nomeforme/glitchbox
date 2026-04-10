import os
import cv2
import numpy as np
# CRITICAL: Do NOT use pycuda.autoinit - it creates a separate CUDA context
# that conflicts with PyTorch's context. We'll use PyTorch's context instead.
# import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from PIL import Image
from .util.transform import load_image
from typing import Optional
import torch

class DepthAnythingTRT:
    """
    Optimized TensorRT implementation of the Depth Anything model.
    This class provides a get_depth() method that can be used to set params.control_image
    for use in the pipeline within generate() in main.py.
    """
    
    def __init__(self, engine_path, device="cuda", grayscale=False, normalized_distance_threshold=0.225, absolute_min=0.0, absolute_max=18.0):
        """
        Initialize the Depth Anything TensorRT model.

        Args:
            engine_path (str): Path to the TensorRT engine file
            device (str): Device to run inference on (cuda or cpu)
            grayscale (bool): Whether to return grayscale depth maps
            normalized_distance_threshold (float): Default threshold from 0.0 (closest) to 1.0 (farthest)
                for background removal. Defaults to 0.225.
            absolute_min (float): Default absolute minimum raw depth value for normalization.
                Defaults to 0.0.
            absolute_max (float): Default absolute maximum raw depth value for normalization.
                Defaults to 18.0.
        """
        self.engine_path = engine_path
        self.device = device
        self.grayscale = grayscale
        self.normalized_distance_threshold = normalized_distance_threshold
        self.absolute_min = absolute_min
        self.absolute_max = absolute_max

        # CRITICAL: Initialize PyCUDA to use PyTorch's CUDA context
        # This prevents conflicts with TensorRT engines in the main pipeline
        cuda.init()
        # Get PyTorch's CUDA context and make PyCUDA use it
        torch_device = torch.cuda.current_device()
        self.cuda_context = cuda.Device(torch_device).retain_primary_context()
        self.cuda_context.push()

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
    
    def get_depth(self, image, normalized_distance_threshold: Optional[float] = None, absolute_min: Optional[float] = None, absolute_max: Optional[float] = None):
        """
        Get depth map from an image with advanced thresholding.
        
        Args:
            image (PIL.Image): Input image.
            normalized_distance_threshold (float, optional): Threshold from 0.0 (closest) to 1.0 (farthest)
                for background removal. If None, uses the instance default.
            absolute_min (float, optional): The absolute minimum raw depth value for normalization.
                Providing this and absolute_max ensures consistent depth mapping across images.
                If None, uses the instance default.
            absolute_max (float, optional): The absolute maximum raw depth value for normalization.
                If None, uses the instance default.
            
        Returns:
            PIL.Image: Depth map as a PIL image.
        """
        # Use instance defaults if parameters are None
        if normalized_distance_threshold is None:
            normalized_distance_threshold = self.normalized_distance_threshold
        if absolute_min is None:
            absolute_min = self.absolute_min
        if absolute_max is None:
            absolute_max = self.absolute_max
        
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
            # print(f"Using absolute depth range for normalization. Min: {depth_min:.2f}, Max: {depth_max:.2f}")
            # Clip the depth values to the specified absolute range for safety
            depth = np.clip(depth, depth_min, depth_max)
        else:
            depth_min, depth_max = depth.min(), depth.max()
            print(f"Using per-image depth range for normalization. Min: {depth_min:.2f}, Max: {depth_max:.2f}")

        # --- Apply Normalized Distance Threshold ---
        if normalized_distance_threshold is not None and 0.0 < normalized_distance_threshold < 1.0:
            # print(f"Applying normalized distance threshold: {normalized_distance_threshold}")
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
        
        # print(f"Depth estimation inference time: {inference_time:.2f} ms")
        
        return depth_pil
    
    def mask_image_with_depth(self, image, depth_map, threshold=10, invert=False):
        """
        Mask an image using a depth map, removing areas based on depth values.

        Args:
            image (PIL.Image): Input RGB image to mask
            depth_map (PIL.Image): Depth map (grayscale or colored)
            threshold (int): Pixel values below this threshold will be masked out (0-255).
                Default is 10, which removes near-black areas (far/invalid depth).
            invert (bool): If True, inverts the mask (removes high-depth areas instead).
                Default is False.

        Returns:
            PIL.Image: Masked image with transparency (RGBA)
        """
        # Convert depth map to grayscale numpy array if needed
        depth_array = np.array(depth_map.convert('L'))

        # Create binary mask: 255 where depth > threshold, 0 elsewhere
        if invert:
            mask = (depth_array <= threshold).astype(np.uint8) * 255
        else:
            mask = (depth_array > threshold).astype(np.uint8) * 255

        # Convert input image to RGBA
        image_rgba = image.convert('RGBA')
        image_array = np.array(image_rgba)

        # Apply mask to alpha channel
        image_array[:, :, 3] = mask

        # Convert back to PIL Image
        masked_image = Image.fromarray(image_array, 'RGBA')

        return masked_image

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