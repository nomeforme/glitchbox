## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
##      Enhanced for Smooth Visualization    ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import signal
import sys

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)
found_rgb = False 
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create enhanced RealSense filters for smoother output (no decimation to maintain size)
spatial = rs.spatial_filter()         # Edge-preserving spatial smoothing
spatial.set_option(rs.option.filter_magnitude, 3)  # Reduced from 5
spatial.set_option(rs.option.filter_smooth_alpha, 0.6)  # Reduced from 1.0
spatial.set_option(rs.option.filter_smooth_delta, 30)   # Reduced from 50
spatial.set_option(rs.option.holes_fill, 2)  # Reduced from 3

temporal = rs.temporal_filter()       # Temporal smoothing
temporal.set_option(rs.option.filter_smooth_alpha, 0.3)  # Reduced from 0.4
temporal.set_option(rs.option.filter_smooth_delta, 15)   # Reduced from 20

hole_filling = rs.hole_filling_filter()  # Fill missing depth data
hole_filling.set_option(rs.option.holes_fill, 1)

# Lighter post-processing focused on edge smoothing
pass  # Parameters now handled inline

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames(100000000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Apply RealSense filters in sequence (removed decimation to maintain resolution)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Enhanced depth processing focused on smooth edges without distortion
        # Convert to 8-bit first
        depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_8bit = 255 - depth_8bit  # Invert the depth
        
        # Apply gentle bilateral filter for edge smoothing (multiple light passes)
        depth_8bit = cv2.bilateralFilter(depth_8bit, 7, 40, 40)
        depth_8bit = cv2.bilateralFilter(depth_8bit, 5, 30, 30)
        
        # Light Gaussian blur to smooth any remaining rough edges
        depth_8bit = cv2.GaussianBlur(depth_8bit, (3, 3), 0.5)

        # Apply colormap on depth image (INFERNO for thermal-like appearance)
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nReceived Ctrl+C, shutting down gracefully...")

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()