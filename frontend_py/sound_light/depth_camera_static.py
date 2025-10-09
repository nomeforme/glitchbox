import pyrealsense2 as rs
import numpy as np
import cv2
import pyvirtualcam # Import the library
import os # To check if the device exists
from realsense_utils import create_pipeline_with_reset, hardware_reset_and_wait

# --- RealSense Configuration ---
WIDTH, HEIGHT, FPS = 640, 480, 30 # Define constants for clarity
TARGET_VIRTUAL_DEVICE = '/dev/video42' # Your configured v4l2loopback device

# Background removal parameter (0.0 = no removal, 1.0 = maximum removal)
BACKGROUND_REMOVAL_THRESHOLD = 0.5  # Adjust this value between 0.0 and 1.0

# --- Depth range configuration ---
# These values define the min/max depth range in millimeters for background removal
MIN_DEPTH_MM = 300   # Minimum depth to consider (closer objects)
MAX_DEPTH_MM = 3000  # Maximum depth to consider (further objects)

# Create pipeline with hardware reset for clean state
pipeline, ctx = create_pipeline_with_reset()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(f"RealSense Device Product Line: {device_product_line}")

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
# config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS) # Color stream not strictly needed if only sending depth

# Start streaming
pipeline.start(config)
print("RealSense pipeline started.")

# Create enhanced RealSense filters
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 3)
spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial.set_option(rs.option.filter_smooth_delta, 30)
spatial.set_option(rs.option.holes_fill, 2)

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
temporal.set_option(rs.option.filter_smooth_delta, 15)

hole_filling = rs.hole_filling_filter()
hole_filling.set_option(rs.option.holes_fill, 1)
print("RealSense filters configured.")

# --- pyvirtualcam Setup ---
# Check if the target virtual device exists
if not os.path.exists(TARGET_VIRTUAL_DEVICE):
    print(f"ERROR: Virtual camera device {TARGET_VIRTUAL_DEVICE} not found!")
    print("Please ensure v4l2loopback is loaded correctly with this video_nr.")
    pipeline.stop()
    exit(1)

print(f"Attempting to use virtual camera: {TARGET_VIRTUAL_DEVICE}")

try:
    # We will send the depth_colormap to the virtual camera.
    # pyvirtualcam expects RGB frames. OpenCV's applyColorMap produces BGR.
    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS,
                             fmt=pyvirtualcam.PixelFormat.RGB,
                             device=TARGET_VIRTUAL_DEVICE) as cam:
        print(f'Virtual camera started successfully. Outputting to {cam.device} at {WIDTH}x{HEIGHT} @ {FPS}fps')
        print("Press Ctrl+C to stop.")

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame() # Not needed if only sending depth

            if not depth_frame:
                continue

            # Apply RealSense filters
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # Convert depth image to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply background removal based on threshold parameter
            if BACKGROUND_REMOVAL_THRESHOLD > 0.0:
                # Calculate the actual depth threshold in millimeters
                # 0.0 = MIN_DEPTH_MM (no removal), 1.0 = MAX_DEPTH_MM (maximum removal)
                depth_threshold_mm = MIN_DEPTH_MM + (BACKGROUND_REMOVAL_THRESHOLD * (MAX_DEPTH_MM - MIN_DEPTH_MM))
                
                # Create mask: pixels beyond threshold become 0 (background)
                # Also mask out pixels that are 0 (no depth data) or too close
                mask = (depth_image > 0) & (depth_image <= depth_threshold_mm)
                depth_image = depth_image * mask.astype(np.uint16)

            # Enhanced depth processing
            depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_8bit = 255 - depth_8bit  # Invert the depth
            depth_8bit = cv2.bilateralFilter(depth_8bit, 7, 40, 40)
            depth_8bit = cv2.bilateralFilter(depth_8bit, 5, 30, 30)
            depth_8bit = cv2.GaussianBlur(depth_8bit, (3, 3), 0.5)

            # Apply colormap
            depth_colormap_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO) # BGR

            # --- Prepare frame for virtual camera ---
            #frame_to_send_rgb = cv2.cvtColor(depth_colormap_bgr, cv2.COLOR_BGR2RGB)

            frame_to_send_rgb = depth_colormap_bgr
            # Send to virtual camera
            cam.send(frame_to_send_rgb)

            # Wait until it's time for the next frame
            cam.sleep_until_next_frame()

            # No local cv2.imshow loop

except KeyboardInterrupt:
    print("\nStreaming stopped by user (Ctrl+C).")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Stopping RealSense pipeline...")
    pipeline.stop()
    print("RealSense pipeline stopped.")

    # Hardware reset for next run
    print("Cleaning up with hardware reset...")
    hardware_reset_and_wait(ctx, verbose=False)
    print("Script finished.")