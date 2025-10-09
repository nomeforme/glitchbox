import pyrealsense2 as rs
import numpy as np
import cv2
import pyvirtualcam # Import the library
import os # To check if the device exists
import threading
import sys
import signal
from realsense_utils import create_pipeline_with_reset, hardware_reset_and_wait

# --- Configuration ---
WIDTH, HEIGHT, FPS = 640, 480, 30 # Define constants for clarity
TARGET_VIRTUAL_DEVICE = '/dev/video42' # Your configured v4l2loopback device

# --- Global variables for live parameter adjustment ---
distance_threshold = 2000  # Distance threshold in millimeters (2 meters default)
threshold_lock = threading.Lock()
running = True

def threshold_input_listener():
    """Thread function to listen for threshold input"""
    global distance_threshold, running
    
    print("\n=== LIVE THRESHOLD CONTROL ===")
    print("Type a new distance threshold in millimeters and press Enter")
    print("Examples: 1500, 2000, 3000")
    print("Type 'q' and press Enter to quit")
    print(f"Current threshold: {distance_threshold}mm")
    print("==============================\n")
    
    try:
        while running:
            try:
                user_input = input(f"Threshold ({distance_threshold}mm) > ").strip()
                
                if user_input.lower() == 'q':
                    print("Quitting...")
                    running = False
                    break
                elif user_input == '':
                    # Empty input, just show current threshold
                    continue
                else:
                    # Try to parse as number
                    new_threshold = int(user_input)
                    if new_threshold < 100:
                        print("Warning: Threshold too low, setting to minimum 100mm")
                        new_threshold = 100
                    elif new_threshold > 10000:
                        print("Warning: Threshold very high, setting to maximum 10000mm")
                        new_threshold = 10000
                    
                    with threshold_lock:
                        distance_threshold = new_threshold
                    print(f"Threshold updated to: {distance_threshold}mm")
                    
            except ValueError:
                print("Invalid input. Please enter a number in millimeters or 'q' to quit.")
            except EOFError:
                # Handle Ctrl+D
                print("\nQuitting...")
                running = False
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C in input thread
                print("\nQuitting...")
                running = False
                break
                
    except Exception as e:
        print(f"Input listener error: {e}")

def apply_distance_threshold(depth_image, threshold_mm):
    """Apply distance threshold for background removal"""
    # Create a copy to avoid modifying the original
    thresholded_depth = depth_image.copy()
    
    # Set all pixels beyond threshold to maximum depth value (0 in RealSense means no data/far)
    # We'll set them to a high value that will appear as background after processing
    max_depth_value = 8000  # 8 meters, which should be beyond most indoor scenes
    thresholded_depth[depth_image > threshold_mm] = max_depth_value
    
    return thresholded_depth

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

# Start threshold input listener thread
input_thread = threading.Thread(target=threshold_input_listener, daemon=True)
input_thread.start()

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
        print("Camera processing started. Use the terminal input to adjust threshold.")

        while running:
            frames = pipeline.wait_for_frames(10000)
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

            # Apply distance threshold for background removal
            with threshold_lock:
                current_threshold = distance_threshold
            depth_image = apply_distance_threshold(depth_image, current_threshold)

            # Enhanced depth processing
            depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
            depth_8bit = 255 - depth_8bit  # Invert the depth
            depth_8bit = cv2.bilateralFilter(depth_8bit, 7, 40, 40)
            depth_8bit = cv2.bilateralFilter(depth_8bit, 5, 30, 30)
            depth_8bit = cv2.GaussianBlur(depth_8bit, (3, 3), 0.5)

            # Apply colormap
            depth_colormap_bgr = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO) # BGR

            # --- Zoom and Crop (1.25x from top-right) ---
            zoom_factor = 1.25
            scaled_width = int(WIDTH * zoom_factor)
            scaled_height = int(HEIGHT * zoom_factor)
            
            scaled_frame = cv2.resize(depth_colormap_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
            
            # Crop window from top-right of the scaled frame
            x_start = scaled_width - WIDTH
            y_start = 0
            cropped_frame = scaled_frame[y_start:y_start+HEIGHT, x_start:x_start+WIDTH]

            # --- Prepare frame for virtual camera ---
            #frame_to_send_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

            frame_to_send_rgb = cropped_frame
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
    running = False  # Stop the input listener thread
    print("Stopping RealSense pipeline...")
    try:
        pipeline.stop()
        print("RealSense pipeline stopped.")
    except Exception as e:
        print(f"Warning: Error stopping pipeline: {e}")

    # Hardware reset for next run with timeout
    print("Cleaning up with hardware reset...")

    def timeout_handler(signum, frame):
        print("Warning: Hardware reset timed out, forcing exit...")
        os._exit(0)

    try:
        # Set a 3-second timeout for hardware reset
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3)

        hardware_reset_and_wait(ctx, verbose=False)

        # Cancel the alarm if reset completes
        signal.alarm(0)
        print("Hardware reset complete.")
    except Exception as e:
        signal.alarm(0)  # Cancel alarm on error
        print(f"Warning: Hardware reset failed (non-critical): {e}")

    print("Script finished.")
    os._exit(0)  # Force immediate exit without waiting for threads