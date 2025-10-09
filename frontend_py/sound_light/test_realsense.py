import pyrealsense2 as rs
import time
from realsense_utils import create_pipeline_with_reset, hardware_reset_and_wait

try:
    # Create pipeline with hardware reset for clean state
    pipeline, ctx = create_pipeline_with_reset()
    config = rs.config()

    print("\nAttempting to start RealSense pipeline...")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    print("RealSense pipeline started.")

    # Give camera time to warm up
    time.sleep(0.5)

    print("Getting frames for 5 seconds...")
    for _ in range(5 * 30): # 5 seconds at 30 fps
        frames = pipeline.wait_for_frames(10000)  # 10 second timeout
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print("No depth frame.")
            continue
        # print(f"Got depth frame: {depth_frame.get_timestamp()}") # Optional: print timestamp
        time.sleep(1/30.0)
    print("Finished getting frames.")
except Exception as e:
    print(f"RealSense test error: {e}")
finally:
    print("\nStopping RealSense pipeline...")
    pipeline.stop()

    # Hardware reset for next run
    print("Cleaning up with hardware reset...")
    hardware_reset_and_wait(ctx)
    print("RealSense pipeline stopped and reset.")