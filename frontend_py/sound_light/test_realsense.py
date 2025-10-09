import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
config = rs.config()
try:
    print("Attempting to start RealSense pipeline...")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    print("RealSense pipeline started.")
    print("Getting frames for 5 seconds...")
    for _ in range(5 * 30): # 5 seconds at 30 fps
        frames = pipeline.wait_for_frames()
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
    print("Stopping RealSense pipeline in test script...")
    pipeline.stop()
    print("RealSense pipeline stopped in test script.")