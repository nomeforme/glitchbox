import pyrealsense2 as rs
import time

def wait_for_device(ctx, timeout=10):
    """Wait for a RealSense device to be available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        devices = ctx.query_devices()
        if len(devices) > 0:
            return devices[0]
        time.sleep(0.1)
    return None

def hardware_reset_and_wait(ctx):
    """Perform hardware reset and wait for device to reconnect"""
    print("Performing hardware reset...")
    devices = ctx.query_devices()
    if len(devices) > 0:
        device = devices[0]
        device.hardware_reset()
        print("Hardware reset command sent. Waiting for device to reconnect...")
        time.sleep(2)  # Initial wait for USB to fully disconnect

        # Wait for device to reappear
        device = wait_for_device(ctx, timeout=10)
        if device:
            print(f"Device reconnected: {device.get_info(rs.camera_info.name)}")
            return True
        else:
            print("Device did not reconnect within timeout")
            return False
    return False

try:
    # Create context (persistent across runs)
    ctx = rs.context()

    # Try hardware reset first to ensure clean state
    hardware_reset_and_wait(ctx)

    # Now create pipeline and start
    pipeline = rs.pipeline(ctx)
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
        time.sleep(1/30.0)
    print("Finished getting frames.")

except Exception as e:
    print(f"RealSense test error: {e}")
finally:
    print("\nStopping RealSense pipeline...")
    pipeline.stop()

    # Perform hardware reset to clean up for next run
    print("Cleaning up with hardware reset...")
    hardware_reset_and_wait(ctx)
    print("RealSense pipeline stopped and reset.")
