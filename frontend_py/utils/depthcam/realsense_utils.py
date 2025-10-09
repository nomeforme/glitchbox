"""
RealSense utility functions for device management and reset handling.
"""
import pyrealsense2 as rs
import time


def wait_for_device(ctx, timeout=10):
    """
    Wait for a RealSense device to be available.

    Args:
        ctx: RealSense context object
        timeout: Maximum time to wait in seconds

    Returns:
        Device object if found, None otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        devices = ctx.query_devices()
        if len(devices) > 0:
            return devices[0]
        time.sleep(0.1)
    return None


def hardware_reset_and_wait(ctx, verbose=True):
    """
    Perform hardware reset and wait for device to reconnect.

    Args:
        ctx: RealSense context object
        verbose: Print status messages

    Returns:
        True if reset successful and device reconnected, False otherwise
    """
    if verbose:
        print("Performing hardware reset...")

    devices = ctx.query_devices()
    if len(devices) > 0:
        device = devices[0]
        device.hardware_reset()

        if verbose:
            print("Hardware reset command sent. Waiting for device to reconnect...")

        time.sleep(2)  # Initial wait for USB to fully disconnect

        # Wait for device to reappear
        device = wait_for_device(ctx, timeout=10)
        if device:
            if verbose:
                print(f"Device reconnected: {device.get_info(rs.camera_info.name)}")
            return True
        else:
            if verbose:
                print("Device did not reconnect within timeout")
            return False
    else:
        if verbose:
            print("No device found to reset")
        return False


def create_pipeline_with_reset(ctx=None):
    """
    Create a RealSense pipeline with initial hardware reset for clean state.

    Args:
        ctx: Optional RealSense context. If None, creates a new one.

    Returns:
        Tuple of (pipeline, context) ready for configuration
    """
    if ctx is None:
        ctx = rs.context()

    # Reset to ensure clean state
    hardware_reset_and_wait(ctx)

    # Create pipeline
    pipeline = rs.pipeline(ctx)

    return pipeline, ctx
