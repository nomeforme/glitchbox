import pyvirtualcam
import numpy as np
import time

WIDTH, HEIGHT, FPS = 640, 480, 30
TARGET_VIRTUAL_DEVICE = '/dev/video42'

print(f"Attempting to use virtual camera: {TARGET_VIRTUAL_DEVICE}")
try:
    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS,
                             fmt=pyvirtualcam.PixelFormat.RGB,
                             device=TARGET_VIRTUAL_DEVICE) as cam:
        print(f'Virtual camera started: {cam.device}')
        print("Sending frames for 10 seconds...")
        for i in range(FPS * 10):
            frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8) # Black frame
            frame[:, :, i % 3] = 255 # Cycle through R, G, B fill
            cam.send(frame)
            cam.sleep_until_next_frame()
        print("Finished sending frames.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Script finished.")