import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

import os
import sys

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(parent_directory)
# Append the parent directory to sys.path
sys.path.append(parent_directory)

from StreamDiffusion.src.streamdiffusion import StreamDiffusion
from StreamDiffusion.src.streamdiffusion.image_utils import postprocess_image
from StreamDiffusion.src.streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
# from StreamDiffusion.src.streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

# You can load any models using diffuser's StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

# Wrap the pipeline in StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45],
    torch_dtype=torch.float16,
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

############# ACCELERATE #############

# Enable acceleration with xformers memory efficient attention
# pipe.enable_xformers_memory_efficient_attention()

stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size=2,
)

# Uncomment the following line to use StableFast
# stream = accelerate_with_stable_fast(stream)

###################################################################

prompt = "1girl with dog hair, thick frame glasses"
# Prepare the stream
stream.prepare(prompt)

# Prepare image
init_image = load_image(os.path.dirname(__file__)+"/assets/img2img_example.png").resize((512, 512))

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    stream(init_image)

# Import time for measuring performance
import time
import numpy as np

# Number of iterations
iterations = 1000
times = []

# Run the stream for specified iterations
for i in range(iterations):
    start_time = time.time()
    x_output = stream(init_image)
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    
    if (i + 1) % 100 == 0:
        print(f"Completed {i+1}/{iterations} iterations")

# Calculate statistics
times = np.array(times)
avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)
std_time = np.std(times)
fps = 1.0 / avg_time

# Print statistics
print("\nPerformance Statistics:")
print(f"Total images generated: {iterations}")
print(f"Average time per image: {avg_time:.4f} seconds")
print(f"Minimum time per image: {min_time:.4f} seconds")
print(f"Maximum time per image: {max_time:.4f} seconds")
print(f"Standard deviation: {std_time:.4f} seconds")
print(f"Frames per second (FPS): {fps:.2f}")