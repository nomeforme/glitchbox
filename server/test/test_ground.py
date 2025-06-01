from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny
import torch
import time

class Timer:
    def __init__(self, description="Code block"):
        self.description = description
        
    def __enter__(self):
        import time  # Import only when needed
        self.start = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time  # Import only when needed
        elapsed = time.time() - self.start
        print(f"{self.description} took {elapsed:.4f} seconds to run")

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-sd21-canny-diffusers",
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("../Real-Time-Latent-Consistency-Model/loras/FKATwigs_A1-000038.safetensors")
pipe.fuse_lora()
pipe.unload_lora_weights()

# print("Using sfast compile\n")
# from sfast.compilers.stable_diffusion_pipeline_compiler import (
#     compile,
#     CompilationConfig,
# )

# config = CompilationConfig.Default()
# config.enable_xformers = True
# config.enable_triton = True
# config.enable_cuda_graph = True
# pipe = compile(pipe, config=config)

# print("\nRunning with sfast compile\n")

from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

original_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

for i in range(1, 10):  # This will run 5 times (1 through 5)
    with Timer(f"Image {i}"):
        output = pipe(
            "the mona lisa", 
            image=control_image, 
            num_inference_steps=1, 
            guidance_scale=1.21,
            height=640,
            width=480,
        ).images[0]

output.save("test.png")