import sys
import os

# Update import to use local pipelines directory
from pipelines.streamdiffusion.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
# base_model = "KBlueLeaf/kohaku-v2.1"
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2imgStreamDiffusion"
        title: str = "Image-to-Image StreamDiffusion"
        description: str = "Generates an image from a text prompt using StreamDiffusion"
        input_mode: str = "image"
        page_content: str = page_content
    
    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        pipe_index: int = Field(
            0,
            min=0,
            max=10,
            step=1,
            title="Pipe Index",
            field="range",
            id="pipe_index",
            description="Select which pipe (LoRA combination) to use"
        )
        width: int = Field(
            640, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            480, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):
        # Store lora_config for later use
        self.lora_config = lora_config
        self.pipes = []

        # Get adapter weights sets from lora_config to determine number of pipes
        if lora_config is not None:
            adapter_weights_sets = lora_config.get_default_adapter_weights()
            print(f"[img2imgStreamDiffusion.py] Creating {len(adapter_weights_sets)} pipes based on lora_config")
        else:
            # Default to single pipe if no lora_config provided
            adapter_weights_sets = [[]]
            print(f"[img2imgStreamDiffusion.py] No lora_config provided, creating single pipe")

        params = self.InputParams()

        # Create one pipe for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[img2imgStreamDiffusion.py] Creating pipe {idx + 1}/{len(adapter_weights_sets)}")

            stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[35, 45],
                frame_buffer_size=1,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration="tensorrt",
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
            )

            stream.prepare(
                prompt=default_prompt,
                negative_prompt=default_negative_prompt,
                num_inference_steps=50,
                guidance_scale=1.2,
            )

            # TODO: Load LoRAs here based on adapter_weights (not implemented yet)

            self.pipes.append(stream)

        # Store current pipe index
        self.current_pipe_idx = 0
        self.last_prompt = default_prompt

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Get pipe_index from params, default to 0 if not provided
        pipe_index = getattr(params, 'pipe_index', 0)

        # Ensure pipe_index is within bounds
        if pipe_index >= len(self.pipes):
            print(f"[img2imgStreamDiffusion.py] Warning: pipe_index {pipe_index} out of bounds, using 0")
            pipe_index = 0

        # Select the appropriate stream
        stream = self.pipes[pipe_index]

        # Preprocess image and generate
        image_tensor = stream.preprocess_image(params.image)
        output_image = stream(image=image_tensor, prompt=params.prompt)

        return output_image