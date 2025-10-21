"""
StreamDiffusion Text-to-Image Pipeline with ControlNet and LoRA support

This pipeline uses StreamDiffusion for real-time txt2img generation with:
- Static ControlNet integration for aesthetic guidance (depth-based by default)
- Dynamic LoRA loading via lora_config
- Multiple pipes with different LoRA weight combinations

Design decisions:
- ControlNet is statically enabled for compositional/aesthetic control in txt2img
- LoRAs are loaded per-pipe based on adapter weights from lora_config
- Uses txt2img mode parameters: t_index_list=[0, 16, 32, 45], no denoising batch
"""
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

default_prompt = "mrnabrmv style, Fragmented digital portrait blending abstract textures and vivid colors, creating a surreal, pixelated visage."
default_negative_prompt = "blurry, low quality, distorted, 3d render"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Text-to-Image SD-Turbo + ControlNet</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Text to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with ControlNet for aesthetic guidance and LoRA support.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "txt2imgStreamDiffusion"
        title: str = "Text-to-Image StreamDiffusion + ControlNet"
        description: str = "Generates an image from a text prompt using StreamDiffusion with ControlNet guidance and LoRAs"
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
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        controlnet_scale: float = Field(
            0.87,
            min=0,
            max=2.0,
            step=0.001,
            title="Controlnet Scale",
            field="range",
            hide=True,
            id="controlnet_scale",
        )
        controlnet_start: float = Field(
            0.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Start",
            field="range",
            hide=True,
            id="controlnet_start",
        )
        controlnet_end: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet End",
            field="range",
            hide=True,
            id="controlnet_end",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):
        # Store lora_config for later use
        self.lora_config = lora_config
        self.pipes = []

        # Get adapter weights sets from lora_config to determine number of pipes
        if lora_config is not None:
            adapter_weights_sets = lora_config.get_default_adapter_weights()
            print(f"[txt2imgStreamDiffusion.py] Creating {len(adapter_weights_sets)} pipes based on lora_config")
        else:
            # Default to single pipe if no lora_config provided
            adapter_weights_sets = [[]]
            print(f"[txt2imgStreamDiffusion.py] No lora_config provided, creating single pipe")

        params = self.InputParams()

        # Define ControlNet configuration (static, for aesthetic guidance in txt2img)
        use_controlnet = True
        controlnet_config = {
            'model_id': 'thibaud/controlnet-sd21-depth-diffusers',
            'preprocessor': 'depth',  # 'depth', 'canny', 'pose', etc.
            'conditioning_scale': 0.87,
            'enabled': True,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        print(f"[txt2imgStreamDiffusion.py] ControlNet enabled with model: {controlnet_config['model_id']}")

        # Create one pipe for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[txt2imgStreamDiffusion.py] Creating pipe {idx + 1}/{len(adapter_weights_sets)}")

            # Build lora_dict from lora_config
            lora_dict = None
            if lora_config is not None:
                curation_key = lora_config.get_curation_keys()[0]
                lora_models_list = lora_config.get_lora_curation()[curation_key]
                lora_models_dict = lora_config.get_lora_models()

                # Build lora_dict: {lora_path: scale}
                lora_dict = {}
                for i, lora_name in enumerate(lora_models_list):
                    if lora_name != "None":
                        lora_path = lora_models_dict[lora_name]
                        # Use adapter_weights to determine scale for this LoRA
                        scale = adapter_weights[i] if i < len(adapter_weights) else 1.0
                        lora_dict[lora_path] = scale

                print(f"[txt2imgStreamDiffusion.py] Pipe {idx}: Built lora_dict with {len(lora_dict)} LoRAs")
                print(f"[txt2imgStreamDiffusion.py] lora_dict: {lora_dict}")

            stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                lora_dict=lora_dict,
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[0, 16, 32, 45],
                frame_buffer_size=1,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration="xformers",
                mode="txt2img",
                use_denoising_batch=False,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                use_controlnet=use_controlnet,
                controlnet_config=controlnet_config,
            )

            stream.prepare(
                prompt=default_prompt,
                num_inference_steps=50,
            )

            self.pipes.append(stream)

        # Store current pipe index
        self.current_pipe_idx = 0
        self.last_prompt = default_prompt

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Handle None params by creating default params
        if params is None:
            params = self.InputParams()
            print(f"[txt2imgStreamDiffusion.py] No params provided, using defaults")

        # Get pipe_index from params, default to 0 if not provided
        pipe_index = getattr(params, 'pipe_index', 0)

        # Ensure pipe_index is within bounds
        if pipe_index >= len(self.pipes):
            print(f"[txt2imgStreamDiffusion.py] Warning: pipe_index {pipe_index} out of bounds, using 0")
            pipe_index = 0

        # Select the appropriate stream
        stream = self.pipes[pipe_index]

        # Generate image from prompt
        print(f"[txt2imgStreamDiffusion.py] Params: {params}")

        # If prompt changed, update it via prepare()
        prompt = params.prompt
        if prompt != self.last_prompt:
            stream.prepare(
                prompt=prompt,
                num_inference_steps=50,
            )
            self.last_prompt = prompt

        # Update ControlNet control image if provided (for aesthetic guidance)
        # ControlNet is statically enabled for this pipeline
        print(f"[txt2imgStreamDiffusion.py] params: {params}")
        
        control_image = getattr(params, 'control_image', None)
        if control_image is not None:
            print(f"[txt2imgStreamDiffusion.py] Updating control image for ControlNet guidance")
            stream.update_control_image(index=0, image=control_image)

        # For txt2img mode, call stream() without parameters
        # Do warmup iterations
        for _ in range(stream.batch_size - 1):
            stream()

        # Generate final image
        output_image = stream()

        return output_image
