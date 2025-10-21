"""
StreamDiffusion Image-to-Image SDXL Pipeline with ControlNet and LoRA support

This pipeline uses StreamDiffusion for real-time SDXL img2img generation with:
- Stable Diffusion XL base model
- SDXL-compatible ControlNet for structural guidance (depth-based by default)
- Dynamic LoRA loading via lora_config
- Multiple pipes with different LoRA weight combinations

Design decisions:
- Uses SDXL model: stabilityai/stable-diffusion-xl-base-1.0
- ControlNet is statically enabled for compositional/structural control in img2img
- LoRAs are loaded per-pipe based on adapter weights from lora_config
- Uses img2img mode parameters: t_index_list=[22, 32, 45], with denoising batch
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

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel

base_model = "stabilityai/sd-turbo"
# base_model = "stabilityai/sd-turbo"
# base_model = "stabilityai/stable-diffusion-2-1-base"
# base_model = "KBlueLeaf/kohaku-v2.1"
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesd"

default_prompt = "mrnabrmv style, Fragmented digital portrait blending abstract textures and vivid colors, creating a surreal, pixelated visage."
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SDXL + ControlNet</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Stable Diffusion XL</a
    > with ControlNet for structural guidance and LoRA support.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2imgStreamDiffusion"
        title: str = "Image-to-Image SDXL StreamDiffusion + ControlNet"
        description: str = "Generates an image from an input image using SDXL StreamDiffusion with ControlNet guidance and LoRAs"
        input_mode: str = "image"
        page_content: str = page_content
    
    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        target_prompt: str = Field(
            default_prompt,
            title="Target Prompt",
            field="textarea",
            id="target_prompt",
            hide=True,
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
        use_prompt_travel: bool = Field(
            True,
            title="Use Prompt Travel",
            field="checkbox",
            id="use_prompt_travel",
        )
        prompt_travel_factor: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Prompt Travel Factor",
            field="range",
            id="prompt_travel_factor",
            hide=True,
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
            print(f"[img2imgStreamDiffusion.py] Creating {len(adapter_weights_sets)} pipes based on lora_config")
        else:
            # Default to single pipe if no lora_config provided
            adapter_weights_sets = [[]]
            print(f"[img2imgStreamDiffusion.py] No lora_config provided, creating single pipe")

        params = self.InputParams()

        # Define ControlNet configuration (static, for structural guidance in img2img)
        use_controlnet = True
        controlnet_config = {
            'model_id': 'thibaud/controlnet-sd21-depth-diffusers',
            'preprocessor': 'depth',  # 'depth', 'canny', 'pose', etc.
            'conditioning_scale': 0.87,
            'enabled': True,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        print(f"[img2imgStreamDiffusion.py] ControlNet enabled with SDXL model: {controlnet_config['model_id']}")

        # Create one pipe for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[img2imgStreamDiffusion.py] Creating pipe {idx + 1}/{len(adapter_weights_sets)}")

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

                print(f"[img2imgStreamDiffusion.py] Pipe {idx}: Built lora_dict with {len(lora_dict)} LoRAs")

            stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                lora_dict=lora_dict,
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[22, 32, 45],
                frame_buffer_size=1,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pil",
                warmup=10,
                vae_id=None,
                acceleration="xformers",
                mode="img2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                use_controlnet=use_controlnet,
                controlnet_config=controlnet_config,
            )

            stream.prepare(
                prompt=default_prompt,
                negative_prompt=default_negative_prompt,
                num_inference_steps=50,
                guidance_scale=1.2,
            )

            # Initialize PromptTravel for this pipe to enable prompt embedding interpolation
            # Access text_encoder and tokenizer from the inner stream object
            stream.prompt_travel = PromptTravel(
                text_encoder=stream.stream.text_encoder,
                tokenizer=stream.stream.pipe.tokenizer,
            )

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

        # Select the appropriate stream wrapper
        stream_wrapper = self.pipes[pipe_index]

        # Generate image from input image and prompt
        print(f"[img2imgStreamDiffusion.py] Params: {params}")

        # Handle prompt travel if enabled
        use_prompt_travel = getattr(params, "use_prompt_travel", False)
        print(f"[img2imgStreamDiffusion.py] use_prompt_travel: {use_prompt_travel}")

        if use_prompt_travel:
            # Get prompts and factor
            source_prompt = params.prompt
            target_prompt = getattr(params, 'target_prompt', params.prompt)
            prompt_travel_factor = getattr(params, 'prompt_travel_factor', 0.5)

            print(f"[img2imgStreamDiffusion.py] Calculating prompt travel embeddings")
            print(f"[img2imgStreamDiffusion.py] source: {source_prompt}")
            print(f"[img2imgStreamDiffusion.py] target: {target_prompt}")
            print(f"[img2imgStreamDiffusion.py] factor: {prompt_travel_factor}")

            # Encode source and target prompts
            source_embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                prompt=source_prompt,
                device=stream_wrapper.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            target_embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                prompt=target_prompt,
                device=stream_wrapper.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            # Interpolate between embeddings
            interpolated_embeds = stream_wrapper.prompt_travel.interpolate_embeddings(
                embeds_from=source_embeds,
                embeds_to=target_embeds,
                factor=prompt_travel_factor,
            )

            print(f"[img2imgStreamDiffusion.py] Interpolated embeddings shape: {interpolated_embeds.shape}")

            # StreamDiffusion repeats embeddings for batch_size, so we need to match that
            batch_size = stream_wrapper.stream.batch_size
            interpolated_embeds_batched = interpolated_embeds.repeat(batch_size, 1, 1)

            # Directly set the embeddings on the inner stream object
            stream_wrapper.stream.prompt_embeds = interpolated_embeds_batched

            print(f"[img2imgStreamDiffusion.py] Set prompt_embeds with shape: {interpolated_embeds_batched.shape}")

        else:
            # If prompt changed and not using prompt travel, update it via prepare()
            prompt = params.prompt
            if prompt != self.last_prompt:
                stream_wrapper.prepare(
                    prompt=prompt,
                    negative_prompt=default_negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=1.2,
                )
                self.last_prompt = prompt

            print(f"[img2imgStreamDiffusion.py] NOTE: No prompt travel used, prepared prompt: {prompt}")

        # Update ControlNet control image (use input image for structural guidance)
        # ControlNet is statically enabled for this pipeline
        control_image = getattr(params, 'control_image', params.image)
        if control_image is not None:
            print(f"[img2imgStreamDiffusion.py] Updating control image for ControlNet structural guidance")
            stream_wrapper.update_control_image(index=0, image=control_image)

        # Preprocess input image and generate
        image_tensor = stream_wrapper.preprocess_image(params.image)
        output_image = stream_wrapper(image=image_tensor)

        return output_image