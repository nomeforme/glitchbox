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

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel

base_model = "stabilityai/sdxl-turbo"
# base_model = "KBlueLeaf/kohaku-v2.1"
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesdxl"

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
        name: str = "txt2imgStreamDiffusionXL"
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
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            1024, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
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
        upscaler_scale_factor: int = Field(
            2,
            min=2,
            max=4,
            step=2,
            title="Upscaler Scale",
            field="range",
            hide=True,
            id="upscaler_scale_factor",
        )
        debug_controlnet: bool = Field(
            False,
            title="Debug ControlNet",
            field="checkbox",
            hide=True,
            id="debug_controlnet",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):
        # Store lora_config for later use
        self.lora_config = lora_config
        self.pipes = []

        # Check if upscaler is enabled
        self.use_upscaler = getattr(args, 'use_upscaler', False)
        self.upscaler_scale_factor = getattr(args, 'upscaler_scale_factor', 2)
        if self.use_upscaler:
            print(f"[txt2imgStreamDiffusion.py] RealESRGAN {self.upscaler_scale_factor}x upscaler enabled (TensorRT)")

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
            'model_id': 'diffusers/controlnet-depth-sdxl-1.0',  # SDXL depth ControlNet
            'preprocessor': 'passthrough',  # Uses input image directly (for physical depth camera)
            # 'preprocessor': 'depth',  # Uncomment to calculate depth from RGB
            # 'preprocessor_params': {
            #     'model_name': 'Intel/dpt-swinv2-tiny-256',  # ~165MB, fastest
            #     # 'model_name': 'Intel/dpt-large',  # ~1.3GB, slower but higher quality
            # },
            'conditioning_scale': 0.67,
            'enabled': False,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        print(f"[txt2imgStreamDiffusion.py] ControlNet enabled with preprocessor type: {controlnet_config['preprocessor']}")

        # Define image postprocessing configuration (RealESRGAN upscaler)
        image_postprocessing_config = None
        if self.use_upscaler:
            image_postprocessing_config = {
                'enabled': False,
                'processors': [
                    {
                        'type': 'realesrgan_trt',
                        'params': {
                            'scale_factor': self.upscaler_scale_factor,
                            'enable_tensorrt': True,
                            'force_rebuild': False
                        }
                    }
                ]
            }
            print(f"[txt2imgStreamDiffusion.py] Image postprocessing configured with RealESRGAN {self.upscaler_scale_factor}x upscaler")

        # Create one pipe for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[txt2imgStreamDiffusion.py] Creating pipe {idx + 1}/{len(adapter_weights_sets)}")
            print(f"[txt2imgStreamDiffusion.py] Pipe {idx}: adapter_weights = {adapter_weights}")

            # Create the StreamDiffusionWrapper WITHOUT lora_dict
            # We'll load LoRAs manually afterwards to support per-pipe adapter weights
            stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                lora_dict=None,  # Don't use lora_dict - we'll load manually
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[12, 22, 32, 38, 45],
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
                image_postprocessing_config=image_postprocessing_config,
            )

            # # Load LoRAs manually with adapter weights (like controlnetSDTurbot2i)
            # if lora_config is not None:
            #     curation_key = lora_config.get_curation_keys()[0]
            #     lora_models_list = lora_config.get_lora_curation()[curation_key]
            #     lora_models_dict = lora_config.get_lora_models()

            #     # Filter out "None" entries
            #     selected_loras = [name for name in lora_models_list if name != "None"]

            #     if selected_loras:
            #         print(f"[txt2imgStreamDiffusion.py] Loading {len(selected_loras)} LoRAs: {selected_loras}")

            #         # Ensure adapter_weights matches the number of loras
            #         if len(adapter_weights) < len(selected_loras):
            #             adapter_weights = adapter_weights + [1.0] * (len(selected_loras) - len(adapter_weights))
            #         elif len(adapter_weights) > len(selected_loras):
            #             adapter_weights = adapter_weights[:len(selected_loras)]

            #         # Load each LoRA with an adapter name
            #         adapter_names = []
            #         for i, lora_name in enumerate(selected_loras):
            #             adapter_name = f"lora_{i}"
            #             lora_path = lora_models_dict[lora_name]
            #             print(f"[txt2imgStreamDiffusion.py] Loading LoRA {i}: {lora_name} as {adapter_name} with weight {adapter_weights[i]}")
            #             stream.stream.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            #             adapter_names.append(adapter_name)

            #         # Set adapter weights and fuse
            #         print(f"[txt2imgStreamDiffusion.py] Setting adapters with weights: {adapter_weights}")
            #         stream.stream.pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)

            #         print(f"[txt2imgStreamDiffusion.py] Fusing LoRAs with scale 1.0")
            #         stream.stream.pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)

            #         # Unload after fusing to free memory
            #         stream.stream.pipe.unload_lora_weights()
            #         print(f"[txt2imgStreamDiffusion.py] LoRAs loaded and fused successfully")

            stream.prepare(
                prompt=default_prompt,
                num_inference_steps=50,
            )

            # Initialize PromptTravel for this pipe to enable prompt embedding interpolation
            # For SDXL, pass both text encoders and tokenizers
            stream.prompt_travel = PromptTravel(
                text_encoder=stream.stream.pipe.text_encoder,
                tokenizer=stream.stream.pipe.tokenizer,
                text_encoder_2=stream.stream.pipe.text_encoder_2,
                tokenizer_2=stream.stream.pipe.tokenizer_2,
            )

            self.pipes.append(stream)

        # Store current pipe index
        self.current_pipe_idx = 0
        self.last_prompt = default_prompt

        # Cache for prompt travel embeddings (per pipe)
        self.prompt_embeds_cache = {}  # {pipe_idx: {prompt: (embeds, pooled)}}

        # Initialize cache for each pipe
        for idx in range(len(self.pipes)):
            self.prompt_embeds_cache[idx] = {}

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

        # Handle prompt travel if enabled
        use_prompt_travel = getattr(params, "use_prompt_travel", False)
        print(f"[txt2imgStreamDiffusion.py] use_prompt_travel: {use_prompt_travel}")

        if use_prompt_travel:
            # Get prompts and factor
            source_prompt = params.prompt
            target_prompt = getattr(params, 'target_prompt', params.prompt)
            prompt_travel_factor = getattr(params, 'prompt_travel_factor', 0.5)

            print(f"[txt2imgStreamDiffusionXL.py] SDXL prompt travel - factor: {prompt_travel_factor}")
            print(f"[txt2imgStreamDiffusionXL.py] source: {source_prompt[:50]}...")
            print(f"[txt2imgStreamDiffusionXL.py] target: {target_prompt[:50]}...")

            # Get or compute source embeddings (with caching)
            cache = self.prompt_embeds_cache[pipe_index]
            if source_prompt not in cache:
                print(f"[txt2imgStreamDiffusionXL.py] Cache MISS - encoding source prompt")
                source_embeds, _, source_pooled, _ = stream.prompt_travel.encode_prompt_sdxl(
                    prompt=source_prompt,
                    device=stream.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                cache[source_prompt] = (source_embeds, source_pooled)
            else:
                print(f"[txt2imgStreamDiffusionXL.py] Cache HIT - reusing source embeddings")
                source_embeds, source_pooled = cache[source_prompt]

            # Get or compute target embeddings (with caching)
            if target_prompt not in cache:
                print(f"[txt2imgStreamDiffusionXL.py] Cache MISS - encoding target prompt")
                target_embeds, _, target_pooled, _ = stream.prompt_travel.encode_prompt_sdxl(
                    prompt=target_prompt,
                    device=stream.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                cache[target_prompt] = (target_embeds, target_pooled)
            else:
                print(f"[txt2imgStreamDiffusionXL.py] Cache HIT - reusing target embeddings")
                target_embeds, target_pooled = cache[target_prompt]

            # Interpolate between embeddings (both concatenated and pooled)
            interpolated_embeds, interpolated_pooled = stream.prompt_travel.interpolate_embeddings_sdxl(
                embeds_from=(source_embeds, source_pooled),
                embeds_to=(target_embeds, target_pooled),
                factor=prompt_travel_factor,
            )

            print(f"[txt2imgStreamDiffusionXL.py] Interpolated embeddings shape: {interpolated_embeds.shape}")
            print(f"[txt2imgStreamDiffusionXL.py] Interpolated pooled embeddings shape: {interpolated_pooled.shape}")

            # StreamDiffusion repeats embeddings for batch_size, so we need to match that
            batch_size = stream.stream.batch_size
            interpolated_embeds_batched = interpolated_embeds.repeat(batch_size, 1, 1)

            # Directly set both the concatenated embeddings and pooled embeddings
            stream.stream.prompt_embeds = interpolated_embeds_batched
            stream.stream.add_text_embeds = interpolated_pooled  # SDXL pooled embeddings

            print(f"[txt2imgStreamDiffusionXL.py] Set prompt_embeds with shape: {interpolated_embeds_batched.shape}")
            print(f"[txt2imgStreamDiffusionXL.py] Set add_text_embeds with shape: {interpolated_pooled.shape}")

        else:
            # If prompt changed and not using prompt travel, update it via prepare()
            prompt = params.prompt
            if prompt != self.last_prompt:
                stream.prepare(
                    prompt=prompt,
                    num_inference_steps=50,
                )
                self.last_prompt = prompt

            print(f"[txt2imgStreamDiffusion.py] NOTE: No prompt travel used, prepared prompt: {prompt}")

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

        # Debug controlnet: paste preprocessed control image in bottom-right corner
        if params.debug_controlnet:
            # Get the preprocessed control image (depth map) that ControlNet is using
            preprocessed_control = stream.get_last_processed_image(index=0)

            if preprocessed_control is not None:
                if self.use_upscaler:
                    scale_factor = 2  # Use server default scale (2x instead of 4x)
                else:
                    scale_factor = 1

                w0, h0 = (scale_factor * 200, scale_factor * 200)
                control_image_resized = preprocessed_control.resize((w0, h0))
                w1, h1 = output_image.size
                output_image.paste(control_image_resized, (w1 - w0, h1 - h0))

        return output_image
