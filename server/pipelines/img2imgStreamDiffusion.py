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
import glob

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel

# Function to read prompt prefix from .txt files
def get_prompt_prefix():
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))
    
    if not prompt_files:
        # Default prompt prefix if no files are found
        return ""
    
    # Read the first prompt file
    with open(prompt_files[0], 'r') as f:
        return f.read().strip()

prompt_prefix = get_prompt_prefix()
default_prompt = prompt_prefix + "mrnabrmv style, Fragmented digital portrait blending abstract textures and vivid colors, creating a surreal, pixelated visage."
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

base_model = "stabilityai/sd-turbo"
# base_model = "stabilityai/sd-turbo"
# base_model = "stabilityai/stable-diffusion-2-1-base"
# base_model = "KBlueLeaf/kohaku-v2.1"
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesd"

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
        client_prompt_prefix: str = Field(
            prompt_prefix,
            title="Client Prompt Prefix",
            field="textarea",
            id="client_prompt_prefix",
            description="Prefix to prepend to client-provided prompts from STT",
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
            680, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
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
        temporal_coherence: float = Field(
            0.03,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Temporal Coherence: Noise Blending",
            field="range",
            id="temporal_coherence",
            description="Blend previous latent into noise (0=off, subtle effect)",
        )
        temporal_coherence_latent: float = Field(
            0.03,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Temporal Coherence: Latent Blending",
            field="range",
            id="temporal_coherence_latent",
            description="Blend input with previous output latent (0=off, stronger effect)",
        )
        # Add boost factor parameters
        boost_factor_bass: float = Field(
            1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Bass",
            field="range",
            id="boost_factor_bass",
        )
        boost_factor_low_mids: float = Field(
            1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Low Mids",
            field="range",
            id="boost_factor_low_mids",
        )
        boost_factor_mids: float = Field(
            1.5,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Mids",
            field="range",
            id="boost_factor_mids",
        )
        boost_factor_high_mids: float = Field(
            2.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - High Mids",
            field="range",
            id="boost_factor_high_mids",
        )
        boost_factor_treble: float = Field(
            2.5,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Treble",
            field="range",
            id="boost_factor_treble",
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
        use_output_bg_removal: bool = Field(
            False,
            title="Use Output Background Removal",
            field="checkbox",
            id="use_output_bg_removal",
        )
        use_prompt_indexing: bool = Field(
            False,
            title="Use Prompt Indexing",
            field="checkbox",
            id="use_prompt_indexing",
            description="Use pipe index to select prompts from file instead of sequential scheduling",
        )
        use_client_prompts: bool = Field(
            False,
            title="Use Client Prompts",
            field="checkbox",
            id="use_client_prompts",
            description="Use client-provided prompts instead of scheduled prompts for prompt travel",
        )
        debug_controlnet: bool = Field(
            False,
            title="Debug ControlNet",
            field="checkbox",
            hide=True,
            id="debug_controlnet",
        )
        use_depth_masking: bool = Field(
            True,
            title="Use Depth Masking",
            field="checkbox",
            id="use_depth_masking",
            description="Apply depth-based masking to remove background from input image",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):
        # Store lora_config for later use
        self.lora_config = lora_config
        self.pipes = []

        # Check if upscaler is enabled
        self.use_upscaler = getattr(args, 'use_upscaler', False)
        self.upscaler_scale_factor = getattr(args, 'upscaler_scale_factor', 2)
        if self.use_upscaler:
            print(f"[img2imgStreamDiffusion.py] RealESRGAN {self.upscaler_scale_factor}x upscaler enabled (TensorRT)")

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
            'preprocessor': 'passthrough',  # 'depth', 'canny', 'pose', etc.
            # 'preprocessor_params': {
            #     'model_name': 'Intel/dpt-swinv2-tiny-256',  # ~165MB, fastest
            #     # 'model_name': 'Intel/dpt-large',  # ~1.3GB, slower but higher quality
            # },
            'conditioning_scale': 0.87,
            'enabled': True,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        print(f"[img2imgStreamDiffusion.py] ControlNet enabled with SDXL model: {controlnet_config['model_id']}")

        # Define image postprocessing configuration (RealESRGAN upscaler)
        image_postprocessing_config = None
        if self.use_upscaler:
            image_postprocessing_config = {
                'enabled': True,
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
            print(f"[img2imgStreamDiffusion.py] Image postprocessing configured with RealESRGAN {self.upscaler_scale_factor}x upscaler")

        # Create one pipe for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[img2imgStreamDiffusion.py] Creating pipe {idx + 1}/{len(adapter_weights_sets)}")
            print(f"[img2imgStreamDiffusion.py] Pipe {idx}: adapter_weights = {adapter_weights}")

            # Build lora_dict with adapter weights for this pipe
            # Format: {lora_path: weight, ...}
            # This will be processed by wrapper BEFORE TensorRT compilation
            lora_dict = None
            if lora_config is not None:
                curation_key = lora_config.get_curation_keys()[0]
                lora_models_list = lora_config.get_lora_curation()[curation_key]
                lora_models_dict = lora_config.get_lora_models()

                # Filter out "None" entries
                selected_loras = [name for name in lora_models_list if name != "None"]

                if selected_loras:
                    print(f"[img2imgStreamDiffusion.py] Building lora_dict with {len(selected_loras)} LoRAs: {selected_loras}")

                    # Ensure adapter_weights matches the number of loras
                    if len(adapter_weights) < len(selected_loras):
                        adapter_weights = adapter_weights + [1.0] * (len(selected_loras) - len(adapter_weights))
                    elif len(adapter_weights) > len(selected_loras):
                        adapter_weights = adapter_weights[:len(selected_loras)]

                    # Build lora_dict: {lora_path: adapter_weight}
                    lora_dict = {}
                    for i, lora_name in enumerate(selected_loras):
                        lora_path = lora_models_dict[lora_name]
                        weight = adapter_weights[i]
                        lora_dict[lora_path] = weight
                        print(f"[img2imgStreamDiffusion.py] lora_dict['{lora_name}'] = {weight}")

            # Create the StreamDiffusionWrapper with lora_dict
            # The wrapper will load and fuse LoRAs BEFORE TensorRT compilation
            stream = StreamDiffusionWrapper(
                model_id_or_path=base_model,
                lora_dict=lora_dict,  # Will be fused before TensorRT compilation
                use_tiny_vae=args.taesd,
                device=device,
                dtype=torch_dtype,
                t_index_list=[5, 18, 32, 45],
                frame_buffer_size=1,
                width=params.width,
                height=params.height,
                use_lcm_lora=False,
                output_type="pil",
                warmup=0,
                vae_id=None,
                acceleration="none",  # TensorRT will compile AFTER LoRA fusion
                mode="txt2img",
                use_denoising_batch=True,
                cfg_type="none",
                use_safety_checker=args.safety_checker,
                use_controlnet=use_controlnet,
                controlnet_config=controlnet_config,
                image_postprocessing_config=image_postprocessing_config,
            )
            print(f"[img2imgStreamDiffusion.py] StreamDiffusionWrapper created with LoRAs fused before TensorRT")

            stream.prepare(
                prompt=default_prompt,
                negative_prompt=default_negative_prompt,
                num_inference_steps=50,
                guidance_scale=1.0,
            )

            # Initialize PromptTravel for this pipe to enable prompt embedding interpolation
            # Access text_encoder and tokenizer from the inner stream object
            # NOTE: img2imgStreamDiffusion only uses non-SDXL models, so explicitly set SDXL encoders to None

            # Debug: Check text encoder config
            text_encoder = stream.stream.text_encoder
            if hasattr(text_encoder, 'config'):
                hidden_size = getattr(text_encoder.config, 'hidden_size', 'unknown')
                print(f"[img2imgStreamDiffusion.py] Text encoder hidden_size: {hidden_size}")

            stream.prompt_travel = PromptTravel(
                text_encoder=text_encoder,
                tokenizer=stream.stream.pipe.tokenizer,
                text_encoder_2=None,
                tokenizer_2=None,
            )
            print(f"[img2imgStreamDiffusion.py] PromptTravel initialized for non-SDXL model (base: {base_model})")

            self.pipes.append(stream)

            # CPU Offloading: Clean up PyTorch models IMMEDIATELY after TensorRT compilation
            # This frees GPU memory before loading the next pipe
            print(f"[img2imgStreamDiffusion.py] Cleaning up PyTorch models for pipe {idx}...")
            stream.cleanup_pytorch_models_after_tensorrt()
            torch.cuda.empty_cache()
            print(f"[img2imgStreamDiffusion.py] Cleaned up PyTorch models for pipe {idx}")

        # Store current pipe index
        self.current_pipe_idx = 0
        self.last_prompt = default_prompt
        self.last_controlnet_scale = None

        # Cache for prompt travel embeddings (per pipe)
        self.prompt_embeds_cache = {}  # {pipe_idx: {prompt: embeds}}

        # Initialize cache for each pipe
        for idx in range(len(self.pipes)):
            self.prompt_embeds_cache[idx] = {}

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Get pipe_index from params, default to 0 if not provided
        pipe_index = getattr(params, 'pipe_index', 0)
        print(f"[img2imgStreamDiffusion.py] USING PIPE INDEX: {pipe_index}")

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
            # Check if we're in multi-file mode
            multi_file_mode = getattr(params, 'multi_file_prompts', False)
            prompt_travel_factor = getattr(params, 'prompt_travel_factor', 0.5)

            if multi_file_mode:
                # MULTI-FILE MODE: Weighted spatial + temporal blending (like SDXL pipeline)
                source_prompts = params.prompt  # List of prompts
                target_prompts = getattr(params, 'target_prompt', params.prompt)  # List of prompts
                spatial_weights = getattr(params, 'spatial_weights', None)

                print(f"[img2imgStreamDiffusion.py] === MULTI-FILE PROMPT TRAVEL ===")
                print(f"[img2imgStreamDiffusion.py] Number of files: {len(source_prompts)}")
                print(f"[img2imgStreamDiffusion.py] Spatial weights: {spatial_weights}")
                print(f"[img2imgStreamDiffusion.py] Temporal factor: {prompt_travel_factor:.3f}")

                # Encode all source prompts and compute weighted spatial blend
                cache = self.prompt_embeds_cache[pipe_index]
                source_embeds_list = []
                for i, prompt in enumerate(source_prompts):
                    if prompt not in cache:
                        embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                            prompt=prompt,
                            device=stream_wrapper.stream.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        cache[prompt] = embeds
                        print(f"[img2imgStreamDiffusion.py] Cache MISS - encoding source prompt {i} (weight={spatial_weights[i]:.2f}): {prompt[:60]}...")
                    else:
                        embeds = cache[prompt]
                        print(f"[img2imgStreamDiffusion.py] Cache HIT - reusing source prompt {i} (weight={spatial_weights[i]:.2f}): {prompt[:60]}...")

                    source_embeds_list.append(embeds)

                # Weighted sum for source (spatial blending)
                source_embeds = sum(w * e for w, e in zip(spatial_weights, source_embeds_list))

                # Encode all target prompts and compute weighted spatial blend
                target_embeds_list = []
                for i, prompt in enumerate(target_prompts):
                    if prompt not in cache:
                        embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                            prompt=prompt,
                            device=stream_wrapper.stream.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        cache[prompt] = embeds
                        print(f"[img2imgStreamDiffusion.py] Cache MISS - encoding target prompt {i} (weight={spatial_weights[i]:.2f}): {prompt[:60]}...")
                    else:
                        embeds = cache[prompt]
                        print(f"[img2imgStreamDiffusion.py] Cache HIT - reusing target prompt {i} (weight={spatial_weights[i]:.2f}): {prompt[:60]}...")

                    target_embeds_list.append(embeds)

                # Weighted sum for target (spatial blending)
                target_embeds = sum(w * e for w, e in zip(spatial_weights, target_embeds_list))

                print(f"[img2imgStreamDiffusion.py] Spatially blended source/target embeddings")
                print(f"[img2imgStreamDiffusion.py] Source embeddings shape: {source_embeds.shape}")
                print(f"[img2imgStreamDiffusion.py] Target embeddings shape: {target_embeds.shape}")
                print(f"[img2imgStreamDiffusion.py] Now applying temporal interpolation: {prompt_travel_factor:.3f}")

            else:
                # SINGLE-FILE MODE: Standard prompt travel
                source_prompt = params.prompt
                target_prompt = getattr(params, 'target_prompt', params.prompt)

                print(f"[img2imgStreamDiffusion.py] === SINGLE-FILE PROMPT TRAVEL ===")
                print(f"[img2imgStreamDiffusion.py] source: {source_prompt[:80]}...")
                print(f"[img2imgStreamDiffusion.py] target: {target_prompt[:80]}...")
                print(f"[img2imgStreamDiffusion.py] factor: {prompt_travel_factor}")

                # Get or compute source embeddings (with caching)
                cache = self.prompt_embeds_cache[pipe_index]
                if source_prompt not in cache:
                    print(f"[img2imgStreamDiffusion.py] Cache MISS - encoding source prompt")
                    source_embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                        prompt=source_prompt,
                        device=stream_wrapper.stream.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    cache[source_prompt] = source_embeds
                else:
                    print(f"[img2imgStreamDiffusion.py] Cache HIT - reusing source prompt")
                    source_embeds = cache[source_prompt]

                print(f"[img2imgStreamDiffusion.py] Source embeddings shape: {source_embeds.shape}")

                # Get or compute target embeddings (with caching)
                if target_prompt not in cache:
                    print(f"[img2imgStreamDiffusion.py] Cache MISS - encoding target prompt")
                    target_embeds, _ = stream_wrapper.prompt_travel.encode_prompt(
                        prompt=target_prompt,
                        device=stream_wrapper.stream.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    cache[target_prompt] = target_embeds
                else:
                    print(f"[img2imgStreamDiffusion.py] Cache HIT - reusing target prompt")
                    target_embeds = cache[target_prompt]

                print(f"[img2imgStreamDiffusion.py] Target embeddings shape: {target_embeds.shape}")

            # Interpolate between embeddings
            interpolated_embeds = stream_wrapper.prompt_travel.interpolate_embeddings(
                embeds_from=source_embeds,
                embeds_to=target_embeds,
                factor=prompt_travel_factor,
            )

            print(f"[img2imgStreamDiffusion.py] Interpolated embeddings shape: {interpolated_embeds.shape}")

            # StreamDiffusion with denoising batch needs embeddings repeated for batch_size
            # batch_size = len(t_index_list) when use_denoising_batch=True
            batch_size = stream_wrapper.stream.batch_size
            if interpolated_embeds.shape[0] != batch_size:
                print(f"[img2imgStreamDiffusion.py] Repeating embeddings from batch_size {interpolated_embeds.shape[0]} to {batch_size}")
                interpolated_embeds = interpolated_embeds.repeat(batch_size, 1, 1)

            # Directly set the embeddings on the inner stream object
            stream_wrapper.stream.prompt_embeds = interpolated_embeds

            print(f"[img2imgStreamDiffusion.py] Set prompt_embeds with shape: {interpolated_embeds.shape}")

        else:
            # If prompt changed and not using prompt travel, update it via prepare()
            prompt = params.prompt
            if prompt != self.last_prompt:
                stream_wrapper.prepare(
                    prompt=prompt,
                    negative_prompt=default_negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=1.0,
                )
                self.last_prompt = prompt

            print(f"[img2imgStreamDiffusion.py] NOTE: No prompt travel used, prepared prompt: {prompt}")

        # Update ControlNet control image (use input image for structural guidance)
        # ControlNet is statically enabled for this pipeline
        control_image = getattr(params, 'control_image', params.image)
        if control_image is not None:
            print(f"[img2imgStreamDiffusion.py] Updating control image for ControlNet structural guidance")
            stream_wrapper.update_control_image(index=0, image=control_image)

        # Update ControlNet conditioning scale from params (runtime adjustable)
        if hasattr(params, 'controlnet_scale') and hasattr(stream_wrapper.stream, '_controlnet_module'):
            if params.controlnet_scale != self.last_controlnet_scale:
                stream_wrapper.stream._controlnet_module.update_controlnet_scale(index=0, scale=params.controlnet_scale)
                self.last_controlnet_scale = params.controlnet_scale
                print(f"[img2imgStreamDiffusion.py] Updated ControlNet scale to {params.controlnet_scale}")

        # Update temporal coherence if provided (runtime adjustable)
        temporal_coherence = getattr(params, 'temporal_coherence', None)
        temporal_coherence_latent = getattr(params, 'temporal_coherence_latent', None)
        if temporal_coherence is not None or temporal_coherence_latent is not None:
            stream_wrapper.update_stream_params(
                temporal_coherence=temporal_coherence,
                temporal_coherence_latent=temporal_coherence_latent
            )

        # Preprocess input image and generate
        image_tensor = stream_wrapper.preprocess_image(params.image)
        output_image = stream_wrapper(image=image_tensor)

        # Debug controlnet: paste preprocessed control image in bottom-right corner
        if params.debug_controlnet:
            # Get the preprocessed control image (depth map) that ControlNet is using
            preprocessed_control = stream_wrapper.get_last_processed_image(index=0)

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