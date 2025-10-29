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
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

import torch
from safetensors.torch import load_file

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

# Use SDXL Lightning 1-step UNet from local safetensors
# base_model = "stabilityai/stable-diffusion-xl-base-1.0"
unet_safetensors_path = None #"models/diffusion/sdxl_lightning_1step_unet_x0.safetensors"
base_model = "stabilityai/sdxl-turbo"
# base_model = "stabilityai/sd-turbo"
# base_model = "stabilityai/stable-diffusion-2-1-base"
# base_model = "KBlueLeaf/kohaku-v2.1"
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesdxl"

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
        name: str = "img2imgStreamDiffusionXL"
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

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):

        # Enable PyTorch optimizations for inference performance
        torch.backends.cudnn.benchmark = True  # Optimize convolution algorithms (fixed input size)
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 on Ampere+ GPUs
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations

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
            'model_id': 'diffusers/controlnet-depth-sdxl-1.0',  # SDXL depth ControlNet
            'preprocessor': 'passthrough',  # Uncomment to calculate depth from RGB
            # 'preprocessor_params': {
            #     'model_name': 'Intel/dpt-swinv2-tiny-256',  # ~165MB, fastest
            #     # 'model_name': 'Intel/dpt-large',  # ~1.3GB, slower but higher quality
            # },
            'conditioning_scale': 0.72,
            'enabled': True,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        print(f"[img2imgStreamDiffusion.py] ControlNet enabled with SDXL model: {controlnet_config['model_id']}")

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
            print(f"[img2imgStreamDiffusion.py] Image postprocessing configured with RealESRGAN {self.upscaler_scale_factor}x upscaler")

        # OPTIMIZATION: Create ONE shared wrapper, build separate UNet engines for each LoRA combo
        # Only pipe 0's UNet will be loaded in VRAM initially
        # Architecture:
        #   - self.shared_wrapper: Single StreamDiffusionWrapper with shared components
        #   - self.unet_engines: List of UNet TensorRT engines (one per pipe)
        #   - self.pipes: List of pipe metadata (for compatibility, references shared_wrapper)

        self.shared_wrapper = None
        self.unet_engines = []  # Store UNet TensorRT engines for each pipe

        # Create first pipe normally to establish shared base
        print(f"[img2imgStreamDiffusionXL.py] Creating shared base wrapper")
        self.shared_wrapper = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            lora_dict=None,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[18],
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
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            image_postprocessing_config=image_postprocessing_config,
        )

        # Now build a unique UNet for each adapter weights set
        for idx, adapter_weights in enumerate(adapter_weights_sets):
            print(f"[img2imgStreamDiffusionXL.py] Building UNet {idx + 1}/{len(adapter_weights_sets)}")
            print(f"[img2imgStreamDiffusionXL.py] UNet {idx}: adapter_weights = {adapter_weights}")

            # For idx==0, use the existing loaded UNet in shared_wrapper
            # For idx>0, we need to load a fresh PyTorch UNet, apply different LoRAs,
            # build TensorRT, then unload it

            if idx == 0:
                # Use the UNet that's already in shared_wrapper
                print(f"[img2imgStreamDiffusionXL.py] Using UNet from shared_wrapper for pipe 0")
                target_pipe = self.shared_wrapper.stream.pipe
            else:
                # Load a fresh PyTorch UNet for LoRA fusion ON CPU
                # This avoids GPU OOM since pipe 0's TensorRT UNet is already loaded
                print(f"[img2imgStreamDiffusionXL.py] Loading fresh PyTorch UNet to CPU for pipe {idx}")

                # Load fresh UNet from base model TO CPU
                fresh_unet = UNet2DConditionModel.from_pretrained(
                    base_model,
                    subfolder="unet",
                    torch_dtype=torch_dtype,
                    device_map="cpu"  # Load directly to CPU
                )
                print(f"[img2imgStreamDiffusionXL.py] Loaded fresh UNet to CPU")

                # Temporarily replace the shared wrapper's UNet for LoRA loading
                target_pipe = self.shared_wrapper.stream.pipe
                original_unet = target_pipe.unet
                target_pipe.unet = fresh_unet
                print(f"[img2imgStreamDiffusionXL.py] Temporarily using CPU UNet for LoRA fusion")

            # Load LoRAs with this pipe's adapter weights
            if lora_config is not None:
                curation_key = lora_config.get_curation_keys()[0]
                lora_models_list = lora_config.get_lora_curation()[curation_key]
                lora_models_dict = lora_config.get_lora_models()

                # Filter out "None" entries
                selected_loras = [name for name in lora_models_list if name != "None"]

                if selected_loras:
                    print(f"[img2imgStreamDiffusion.py] Loading {len(selected_loras)} LoRAs: {selected_loras}")

                    # Ensure adapter_weights matches the number of loras
                    if len(adapter_weights) < len(selected_loras):
                        adapter_weights = adapter_weights + [1.0] * (len(selected_loras) - len(adapter_weights))
                    elif len(adapter_weights) > len(selected_loras):
                        adapter_weights = adapter_weights[:len(selected_loras)]

                    # Load each LoRA with an adapter name
                    adapter_names = []
                    for i, lora_name in enumerate(selected_loras):
                        adapter_name = f"lora_{i}"
                        lora_path = lora_models_dict[lora_name]
                        print(f"[img2imgStreamDiffusionXL.py] Loading LoRA {i}: {lora_name} as {adapter_name} with weight {adapter_weights[i]}")
                        target_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                        adapter_names.append(adapter_name)

                    # Set adapter weights and fuse
                    print(f"[img2imgStreamDiffusionXL.py] Setting adapters with weights: {adapter_weights}")
                    target_pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)

                    print(f"[img2imgStreamDiffusionXL.py] Fusing LoRAs with scale 1.0")
                    target_pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)

                    # Unload after fusing to free memory
                    target_pipe.unload_lora_weights()
                    print(f"[img2imgStreamDiffusionXL.py] LoRAs loaded and fused successfully")

            # Now build TensorRT engine for this LoRA-fused UNet
            # For idx==0, keep it loaded. For idx>0, build and unload
            print(f"[img2imgStreamDiffusionXL.py] Building TensorRT engine for UNet {idx}...")

            # TODO: Implement TensorRT building here
            # For now, we'll just store the UNet engine reference

            if idx == 0:
                # Keep pipe 0's UNet loaded in GPU - it's already in shared_wrapper
                print(f"[img2imgStreamDiffusionXL.py] UNet {idx} will remain loaded in GPU (shared_wrapper)")
                self.unet_engines.append(self.shared_wrapper.stream.unet)
            else:
                # For pipe 1+: Store the fused PyTorch UNet (already on CPU)
                print(f"[img2imgStreamDiffusionXL.py] Storing UNet {idx} (already on CPU)")

                # Store the fused UNet (it's already on CPU)
                self.unet_engines.append(target_pipe.unet)

                # Restore the original UNet to shared_wrapper
                target_pipe.unet = original_unet
                print(f"[img2imgStreamDiffusionXL.py] Restored original UNet to shared_wrapper")

                # Clean up
                del fresh_unet
                import gc
                gc.collect()
                print(f"[img2imgStreamDiffusionXL.py] UNet {idx} stored on CPU")

        # Clean up PyTorch UNet/VAE from shared_wrapper now that LoRAs are fused
        print(f"[img2imgStreamDiffusionXL.py] Cleaning up PyTorch models after TensorRT load...")
        self.shared_wrapper.cleanup_pytorch_models_after_tensorrt()

        # Prepare the shared wrapper with default prompt
        self.shared_wrapper.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.0,
        )

        # Initialize PromptTravel for shared wrapper
        self.shared_wrapper.prompt_travel = PromptTravel(
            text_encoder=self.shared_wrapper.stream.pipe.text_encoder,
            tokenizer=self.shared_wrapper.stream.pipe.tokenizer,
            text_encoder_2=self.shared_wrapper.stream.pipe.text_encoder_2,
            tokenizer_2=self.shared_wrapper.stream.pipe.tokenizer_2,
        )

        # For compatibility, make self.pipes reference the shared wrapper multiple times
        self.pipes = [self.shared_wrapper] * len(adapter_weights_sets)
        print(f"[img2imgStreamDiffusionXL.py] Created {len(self.pipes)} pipe references (all share same wrapper)")
        print(f"[img2imgStreamDiffusionXL.py] Built {len(self.unet_engines)} UNet engines")


        # Store current pipe index
        self.current_pipe_idx = 0
        self.last_prompt = default_prompt

        # Cache for prompt travel embeddings (per pipe)
        self.prompt_embeds_cache = {}  # {pipe_idx: {prompt: (embeds, pooled)}}

        # Initialize cache for each pipe
        for idx in range(len(self.pipes)):
            self.prompt_embeds_cache[idx] = {}

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Get pipe_index from params, default to 0 if not provided
        pipe_index = getattr(params, 'pipe_index', 0)
        print(f"[img2imgStreamDiffusionXL.py] USING PIPE INDEX: {pipe_index}")

        # Ensure pipe_index is within bounds
        if pipe_index >= len(self.pipes):
            print(f"[img2imgStreamDiffusionXL.py] Warning: pipe_index {pipe_index} out of bounds, using 0")
            pipe_index = 0

        # Swap UNet if pipe index changed
        if pipe_index != self.current_pipe_idx:
            print(f"[img2imgStreamDiffusionXL.py] Swapping UNet from pipe {self.current_pipe_idx} to pipe {pipe_index}")
            # TODO: Implement actual UNet swapping
            # For now, this is a placeholder since all pipes share the same wrapper
            self.current_pipe_idx = pipe_index

        # Use the shared wrapper (all pipes reference it)
        stream_wrapper = self.shared_wrapper

        # Generate image from input image and prompt
        print(f"[img2imgStreamDiffusion.py] Params: {params}")

        # Handle prompt travel if enabled
        use_prompt_travel = getattr(params, "use_prompt_travel", False)
        print(f"[img2imgStreamDiffusion.py] use_prompt_travel: {use_prompt_travel}")

        if use_prompt_travel:
            # Get prompts
            source_prompt = params.prompt
            target_prompt = getattr(params, 'target_prompt', params.prompt)
            prompt_travel_factor = getattr(params, 'prompt_travel_factor', 0.5)

            print(f"[img2imgStreamDiffusionXL.py] === PROMPT TRAVEL DEBUG ===")
            print(f"[img2imgStreamDiffusionXL.py] factor: {prompt_travel_factor:.3f}")
            print(f"[img2imgStreamDiffusionXL.py] source: {source_prompt[:80]}...")
            print(f"[img2imgStreamDiffusionXL.py] target: {target_prompt[:80]}...")
            print(f"[img2imgStreamDiffusionXL.py] Interpolation: lerp(source, target, {prompt_travel_factor:.3f})")
            print(f"[img2imgStreamDiffusionXL.py]   -> factor=0.0 gives 100% source")
            print(f"[img2imgStreamDiffusionXL.py]   -> factor=1.0 gives 100% target")
            print(f"[img2imgStreamDiffusionXL.py] ==========================")

            # Get or compute source embeddings (with caching)
            cache = self.prompt_embeds_cache[pipe_index]
            if source_prompt not in cache:
                print(f"[img2imgStreamDiffusionXL.py] Cache MISS - encoding source prompt")
                source_embeds, _, source_pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                    prompt=source_prompt,
                    device=stream_wrapper.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                cache[source_prompt] = (source_embeds, source_pooled)
            else:
                print(f"[img2imgStreamDiffusionXL.py] Cache HIT - reusing source embeddings")
                source_embeds, source_pooled = cache[source_prompt]

            # Get or compute target embeddings (with caching)
            if target_prompt not in cache:
                print(f"[img2imgStreamDiffusionXL.py] Cache MISS - encoding target prompt")
                target_embeds, _, target_pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                    prompt=target_prompt,
                    device=stream_wrapper.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                cache[target_prompt] = (target_embeds, target_pooled)
            else:
                print(f"[img2imgStreamDiffusionXL.py] Cache HIT - reusing target embeddings")
                target_embeds, target_pooled = cache[target_prompt]

            # Interpolate between embeddings (both concatenated and pooled)
            interpolated_embeds, interpolated_pooled = stream_wrapper.prompt_travel.interpolate_embeddings_sdxl(
                embeds_from=(source_embeds, source_pooled),
                embeds_to=(target_embeds, target_pooled),
                factor=prompt_travel_factor,
            )

            print(f"[img2imgStreamDiffusionXL.py] Interpolated embeddings shape: {interpolated_embeds.shape}")
            print(f"[img2imgStreamDiffusionXL.py] Interpolated pooled embeddings shape: {interpolated_pooled.shape}")

            # StreamDiffusion repeats embeddings for batch_size, so we need to match that
            batch_size = stream_wrapper.stream.batch_size
            interpolated_embeds_batched = interpolated_embeds.repeat(batch_size, 1, 1)

            # Directly set both the concatenated embeddings and pooled embeddings
            stream_wrapper.stream.prompt_embeds = interpolated_embeds_batched
            stream_wrapper.stream.add_text_embeds = interpolated_pooled  # SDXL pooled embeddings

            print(f"[img2imgStreamDiffusionXL.py] Set prompt_embeds with shape: {interpolated_embeds_batched.shape}")
            print(f"[img2imgStreamDiffusionXL.py] Set add_text_embeds with shape: {interpolated_pooled.shape}")

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