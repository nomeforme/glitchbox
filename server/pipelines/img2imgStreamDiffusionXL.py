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
        use_latent_travel: bool = Field(
            True,
            title="Use Latent Travel",
            field="checkbox",
            id="use_latent_travel",
            hide=True,
        )
        latent_travel_method: str = Field(
            "slerp",
            title="Latent Travel Method",
            field="select",
            id="latent_travel_method",
            options=["slerp", "linear"],
            hide=True,
        )
        latent_travel_factor: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Latent Travel Factor",
            field="range",
            id="latent_travel_factor",
            hide=True,
        )
        seed: int = Field(
            4402026899276587, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        target_seed: int | None = Field(
            None, min=0, title="Target Seed", field="seed", hide=True, id="target_seed"
        )
        width: int = Field(
            1024, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            768, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        controlnet_scale: float = Field(
            0.55,
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
        prompt_index_interpolation_duration: float = Field(
            0.5,
            min=0.0,
            max=5.0,
            step=0.1,
            title="Prompt Index Interpolation Duration (s)",
            field="range",
            hide=True,
            id="prompt_index_interpolation_duration",
            description="Duration in seconds to interpolate between prompts when pipe index changes",
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

        # Check if TensorRT acceleration is enabled
        self.use_tensorrt = getattr(args, 'tensorrt', False)
        if self.use_tensorrt:
            print(f"[img2imgStreamDiffusionXL.py] TensorRT acceleration ENABLED")
        else:
            print(f"[img2imgStreamDiffusionXL.py] TensorRT acceleration DISABLED (use --tensorrt to enable)")

        # Check if torch.compile acceleration is enabled
        self.use_torch_compile = getattr(args, 'torch_compile', False)
        self.use_regional_compile = getattr(args, 'torch_compile_regional', True)
        if self.use_torch_compile:
            compile_mode = "regional (faster cold start)" if self.use_regional_compile else "full model"
            print(f"[img2imgStreamDiffusionXL.py] torch.compile acceleration ENABLED ({compile_mode})")
        else:
            print(f"[img2imgStreamDiffusionXL.py] torch.compile DISABLED (use --torch-compile to enable)")

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
            # 'preprocessor': 'depth',  # Uncomment to calculate depth from RGB
            # 'preprocessor_params': {
            #     'model_name': 'Intel/dpt-swinv2-tiny-256',  # ~165MB, fastest
            #     # 'model_name': 'Intel/dpt-large',  # ~1.3GB, slower but higher quality
            # },
            'conditioning_scale': 0.55,  # Default scale - can be adjusted at runtime via params.controlnet_scale
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
            t_index_list=[6],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration="none",
            mode="txt2img",
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

                        # Resolve path relative to server directory if not absolute
                        if not os.path.isabs(lora_path):
                            # Get server directory (parent of pipelines directory)
                            server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            lora_path = os.path.join(server_dir, lora_path)

                        # Handle directory-based LoRA paths
                        actual_lora_path = lora_path
                        if os.path.isdir(lora_path):
                            # Look for LoRA safetensors file in directory
                            lora_files = glob.glob(os.path.join(lora_path, "*_lora.safetensors"))
                            if not lora_files:
                                # Fallback: look for any safetensors that's not embeddings
                                lora_files = [f for f in glob.glob(os.path.join(lora_path, "*.safetensors"))
                                            if "embedding" not in f.lower()]
                            if lora_files:
                                actual_lora_path = lora_files[0]
                                print(f"[img2imgStreamDiffusionXL.py] Found LoRA file in directory: {actual_lora_path}")
                            else:
                                print(f"[img2imgStreamDiffusionXL.py] WARNING: No LoRA file found in directory: {lora_path}")

                        print(f"[img2imgStreamDiffusionXL.py] Loading LoRA {i}: {lora_name} as {adapter_name} with weight {adapter_weights[i]}")
                        target_pipe.load_lora_weights(actual_lora_path, adapter_name=adapter_name)
                        adapter_names.append(adapter_name)

                    # Set adapter weights and fuse
                    print(f"[img2imgStreamDiffusionXL.py] Setting adapters with weights: {adapter_weights}")
                    target_pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)

                    # Get lora_scale from config (defaults to 1.0)
                    lora_scale = lora_config.DEFAULT_LORA_SCALE if lora_config is not None else 1.0
                    print(f"[img2imgStreamDiffusionXL.py] Fusing LoRAs with scale {lora_scale}")
                    target_pipe.fuse_lora(adapter_names=adapter_names, lora_scale=lora_scale)

                    # Unload after fusing to free memory
                    target_pipe.unload_lora_weights()
                    print(f"[img2imgStreamDiffusionXL.py] LoRAs loaded and fused successfully")

                    # Check for textual inversion embeddings alongside LoRAs
                    for i, lora_name in enumerate(selected_loras):
                        lora_path = lora_models_dict[lora_name]

                        # Resolve path relative to server directory if not absolute
                        if not os.path.isabs(lora_path):
                            server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                            lora_path = os.path.join(server_dir, lora_path)

                        # Check if lora_path is a directory or file
                        if os.path.isdir(lora_path):
                            # Look for embedding files in the directory
                            embedding_files = glob.glob(os.path.join(lora_path, "*embeddings.safetensors"))
                            special_params_file = os.path.join(lora_path, "special_params.json")
                        else:
                            # Look in the same directory as the LoRA file
                            lora_dir = os.path.dirname(lora_path)
                            lora_basename = os.path.splitext(os.path.basename(lora_path))[0]
                            embedding_files = glob.glob(os.path.join(lora_dir, f"{lora_basename}_embeddings.safetensors"))
                            if not embedding_files:
                                # Also check for generic embedding files
                                embedding_files = glob.glob(os.path.join(lora_dir, "*embeddings.safetensors"))
                            special_params_file = os.path.join(lora_dir, "special_params.json")

                        if embedding_files:
                            embedding_path = embedding_files[0]
                            print(f"[img2imgStreamDiffusionXL.py] Found textual inversion embedding: {embedding_path}")

                            # Try to read token string from special_params.json
                            token_str = "<s0><s1><s2>"  # Default
                            if os.path.exists(special_params_file):
                                try:
                                    import json
                                    with open(special_params_file, 'r') as f:
                                        special_params = json.load(f)
                                        token_str = special_params.get('TOK', token_str)
                                    print(f"[img2imgStreamDiffusionXL.py] Loaded token string from special_params.json: {token_str}")
                                except Exception as e:
                                    print(f"[img2imgStreamDiffusionXL.py] Warning: Could not read special_params.json: {e}")

                            # Load and fuse the embeddings
                            self._load_and_fuse_embeddings(target_pipe, embedding_path, token_str=token_str)
                            print(f"[img2imgStreamDiffusionXL.py] Textual inversion embeddings fused for LoRA: {lora_name}")

            # Now build TensorRT engine for this LoRA-fused UNet (if enabled)
            # For idx==0, build and keep loaded. For idx>0, store PyTorch UNet for now
            print(f"[img2imgStreamDiffusionXL.py] Processing UNet {idx}...")

            if idx == 0:
                if self.use_tensorrt:
                    # Build TensorRT engine for pipe 0's LoRA-fused UNet
                    print(f"[img2imgStreamDiffusionXL.py] Building TensorRT engine for pipe 0...")
                    unet_engine = self.shared_wrapper.build_tensorrt_unet_engine(
                        pipe_identifier=f"pipe_{idx}",
                        replace_stream_unet=True
                    )
                    if unet_engine is not None:
                        self.unet_engines.append(unet_engine)
                        print(f"[img2imgStreamDiffusionXL.py] TensorRT UNet engine built and loaded for pipe 0")
                    else:
                        # Fallback to PyTorch UNet if TensorRT build fails
                        print(f"[img2imgStreamDiffusionXL.py] TensorRT build failed, using PyTorch UNet for pipe 0")
                        self.unet_engines.append(self.shared_wrapper.stream.unet)
                else:
                    # No TensorRT - use PyTorch UNet
                    print(f"[img2imgStreamDiffusionXL.py] Using PyTorch UNet for pipe 0 (TensorRT disabled)")
                    self.unet_engines.append(self.shared_wrapper.stream.unet)
            else:
                # For pipe 1+: Store the fused PyTorch UNet (already on CPU)
                # TensorRT building for multiple pipes requires more complex architecture
                # For now, store PyTorch UNet for hot-swapping
                print(f"[img2imgStreamDiffusionXL.py] Storing PyTorch UNet {idx} (already on CPU)")
                print(f"[img2imgStreamDiffusionXL.py] NOTE: Pipe {idx} uses PyTorch UNet (no TensorRT)")

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

        # Build TensorRT VAE engines (shared across all pipes) if TensorRT is enabled
        if self.use_tensorrt:
            print(f"[img2imgStreamDiffusionXL.py] Building TensorRT VAE engines...")
            vae_engine = self.shared_wrapper.build_tensorrt_vae_engines(replace_stream_vae=True)
            if vae_engine is not None:
                print(f"[img2imgStreamDiffusionXL.py] TensorRT VAE engines built successfully")
            else:
                print(f"[img2imgStreamDiffusionXL.py] TensorRT VAE build failed, using PyTorch VAE")

            # Clean up PyTorch UNet/VAE from shared_wrapper now that TensorRT engines are loaded
            print(f"[img2imgStreamDiffusionXL.py] Cleaning up PyTorch models after TensorRT load...")
            self.shared_wrapper.cleanup_pytorch_models_after_tensorrt()
        elif self.use_torch_compile:
            # Apply torch.compile if TensorRT is not enabled
            print(f"[img2imgStreamDiffusionXL.py] Applying torch.compile to UNet...")
            self._apply_torch_compile()
        else:
            print(f"[img2imgStreamDiffusionXL.py] No acceleration enabled (use --tensorrt or --torch-compile)")

        # Prepare the shared wrapper with default prompt
        default_seed = 4402026899276587
        self.shared_wrapper.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.0,
        )

        # Set the default seed separately
        self.shared_wrapper.update_stream_params(seed=default_seed)
        print(f"[img2imgStreamDiffusionXL.py] Initialized with default seed: {default_seed}")

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
        self.last_controlnet_scale = None
        self.last_seed = None
        self.last_target_seed = None
        self.last_latent_travel_settings = None

        # Cache for prompt travel embeddings (per pipe)
        self.prompt_embeds_cache = {}  # {pipe_idx: {prompt: (embeds, pooled)}}

        # Initialize cache for each pipe
        for idx in range(len(self.pipes)):
            self.prompt_embeds_cache[idx] = {}

    def _load_and_fuse_embeddings(self, pipe, embedding_path, token_str="<s0><s1><s2>"):
        """
        Load textual inversion embeddings and fuse them into text encoders.

        Args:
            pipe: The diffusion pipeline with text_encoder and text_encoder_2
            embedding_path: Path to the .safetensors file containing embeddings
            token_str: The token string (e.g., "<s0><s1><s2>")
        """
        import re

        # Load embedding tensors
        print(f"[img2imgStreamDiffusionXL.py] Loading embeddings from {embedding_path}")
        embedding_data = load_file(embedding_path)

        # Inspect the embedding data
        print(f"[img2imgStreamDiffusionXL.py] Embedding keys: {list(embedding_data.keys())}")
        for key in embedding_data.keys():
            print(f"[img2imgStreamDiffusionXL.py]   {key}: shape={embedding_data[key].shape}, dtype={embedding_data[key].dtype}")

        # Parse tokens from token_str
        # Expecting format like "<s0><s1><s2>"
        tokens = re.findall(r'<s\d+>', token_str)
        print(f"[img2imgStreamDiffusionXL.py] Parsed tokens: {tokens}")

        num_tokens = len(tokens)

        # Verify embedding shapes match expected number of tokens
        clip_l_embeds = embedding_data.get('clip_l')
        clip_g_embeds = embedding_data.get('clip_g')

        if clip_l_embeds is None or clip_g_embeds is None:
            print(f"[img2imgStreamDiffusionXL.py] ERROR: Missing clip_l or clip_g embeddings!")
            return

        if clip_l_embeds.shape[0] != num_tokens or clip_g_embeds.shape[0] != num_tokens:
            print(f"[img2imgStreamDiffusionXL.py] WARNING: Embedding shape mismatch!")
            print(f"[img2imgStreamDiffusionXL.py]   Expected {num_tokens} tokens, got clip_l={clip_l_embeds.shape[0]}, clip_g={clip_g_embeds.shape[0]}")

        # Add tokens to tokenizers and resize embeddings
        # Text Encoder 1 (CLIP-L)
        tokenizer_1 = pipe.tokenizer
        text_encoder_1 = pipe.text_encoder

        # Text Encoder 2 (CLIP-G)
        tokenizer_2 = pipe.tokenizer_2
        text_encoder_2 = pipe.text_encoder_2

        print(f"[img2imgStreamDiffusionXL.py] Adding {num_tokens} tokens to tokenizers")

        # Add tokens to tokenizer 1 (CLIP-L)
        num_added_tokens_1 = tokenizer_1.add_tokens(tokens)
        print(f"[img2imgStreamDiffusionXL.py] Added {num_added_tokens_1} new tokens to tokenizer_1")

        # Resize token embeddings for text_encoder_1
        text_encoder_1.resize_token_embeddings(len(tokenizer_1))

        # Get token IDs and set embeddings
        token_ids_1 = tokenizer_1.convert_tokens_to_ids(tokens)
        print(f"[img2imgStreamDiffusionXL.py] Token IDs in tokenizer_1: {token_ids_1}")

        # Set the embedding weights for each token in text_encoder_1
        with torch.no_grad():
            for i, token_id in enumerate(token_ids_1):
                text_encoder_1.get_input_embeddings().weight[token_id] = clip_l_embeds[i].to(
                    device=text_encoder_1.device,
                    dtype=text_encoder_1.dtype
                )
        print(f"[img2imgStreamDiffusionXL.py] Set {len(token_ids_1)} embeddings in text_encoder_1")

        # Add tokens to tokenizer 2 (CLIP-G)
        num_added_tokens_2 = tokenizer_2.add_tokens(tokens)
        print(f"[img2imgStreamDiffusionXL.py] Added {num_added_tokens_2} new tokens to tokenizer_2")

        # Resize token embeddings for text_encoder_2
        text_encoder_2.resize_token_embeddings(len(tokenizer_2))

        # Get token IDs and set embeddings
        token_ids_2 = tokenizer_2.convert_tokens_to_ids(tokens)
        print(f"[img2imgStreamDiffusionXL.py] Token IDs in tokenizer_2: {token_ids_2}")

        # Set the embedding weights for each token in text_encoder_2
        with torch.no_grad():
            for i, token_id in enumerate(token_ids_2):
                text_encoder_2.get_input_embeddings().weight[token_id] = clip_g_embeds[i].to(
                    device=text_encoder_2.device,
                    dtype=text_encoder_2.dtype
                )
        print(f"[img2imgStreamDiffusionXL.py] Set {len(token_ids_2)} embeddings in text_encoder_2")

        print(f"[img2imgStreamDiffusionXL.py] Successfully fused textual inversion embeddings!")
        print(f"[img2imgStreamDiffusionXL.py] You can now use tokens: {token_str} in your prompts")

    def _apply_torch_compile(self):
        """
        Apply torch.compile to the UNet for acceleration.

        Supports two modes:
        - Regional compilation: Compiles repeated transformer blocks individually.
          This drastically reduces cold start time (7-8x faster) while maintaining
          similar runtime speedup as full compilation.
        - Full compilation: Compiles the entire UNet as one graph.
          Slightly faster at runtime but much longer cold start.
        """
        import torch

        unet = self.shared_wrapper.stream.unet

        # Check if UNet is a TensorRT engine (shouldn't compile TRT engines)
        if 'Engine' in type(unet).__name__:
            print(f"[img2imgStreamDiffusionXL.py] UNet is TensorRT engine, skipping torch.compile")
            return

        # Enable TF32 for better performance on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')

        if self.use_regional_compile:
            # Regional compilation: compile repeated transformer blocks
            # This reduces cold start from ~60s to ~8s while maintaining speedup
            print(f"[img2imgStreamDiffusionXL.py] Applying REGIONAL torch.compile to transformer blocks...")

            compiled_blocks = 0

            # Compile down_blocks (repeated encoder blocks)
            if hasattr(unet, 'down_blocks'):
                for i, block in enumerate(unet.down_blocks):
                    if hasattr(block, 'attentions') and block.attentions is not None:
                        for j, attn in enumerate(block.attentions):
                            unet.down_blocks[i].attentions[j] = torch.compile(
                                attn,
                                mode="reduce-overhead",
                                fullgraph=True,
                                dynamic=True
                            )
                            compiled_blocks += 1
                    if hasattr(block, 'resnets'):
                        for j, resnet in enumerate(block.resnets):
                            unet.down_blocks[i].resnets[j] = torch.compile(
                                resnet,
                                mode="reduce-overhead",
                                fullgraph=True,
                                dynamic=True
                            )
                            compiled_blocks += 1

            # Compile mid_block
            if hasattr(unet, 'mid_block') and unet.mid_block is not None:
                if hasattr(unet.mid_block, 'attentions'):
                    for i, attn in enumerate(unet.mid_block.attentions):
                        unet.mid_block.attentions[i] = torch.compile(
                            attn,
                            mode="reduce-overhead",
                            fullgraph=True,
                            dynamic=True
                        )
                        compiled_blocks += 1
                if hasattr(unet.mid_block, 'resnets'):
                    for i, resnet in enumerate(unet.mid_block.resnets):
                        unet.mid_block.resnets[i] = torch.compile(
                            resnet,
                            mode="reduce-overhead",
                            fullgraph=True,
                            dynamic=True
                        )
                        compiled_blocks += 1

            # Compile up_blocks (repeated decoder blocks)
            if hasattr(unet, 'up_blocks'):
                for i, block in enumerate(unet.up_blocks):
                    if hasattr(block, 'attentions') and block.attentions is not None:
                        for j, attn in enumerate(block.attentions):
                            unet.up_blocks[i].attentions[j] = torch.compile(
                                attn,
                                mode="reduce-overhead",
                                fullgraph=True,
                                dynamic=True
                            )
                            compiled_blocks += 1
                    if hasattr(block, 'resnets'):
                        for j, resnet in enumerate(block.resnets):
                            unet.up_blocks[i].resnets[j] = torch.compile(
                                resnet,
                                mode="reduce-overhead",
                                fullgraph=True,
                                dynamic=True
                            )
                            compiled_blocks += 1

            print(f"[img2imgStreamDiffusionXL.py] Regional compilation complete: {compiled_blocks} blocks compiled")
            print(f"[img2imgStreamDiffusionXL.py] First inference will trigger JIT compilation (faster than full compile)")

        else:
            # Full model compilation
            print(f"[img2imgStreamDiffusionXL.py] Applying FULL torch.compile to UNet...")
            print(f"[img2imgStreamDiffusionXL.py] WARNING: First inference will have long cold start (~60s+)")

            self.shared_wrapper.stream.unet = torch.compile(
                unet,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True
            )

            print(f"[img2imgStreamDiffusionXL.py] Full UNet compilation complete")

        # Also compile VAE decoder for additional speedup
        vae = self.shared_wrapper.stream.vae
        if 'Engine' not in type(vae).__name__ and hasattr(vae, 'decoder'):
            print(f"[img2imgStreamDiffusionXL.py] Compiling VAE decoder...")
            vae.decoder = torch.compile(
                vae.decoder,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True
            )
            print(f"[img2imgStreamDiffusionXL.py] VAE decoder compilation complete")

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
            # UNet switching happens at TensorRT engine level
            # Each pipe has its own LoRA-fused TensorRT UNet engine
            # Other components (VAE, text encoders, ControlNet) are shared

            # NOTE: UNet swap is disabled - causes OOM with multiple PyTorch UNets
            # Perform the actual UNet swap
            # target_unet = self.unet_engines[pipe_index]
            # self.shared_wrapper.stream.unet = target_unet
            # print(f"[img2imgStreamDiffusionXL.py] Successfully swapped to UNet engine {pipe_index}")
            # print(f"[img2imgStreamDiffusionXL.py] UNet type: {type(target_unet).__name__}")

            self.current_pipe_idx = pipe_index

        # Use the shared wrapper (all pipes reference it)
        stream_wrapper = self.shared_wrapper

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
                # MULTI-FILE MODE: Weighted spatial + temporal blending
                source_prompts = params.prompt  # List of prompts
                target_prompts = getattr(params, 'target_prompt', params.prompt)  # List of prompts
                spatial_weights = getattr(params, 'spatial_weights', None)

                print(f"[img2imgStreamDiffusionXL.py] === MULTI-FILE PROMPT TRAVEL ===")
                print(f"[img2imgStreamDiffusionXL.py] Number of files: {len(source_prompts)}")
                print(f"[img2imgStreamDiffusionXL.py] Spatial weights: {spatial_weights}")
                print(f"[img2imgStreamDiffusionXL.py] Temporal factor: {prompt_travel_factor:.3f}")

                # Encode all prompts and compute weighted spatial blend for source
                cache = self.prompt_embeds_cache[pipe_index]
                source_embeds_list = []
                source_pooled_list = []

                for i, prompt in enumerate(source_prompts):
                    if prompt not in cache:
                        print(f"[img2imgStreamDiffusionXL.py] Cache MISS - encoding source prompt {i}")
                        embeds, _, pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                            prompt=prompt,
                            device=stream_wrapper.stream.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        cache[prompt] = (embeds, pooled)
                    else:
                        embeds, pooled = cache[prompt]
                    source_embeds_list.append(embeds)
                    source_pooled_list.append(pooled)

                # Weighted sum for source (spatial blending)
                source_embeds = sum(w * e for w, e in zip(spatial_weights, source_embeds_list))
                source_pooled = sum(w * p for w, p in zip(spatial_weights, source_pooled_list))

                # Encode all prompts and compute weighted spatial blend for target
                target_embeds_list = []
                target_pooled_list = []

                for i, prompt in enumerate(target_prompts):
                    if prompt not in cache:
                        print(f"[img2imgStreamDiffusionXL.py] Cache MISS - encoding target prompt {i}")
                        embeds, _, pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                            prompt=prompt,
                            device=stream_wrapper.stream.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                        )
                        cache[prompt] = (embeds, pooled)
                    else:
                        embeds, pooled = cache[prompt]
                    target_embeds_list.append(embeds)
                    target_pooled_list.append(pooled)

                # Weighted sum for target (spatial blending)
                target_embeds = sum(w * e for w, e in zip(spatial_weights, target_embeds_list))
                target_pooled = sum(w * p for w, p in zip(spatial_weights, target_pooled_list))

                print(f"[img2imgStreamDiffusionXL.py] Spatially blended source/target embeddings")
                print(f"[img2imgStreamDiffusionXL.py] Now applying temporal interpolation: {prompt_travel_factor:.3f}")

            else:
                # SINGLE-FILE MODE: Original behavior
                source_prompt = params.prompt
                target_prompt = getattr(params, 'target_prompt', params.prompt)

                print(f"[img2imgStreamDiffusionXL.py] === SINGLE-FILE PROMPT TRAVEL ===")
                print(f"[img2imgStreamDiffusionXL.py] factor: {prompt_travel_factor:.3f}")
                print(f"[img2imgStreamDiffusionXL.py] source: {source_prompt[:80]}...")
                print(f"[img2imgStreamDiffusionXL.py] target: {target_prompt[:80]}...")

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

            # Temporal interpolation (LERP) between spatially-blended (or single) source and target
            interpolated_embeds, interpolated_pooled = stream_wrapper.prompt_travel.interpolate_embeddings_sdxl(
                embeds_from=(source_embeds, source_pooled),
                embeds_to=(target_embeds, target_pooled),
                factor=prompt_travel_factor,
            )

            print(f"[img2imgStreamDiffusionXL.py] Final interpolated embeddings shape: {interpolated_embeds.shape}")
            print(f"[img2imgStreamDiffusionXL.py] Final interpolated pooled embeddings shape: {interpolated_pooled.shape}")

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

        # Handle seed and latent travel
        use_latent_travel = getattr(params, 'use_latent_travel', False)
        seed = getattr(params, 'seed', 4402026899276587)
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = seed + 1

        latent_travel_factor = getattr(params, 'latent_travel_factor', 0.5)
        latent_travel_method = getattr(params, 'latent_travel_method', 'slerp')

        # Create settings tuple to check if anything changed
        current_latent_travel_settings = (use_latent_travel, seed, target_seed, latent_travel_factor, latent_travel_method)

        # DEBUG: Always log to see if we're checking seeds every frame
        print(f"[img2imgStreamDiffusionXL.py] Frame seed check - use_latent_travel: {use_latent_travel}, seed: {seed}, factor: {latent_travel_factor:.3f}")
        print(f"[img2imgStreamDiffusionXL.py] Settings changed: {current_latent_travel_settings != self.last_latent_travel_settings}")

        # Check current init_noise to see if it's changing
        if hasattr(stream_wrapper.stream, 'init_noise') and stream_wrapper.stream.init_noise is not None:
            noise_mean = stream_wrapper.stream.init_noise.mean().item()
            noise_std = stream_wrapper.stream.init_noise.std().item()
            print(f"[img2imgStreamDiffusionXL.py] Current init_noise stats - mean: {noise_mean:.6f}, std: {noise_std:.6f}")

        # IMPORTANT: Always update if latent travel is enabled, because the factor might be animating
        # Only skip update if disabled and settings haven't changed
        should_update = use_latent_travel or (current_latent_travel_settings != self.last_latent_travel_settings)

        if should_update:
            print(f"[img2imgStreamDiffusionXL.py] Updating seed/latent travel settings")
            print(f"[img2imgStreamDiffusionXL.py]   use_latent_travel: {use_latent_travel}")
            print(f"[img2imgStreamDiffusionXL.py]   seed: {seed}, target_seed: {target_seed}")
            print(f"[img2imgStreamDiffusionXL.py]   latent_travel_factor: {latent_travel_factor}, method: {latent_travel_method}")

            if use_latent_travel:
                # Use seed blending with StreamDiffusion's built-in seed_list
                # Weight calculation: source gets (1 - factor), target gets factor
                source_weight = 1.0 - latent_travel_factor
                target_weight = latent_travel_factor

                seed_list = [
                    (seed, source_weight),
                    (target_seed, target_weight)
                ]

                print(f"[img2imgStreamDiffusionXL.py] Applying seed blending: seed_list={seed_list}, method={latent_travel_method}")
                print(f"[img2imgStreamDiffusionXL.py] This will blend {source_weight*100:.1f}% of seed {seed} with {target_weight*100:.1f}% of seed {target_seed}")

                # Store noise BEFORE update
                noise_before = stream_wrapper.stream.init_noise.clone() if hasattr(stream_wrapper.stream, 'init_noise') else None

                # Update with seed blending
                stream_wrapper.update_stream_params(
                    seed_list=seed_list,
                    seed_interpolation_method=latent_travel_method
                )

                # Check if noise actually changed
                if noise_before is not None and hasattr(stream_wrapper.stream, 'init_noise'):
                    noise_after = stream_wrapper.stream.init_noise
                    noise_diff = (noise_after - noise_before).abs().mean().item()
                    print(f"[img2imgStreamDiffusionXL.py] Noise changed by: {noise_diff:.6f} (should be >0 if blending worked)")
                    if noise_diff < 1e-6:
                        print(f"[img2imgStreamDiffusionXL.py] WARNING: Noise didn't change! Seed blending may not be working.")
            else:
                # Just use fixed seed without blending
                print(f"[img2imgStreamDiffusionXL.py] Using fixed seed: {seed}")
                stream_wrapper.update_stream_params(seed=seed)

            self.last_latent_travel_settings = current_latent_travel_settings

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
        print(f"[img2imgStreamDiffusionXL.py] About to call stream_wrapper(image=...), type={type(stream_wrapper)}", flush=True)
        output_image = stream_wrapper(image=image_tensor)
        print(f"[img2imgStreamDiffusionXL.py] stream_wrapper() returned, type={type(output_image)}", flush=True)

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