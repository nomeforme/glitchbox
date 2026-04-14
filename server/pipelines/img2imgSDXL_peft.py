"""
StreamDiffusion Image-to-Image SDXL Pipeline with PEFT LoRA Runtime Weight Control

This pipeline loads LoRAs as PEFT adapters on a SINGLE UNet and allows
changing LoRA weights at runtime via set_adapters(). Unlike
img2imgStreamDiffusionXL.py which pre-fuses LoRAs into multiple UNet copies
for real-time switching, this pipeline:

- Loads all LoRAs once as named PEFT adapters (no fusion)
- Uses a single UNet in VRAM
- Changes LoRA weights at runtime via peft set_adapters()
- Accepts lora_weights as a runtime parameter (list of floats)

This is more memory-efficient (no duplicate UNets) at the cost of slightly
slower inference (unfused LoRA adds overhead per forward pass). Ideal for
non-realtime use cases where memory matters more than frame rate.

Design decisions:
- Uses SDXL model: stabilityai/sdxl-turbo
- ControlNet is statically enabled for compositional/structural control in img2img
- LoRAs are loaded as PEFT adapters, NOT fused — weights changeable at runtime
- Single UNet, single pipe — no multi-pipe architecture
"""
import sys
import os

from pipelines.streamdiffusion.wrapper import StreamDiffusionWrapper
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

import torch
from safetensors.torch import load_file

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
import glob
from typing import Optional

from modules.prompt_travel.prompt_travel import PromptTravel

# Function to read prompt prefix from .txt files
def get_prompt_prefix():
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))

    if not prompt_files:
        return ""

    with open(prompt_files[0], 'r') as f:
        return f.read().strip()

prompt_prefix = get_prompt_prefix()
default_prompt = prompt_prefix + "mrnabrmv style, Fragmented digital portrait blending abstract textures and vivid colors, creating a surreal, pixelated visage."
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

unet_safetensors_path = None
base_model = "stabilityai/sdxl-turbo"
taesd_model = "madebyollin/taesdxl"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion PEFT</h1>
<h3 class="text-xl font-bold">Image-to-Image SDXL + ControlNet + Runtime LoRA Weights</h3>
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
    > with ControlNet, and runtime-adjustable LoRA weights via PEFT.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "img2imgSDXL_peft"
        title: str = "Image-to-Image SDXL StreamDiffusion + ControlNet + PEFT LoRA"
        description: str = "Generates an image from an input image using SDXL StreamDiffusion with ControlNet and runtime-adjustable LoRA weights via PEFT"
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
        # LoRA weights as a comma-separated string for UI compatibility
        # Parsed into a list of floats at runtime
        lora_weights: str = Field(
            "1.0,0.0",
            title="LoRA Weights",
            field="textarea",
            id="lora_weights",
            description="Comma-separated LoRA adapter weights (e.g. '1.0,0.0' or '0.5,0.5')",
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
        # Sound-reactive boost factors (kept for compatibility with LoraSoundController)
        boost_factor_bass: float = Field(
            1.0, min=0.0, max=3.0, step=0.1, title="Boost - Bass", field="range", id="boost_factor_bass",
        )
        boost_factor_low_mids: float = Field(
            1.0, min=0.0, max=3.0, step=0.1, title="Boost - Low Mids", field="range", id="boost_factor_low_mids",
        )
        boost_factor_mids: float = Field(
            1.0, min=0.0, max=3.0, step=0.1, title="Boost - Mids", field="range", id="boost_factor_mids",
        )
        boost_factor_high_mids: float = Field(
            1.0, min=0.0, max=3.0, step=0.1, title="Boost - High Mids", field="range", id="boost_factor_high_mids",
        )
        boost_factor_treble: float = Field(
            1.0, min=0.0, max=3.0, step=0.1, title="Boost - Treble", field="range", id="boost_factor_treble",
        )
        controlnet_start: float = Field(
            0.0, min=0, max=1.0, step=0.001, title="Controlnet Start", field="range", hide=True, id="controlnet_start",
        )
        controlnet_end: float = Field(
            1.0, min=0, max=1.0, step=0.001, title="Controlnet End", field="range", hide=True, id="controlnet_end",
        )
        upscaler_scale_factor: int = Field(
            2, min=2, max=4, step=2, title="Upscaler Scale", field="range", hide=True, id="upscaler_scale_factor",
        )
        use_output_bg_removal: bool = Field(
            False, title="Use Output Background Removal", field="checkbox", id="use_output_bg_removal",
        )
        use_prompt_indexing: bool = Field(
            False,
            title="Use Prompt Indexing",
            field="checkbox",
            id="use_prompt_indexing",
            description="Use pipe index to select prompts from file instead of sequential scheduling",
        )
        prompt_index_interpolation_duration: float = Field(
            0.5, min=0.0, max=5.0, step=0.1,
            title="Prompt Index Interpolation Duration (s)",
            field="range", hide=True, id="prompt_index_interpolation_duration",
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
            False, title="Debug ControlNet", field="checkbox", hide=True, id="debug_controlnet",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype, lora_config=None):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.lora_config = lora_config
        self._verbose = getattr(args, 'debug', False)

        # torch.compile support (TensorRT not supported with unfused LoRAs)
        self.use_torch_compile = getattr(args, 'torch_compile', False)
        self.use_regional_compile = getattr(args, 'torch_compile_regional', True)
        if self.use_torch_compile:
            compile_mode = "regional (faster cold start)" if self.use_regional_compile else "full model"
            print(f"[img2imgSDXL_peft.py] torch.compile acceleration ENABLED ({compile_mode})")
            print(f"[img2imgSDXL_peft.py] NOTE: torch.compile may need to recompile when LoRA weights change")
        else:
            print(f"[img2imgSDXL_peft.py] torch.compile DISABLED (use --torch-compile to enable)")

        # Upscaler
        self.use_upscaler = getattr(args, 'use_upscaler', False)
        self.upscaler_scale_factor = getattr(args, 'upscaler_scale_factor', 2)
        if self.use_upscaler:
            print(f"[img2imgSDXL_peft.py] RealESRGAN {self.upscaler_scale_factor}x upscaler enabled")

        params = self.InputParams()

        # ControlNet configuration (disable with --no-controlnet to save ~2.5GB VRAM)
        use_controlnet = not getattr(args, 'no_controlnet', False)
        controlnet_config = None
        if use_controlnet:
            controlnet_config = {
                'model_id': 'diffusers/controlnet-depth-sdxl-1.0',
                'preprocessor': 'passthrough',
                'conditioning_scale': 0.55,
                'enabled': True,
                'control_guidance_start': 0.0,
                'control_guidance_end': 1.0,
            }
            print(f"[img2imgSDXL_peft.py] ControlNet enabled with SDXL model: {controlnet_config['model_id']}")
        else:
            print(f"[img2imgSDXL_peft.py] ControlNet DISABLED (--no-controlnet)")

        # Image postprocessing configuration
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

        # Create single wrapper — no multi-UNet architecture needed
        print(f"[img2imgSDXL_peft.py] Creating single wrapper with PEFT LoRA support")
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
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=args.safety_checker,
            use_controlnet=use_controlnet,
            controlnet_config=controlnet_config,
            image_postprocessing_config=image_postprocessing_config,
        )

        # Load LoRAs as PEFT adapters (NOT fused — weights adjustable at runtime)
        self.adapter_names = []
        self.current_lora_weights = []

        if lora_config is not None:
            curation_key = lora_config.get_curation_keys()[0]
            lora_models_list = lora_config.get_lora_curation()[curation_key]
            lora_models_dict = lora_config.get_lora_models()

            selected_loras = [name for name in lora_models_list if name != "None"]

            if selected_loras:
                print(f"[img2imgSDXL_peft.py] Loading {len(selected_loras)} LoRAs as PEFT adapters: {selected_loras}")
                target_pipe = self.shared_wrapper.stream.pipe

                for i, lora_name in enumerate(selected_loras):
                    adapter_name = f"lora_{i}"
                    lora_path = lora_models_dict[lora_name]

                    # Resolve path
                    if not os.path.isabs(lora_path):
                        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        lora_path = os.path.join(server_dir, lora_path)

                    # Handle directory-based LoRA paths
                    actual_lora_path = lora_path
                    if os.path.isdir(lora_path):
                        lora_files = glob.glob(os.path.join(lora_path, "*_lora.safetensors"))
                        if not lora_files:
                            lora_files = [f for f in glob.glob(os.path.join(lora_path, "*.safetensors"))
                                        if "embedding" not in f.lower()]
                        if lora_files:
                            actual_lora_path = lora_files[0]
                            print(f"[img2imgSDXL_peft.py] Found LoRA file in directory: {actual_lora_path}")
                        else:
                            print(f"[img2imgSDXL_peft.py] WARNING: No LoRA file found in directory: {lora_path}")

                    print(f"[img2imgSDXL_peft.py] Loading LoRA {i}: {lora_name} as adapter '{adapter_name}'")
                    target_pipe.load_lora_weights(actual_lora_path, adapter_name=adapter_name)
                    self.adapter_names.append(adapter_name)

                # Set initial weights from first adapter_weights_set
                adapter_weights_sets = lora_config.get_default_adapter_weights()
                initial_weights = adapter_weights_sets[0] if adapter_weights_sets else [1.0] * len(selected_loras)

                # Ensure weights match number of adapters
                if len(initial_weights) < len(self.adapter_names):
                    initial_weights = initial_weights + [0.0] * (len(self.adapter_names) - len(initial_weights))
                elif len(initial_weights) > len(self.adapter_names):
                    initial_weights = initial_weights[:len(self.adapter_names)]

                self.current_lora_weights = list(initial_weights)
                lora_scale = lora_config.DEFAULT_LORA_SCALE if lora_config is not None else 1.0
                # Apply lora_scale to the weights
                scaled_weights = [w * lora_scale for w in self.current_lora_weights]
                target_pipe.set_adapters(adapter_names=self.adapter_names, adapter_weights=scaled_weights)
                print(f"[img2imgSDXL_peft.py] Initial adapter weights set: {self.current_lora_weights} (scale={lora_scale})")
                print(f"[img2imgSDXL_peft.py] LoRAs loaded as PEFT adapters — weights adjustable at runtime")

                # Check for textual inversion embeddings alongside LoRAs
                for i, lora_name in enumerate(selected_loras):
                    lora_path = lora_models_dict[lora_name]
                    if not os.path.isabs(lora_path):
                        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        lora_path = os.path.join(server_dir, lora_path)

                    if os.path.isdir(lora_path):
                        embedding_files = glob.glob(os.path.join(lora_path, "*embeddings.safetensors"))
                        special_params_file = os.path.join(lora_path, "special_params.json")
                    else:
                        lora_dir = os.path.dirname(lora_path)
                        lora_basename = os.path.splitext(os.path.basename(lora_path))[0]
                        embedding_files = glob.glob(os.path.join(lora_dir, f"{lora_basename}_embeddings.safetensors"))
                        if not embedding_files:
                            embedding_files = glob.glob(os.path.join(lora_dir, "*embeddings.safetensors"))
                        special_params_file = os.path.join(lora_dir, "special_params.json")

                    if embedding_files:
                        embedding_path = embedding_files[0]
                        print(f"[img2imgSDXL_peft.py] Found textual inversion embedding: {embedding_path}")

                        token_str = "<s0><s1><s2>"
                        if os.path.exists(special_params_file):
                            try:
                                import json
                                with open(special_params_file, 'r') as f:
                                    special_params = json.load(f)
                                    token_str = special_params.get('TOK', token_str)
                                print(f"[img2imgSDXL_peft.py] Loaded token string from special_params.json: {token_str}")
                            except Exception as e:
                                print(f"[img2imgSDXL_peft.py] Warning: Could not read special_params.json: {e}")

                        self._load_and_fuse_embeddings(target_pipe, embedding_path, token_str=token_str)
                        print(f"[img2imgSDXL_peft.py] Textual inversion embeddings fused for LoRA: {lora_name}")

        # Store the lora_scale for runtime use
        self.lora_scale = lora_config.DEFAULT_LORA_SCALE if lora_config is not None else 1.0

        # Apply torch.compile if enabled
        if self.use_torch_compile:
            print(f"[img2imgSDXL_peft.py] Applying torch.compile to UNet...")
            self._apply_torch_compile()
        else:
            print(f"[img2imgSDXL_peft.py] No acceleration enabled (use --torch-compile to enable)")

        # Prepare with default prompt
        default_seed = 4402026899276587
        self.shared_wrapper.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.0,
        )
        self.shared_wrapper.update_stream_params(seed=default_seed)
        print(f"[img2imgSDXL_peft.py] Initialized with default seed: {default_seed}")

        # Initialize PromptTravel
        self.shared_wrapper.prompt_travel = PromptTravel(
            text_encoder=self.shared_wrapper.stream.pipe.text_encoder,
            tokenizer=self.shared_wrapper.stream.pipe.tokenizer,
            text_encoder_2=self.shared_wrapper.stream.pipe.text_encoder_2,
            tokenizer_2=self.shared_wrapper.stream.pipe.tokenizer_2,
        )

        # For compatibility with main.py which references self.pipes
        # Use a single-element list since we have only one pipe
        self.pipes = [self.shared_wrapper]
        print(f"[img2imgSDXL_peft.py] Created 1 pipe (single UNet with PEFT adapters)")
        print(f"[img2imgSDXL_peft.py] Loaded adapters: {self.adapter_names}")

        # State tracking
        self.current_pipe_idx = 0  # Kept for compatibility with main.py
        self.last_prompt = default_prompt
        self.last_controlnet_scale = None
        self.last_seed = None
        self.last_target_seed = None
        self.last_latent_travel_settings = None

        # Cache for prompt travel embeddings
        self.prompt_embeds_cache = {0: {}}

    def _parse_lora_weights(self, weights_param) -> list[float]:
        """Parse lora_weights from various input formats into a list of floats."""
        if isinstance(weights_param, list):
            return [float(w) for w in weights_param]
        if isinstance(weights_param, str):
            try:
                return [float(w.strip()) for w in weights_param.split(",") if w.strip()]
            except ValueError:
                print(f"[img2imgSDXL_peft.py] WARNING: Could not parse lora_weights string: {weights_param}")
                return list(self.current_lora_weights)
        return list(self.current_lora_weights)

    def _update_lora_weights(self, new_weights: list[float]):
        """Update LoRA adapter weights via PEFT set_adapters if they changed."""
        if not self.adapter_names:
            return

        # Ensure weights match adapter count
        if len(new_weights) < len(self.adapter_names):
            new_weights = new_weights + [0.0] * (len(self.adapter_names) - len(new_weights))
        elif len(new_weights) > len(self.adapter_names):
            new_weights = new_weights[:len(self.adapter_names)]

        # Check if weights actually changed
        if new_weights == self.current_lora_weights:
            return

        # Apply new weights via PEFT
        scaled_weights = [w * self.lora_scale for w in new_weights]
        self.shared_wrapper.stream.pipe.set_adapters(
            adapter_names=self.adapter_names,
            adapter_weights=scaled_weights,
        )
        old_weights = self.current_lora_weights
        self.current_lora_weights = new_weights
        print(f"[img2imgSDXL_peft.py] LoRA weights updated: {old_weights} -> {new_weights} (scale={self.lora_scale})")

    def _load_and_fuse_embeddings(self, pipe, embedding_path, token_str="<s0><s1><s2>"):
        """Load textual inversion embeddings and fuse them into text encoders."""
        import re

        print(f"[img2imgSDXL_peft.py] Loading embeddings from {embedding_path}")
        embedding_data = load_file(embedding_path)

        print(f"[img2imgSDXL_peft.py] Embedding keys: {list(embedding_data.keys())}")
        for key in embedding_data.keys():
            print(f"[img2imgSDXL_peft.py]   {key}: shape={embedding_data[key].shape}, dtype={embedding_data[key].dtype}")

        tokens = re.findall(r'<s\d+>', token_str)
        print(f"[img2imgSDXL_peft.py] Parsed tokens: {tokens}")
        num_tokens = len(tokens)

        clip_l_embeds = embedding_data.get('clip_l')
        clip_g_embeds = embedding_data.get('clip_g')

        if clip_l_embeds is None or clip_g_embeds is None:
            print(f"[img2imgSDXL_peft.py] ERROR: Missing clip_l or clip_g embeddings!")
            return

        if clip_l_embeds.shape[0] != num_tokens or clip_g_embeds.shape[0] != num_tokens:
            print(f"[img2imgSDXL_peft.py] WARNING: Embedding shape mismatch!")

        tokenizer_1 = pipe.tokenizer
        text_encoder_1 = pipe.text_encoder
        tokenizer_2 = pipe.tokenizer_2
        text_encoder_2 = pipe.text_encoder_2

        num_added_tokens_1 = tokenizer_1.add_tokens(tokens)
        text_encoder_1.resize_token_embeddings(len(tokenizer_1))
        token_ids_1 = tokenizer_1.convert_tokens_to_ids(tokens)

        with torch.no_grad():
            for i, token_id in enumerate(token_ids_1):
                text_encoder_1.get_input_embeddings().weight[token_id] = clip_l_embeds[i].to(
                    device=text_encoder_1.device, dtype=text_encoder_1.dtype
                )

        num_added_tokens_2 = tokenizer_2.add_tokens(tokens)
        text_encoder_2.resize_token_embeddings(len(tokenizer_2))
        token_ids_2 = tokenizer_2.convert_tokens_to_ids(tokens)

        with torch.no_grad():
            for i, token_id in enumerate(token_ids_2):
                text_encoder_2.get_input_embeddings().weight[token_id] = clip_g_embeds[i].to(
                    device=text_encoder_2.device, dtype=text_encoder_2.dtype
                )

        print(f"[img2imgSDXL_peft.py] Successfully fused textual inversion embeddings!")
        print(f"[img2imgSDXL_peft.py] You can now use tokens: {token_str} in your prompts")

    def _apply_torch_compile(self):
        """Apply torch.compile to the UNet for acceleration."""
        import torch

        unet = self.shared_wrapper.stream.unet

        if 'Engine' in type(unet).__name__:
            print(f"[img2imgSDXL_peft.py] UNet is TensorRT engine, skipping torch.compile")
            return

        torch.set_float32_matmul_precision('high')

        if self.use_regional_compile:
            print(f"[img2imgSDXL_peft.py] Applying REGIONAL torch.compile to transformer blocks...")
            compiled_blocks = 0

            if hasattr(unet, 'down_blocks'):
                for i, block in enumerate(unet.down_blocks):
                    if hasattr(block, 'attentions') and block.attentions is not None:
                        for j, attn in enumerate(block.attentions):
                            unet.down_blocks[i].attentions[j] = torch.compile(
                                attn, mode="reduce-overhead", fullgraph=True, dynamic=True
                            )
                            compiled_blocks += 1
                    if hasattr(block, 'resnets'):
                        for j, resnet in enumerate(block.resnets):
                            unet.down_blocks[i].resnets[j] = torch.compile(
                                resnet, mode="reduce-overhead", fullgraph=True, dynamic=True
                            )
                            compiled_blocks += 1

            if hasattr(unet, 'mid_block') and unet.mid_block is not None:
                if hasattr(unet.mid_block, 'attentions'):
                    for i, attn in enumerate(unet.mid_block.attentions):
                        unet.mid_block.attentions[i] = torch.compile(
                            attn, mode="reduce-overhead", fullgraph=True, dynamic=True
                        )
                        compiled_blocks += 1
                if hasattr(unet.mid_block, 'resnets'):
                    for i, resnet in enumerate(unet.mid_block.resnets):
                        unet.mid_block.resnets[i] = torch.compile(
                            resnet, mode="reduce-overhead", fullgraph=True, dynamic=True
                        )
                        compiled_blocks += 1

            if hasattr(unet, 'up_blocks'):
                for i, block in enumerate(unet.up_blocks):
                    if hasattr(block, 'attentions') and block.attentions is not None:
                        for j, attn in enumerate(block.attentions):
                            unet.up_blocks[i].attentions[j] = torch.compile(
                                attn, mode="reduce-overhead", fullgraph=True, dynamic=True
                            )
                            compiled_blocks += 1
                    if hasattr(block, 'resnets'):
                        for j, resnet in enumerate(block.resnets):
                            unet.up_blocks[i].resnets[j] = torch.compile(
                                resnet, mode="reduce-overhead", fullgraph=True, dynamic=True
                            )
                            compiled_blocks += 1

            print(f"[img2imgSDXL_peft.py] Regional compilation complete: {compiled_blocks} blocks compiled")
        else:
            print(f"[img2imgSDXL_peft.py] Applying FULL torch.compile to UNet...")
            self.shared_wrapper.stream.unet = torch.compile(
                unet, mode="reduce-overhead", fullgraph=True, dynamic=True
            )
            print(f"[img2imgSDXL_peft.py] Full UNet compilation complete")

        vae = self.shared_wrapper.stream.vae
        if 'Engine' not in type(vae).__name__ and hasattr(vae, 'decoder'):
            print(f"[img2imgSDXL_peft.py] Compiling VAE decoder...")
            vae.decoder = torch.compile(
                vae.decoder, mode="reduce-overhead", fullgraph=True, dynamic=True
            )
            print(f"[img2imgSDXL_peft.py] VAE decoder compilation complete")

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        _v = self._verbose

        # Parse and apply LoRA weights from params
        # Supports both:
        #   1. lora_weights as a string/list directly on params
        #   2. pipe_index mapped to adapter_weights_sets (set by main.py)
        lora_weights_param = getattr(params, 'lora_weights', None)
        if lora_weights_param is not None:
            new_weights = self._parse_lora_weights(lora_weights_param)
            self._update_lora_weights(new_weights)

        stream_wrapper = self.shared_wrapper

        if _v: print(f"[img2imgSDXL_peft.py] Params: {params}")

        # Handle prompt travel
        use_prompt_travel = getattr(params, "use_prompt_travel", False)
        if _v: print(f"[img2imgSDXL_peft.py] use_prompt_travel: {use_prompt_travel}")

        if use_prompt_travel:
            multi_file_mode = getattr(params, 'multi_file_prompts', False)
            prompt_travel_factor = getattr(params, 'prompt_travel_factor', 0.5)

            if multi_file_mode:
                source_prompts = params.prompt
                target_prompts = getattr(params, 'target_prompt', params.prompt)
                spatial_weights = getattr(params, 'spatial_weights', None)

                if _v:
                    print(f"[img2imgSDXL_peft.py] === MULTI-FILE PROMPT TRAVEL ===")
                    print(f"[img2imgSDXL_peft.py] Number of files: {len(source_prompts)}")
                    print(f"[img2imgSDXL_peft.py] Spatial weights: {spatial_weights}")
                    print(f"[img2imgSDXL_peft.py] Temporal factor: {prompt_travel_factor:.3f}")

                cache = self.prompt_embeds_cache[0]
                source_embeds_list = []
                source_pooled_list = []

                for i, prompt in enumerate(source_prompts):
                    if prompt not in cache:
                        if _v: print(f"[img2imgSDXL_peft.py] Cache MISS - encoding source prompt {i}")
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

                source_embeds = sum(w * e for w, e in zip(spatial_weights, source_embeds_list))
                source_pooled = sum(w * p for w, p in zip(spatial_weights, source_pooled_list))

                target_embeds_list = []
                target_pooled_list = []

                for i, prompt in enumerate(target_prompts):
                    if prompt not in cache:
                        if _v: print(f"[img2imgSDXL_peft.py] Cache MISS - encoding target prompt {i}")
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

                target_embeds = sum(w * e for w, e in zip(spatial_weights, target_embeds_list))
                target_pooled = sum(w * p for w, p in zip(spatial_weights, target_pooled_list))

            else:
                source_prompt = params.prompt
                target_prompt = getattr(params, 'target_prompt', params.prompt)

                if _v:
                    print(f"[img2imgSDXL_peft.py] === SINGLE-FILE PROMPT TRAVEL ===")
                    print(f"[img2imgSDXL_peft.py] factor: {prompt_travel_factor:.3f}")

                cache = self.prompt_embeds_cache[0]
                if source_prompt not in cache:
                    source_embeds, _, source_pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                        prompt=source_prompt,
                        device=stream_wrapper.stream.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    cache[source_prompt] = (source_embeds, source_pooled)
                else:
                    source_embeds, source_pooled = cache[source_prompt]

                if target_prompt not in cache:
                    target_embeds, _, target_pooled, _ = stream_wrapper.prompt_travel.encode_prompt_sdxl(
                        prompt=target_prompt,
                        device=stream_wrapper.stream.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    cache[target_prompt] = (target_embeds, target_pooled)
                else:
                    target_embeds, target_pooled = cache[target_prompt]

            interpolated_embeds, interpolated_pooled = stream_wrapper.prompt_travel.interpolate_embeddings_sdxl(
                embeds_from=(source_embeds, source_pooled),
                embeds_to=(target_embeds, target_pooled),
                factor=prompt_travel_factor,
            )

            batch_size = stream_wrapper.stream.batch_size
            interpolated_embeds_batched = interpolated_embeds.repeat(batch_size, 1, 1)

            stream_wrapper.stream.prompt_embeds = interpolated_embeds_batched
            stream_wrapper.stream.add_text_embeds = interpolated_pooled

        else:
            prompt = params.prompt
            if prompt != self.last_prompt:
                stream_wrapper.prepare(
                    prompt=prompt,
                    negative_prompt=default_negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=1.0,
                )
                self.last_prompt = prompt

        # Handle seed and latent travel
        use_latent_travel = getattr(params, 'use_latent_travel', False)
        seed = getattr(params, 'seed', 4402026899276587)
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = seed + 1

        latent_travel_factor = getattr(params, 'latent_travel_factor', 0.5)
        latent_travel_method = getattr(params, 'latent_travel_method', 'slerp')

        current_latent_travel_settings = (use_latent_travel, seed, target_seed, latent_travel_factor, latent_travel_method)

        should_update = use_latent_travel or (current_latent_travel_settings != self.last_latent_travel_settings)

        if should_update:
            if use_latent_travel:
                source_weight = 1.0 - latent_travel_factor
                target_weight = latent_travel_factor
                seed_list = [
                    (seed, source_weight),
                    (target_seed, target_weight)
                ]
                stream_wrapper.update_stream_params(
                    seed_list=seed_list,
                    seed_interpolation_method=latent_travel_method
                )
            else:
                stream_wrapper.update_stream_params(seed=seed)

            self.last_latent_travel_settings = current_latent_travel_settings

        # Update ControlNet control image (only if ControlNet is enabled)
        if hasattr(stream_wrapper.stream, '_controlnet_module') and stream_wrapper.stream._controlnet_module is not None:
            control_image = getattr(params, 'control_image', params.image)
            if control_image is not None:
                stream_wrapper.update_control_image(index=0, image=control_image)

            # Update ControlNet conditioning scale
            if hasattr(params, 'controlnet_scale') and params.controlnet_scale != self.last_controlnet_scale:
                stream_wrapper.stream._controlnet_module.update_controlnet_scale(index=0, scale=params.controlnet_scale)
                self.last_controlnet_scale = params.controlnet_scale

        # Update temporal coherence
        temporal_coherence = getattr(params, 'temporal_coherence', None)
        temporal_coherence_latent = getattr(params, 'temporal_coherence_latent', None)
        if temporal_coherence is not None or temporal_coherence_latent is not None:
            stream_wrapper.update_stream_params(
                temporal_coherence=temporal_coherence,
                temporal_coherence_latent=temporal_coherence_latent
            )

        # Generate
        image_tensor = stream_wrapper.preprocess_image(params.image)
        output_image = stream_wrapper(image=image_tensor)

        # Debug controlnet overlay
        if params.debug_controlnet:
            preprocessed_control = stream_wrapper.get_last_processed_image(index=0)
            if preprocessed_control is not None:
                scale_factor = 2 if self.use_upscaler else 1
                w0, h0 = (scale_factor * 200, scale_factor * 200)
                control_image_resized = preprocessed_control.resize((w0, h0))
                w1, h1 = output_image.size
                output_image.paste(control_image_resized, (w1 - w0, h1 - h0))

        return output_image
