from diffusers import (
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
)
from pipelines.diffusers_pipelines.pipeline_controlnet import StableDiffusionControlNetPipeline
from compel import Compel
import torch
import os
import glob
from typing import List, Dict, Optional, Union
import gc
from pathlib import Path
import time

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
import uuid
import torchvision.transforms as T
from diffusers.utils.remote_utils import remote_decode
from typing import Optional
import torch

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel
# Import the upscaler processor
from modules.upscaler import get_processor as get_upscaler_processor

taesd_model = "madebyollin/taesd"
controlnet_model = "thibaud/controlnet-sd21-depth-diffusers"
base_model = "stabilityai/sd-turbo"

# LoRA models
lora_models = {
    "None": None,
    "radames/sd-21-DPO-LoRA": "radames/sd-21-DPO-LoRA",
    "latent-consistency/lcm-lora-sdv2-1": "latent-consistency/lcm-lora-sdv2-1",
    "latent-consistency/lcm-lora-sdv2-1-turbo": "latent-consistency/lcm-lora-sdv2-1-turbo",
    "hakurei/waifu-diffusion": "hakurei/waifu-diffusion",
    "ostris/ikea-instructions-lora": "ostris/ikea-instructions-lora",
    "ostris/super-cereal-sdxl-lora": "ostris/super-cereal-sdxl-lora",
    "pbarbarant/sd-sonio": "pbarbarant/sd-sonio",
    "artificialguybr/studioghibli-redmond-2-1v-studio-ghibli-lora-for-freedom-redmond-sd-2-1": "artificialguybr/studioghibli-redmond-2-1v-studio-ghibli-lora-for-freedom-redmond-sd-2-1",
    "style_pi_2": "server/loras/style_pi_2.safetensors",
    "pytorch_lora_weights": "server/loras/pytorch_lora_weights.safetensors",
    "FKATwigs_A1-000038": "server/loras/FKATwigs_A1-000038.safetensors",
    "dark": "server/loras/flowers-000022.safetensors",
    "marina1": "server/loras/marina-glitch-000140.safetensors",
    "marina-red": "server/loras/marina-red-000140.safetensors",
    "abstract-monochrome": "server/loras/abstract-monochrome-000140.safetensors",
    "abstract-brokenglass-red": "server/loras/abstract_brokenglass_red-000140.safetensors",
    "full-body-glitch-monochrome": "server/loras/full_body_glitch-monochrome-000140.safetensors",
    "full-body-glitch-reddish": "server/loras/full_body_glitch-reddish-000140.safetensors",
    "mid-body-shoulders-glitch-monochrome": "server/loras/mid_body_shoulders_glitch-monochrome-000140.safetensors",
    "mid-body-shoulders-glitch-reddish": "server/loras/mid_body_shoulders_glitch-reddish-000140.safetensors",
    "mid-body-torso-glitch-monochrome": "server/loras/mid_body_torso_glitch-monochrome-000140.safetensors",
    "mid-body-torso-glitch-reddish": "server/loras/mid_body_torso_glitch-reddish-000140.safetensors",
    "melier-bw": "server/loras/melier_bw-000052.safetensors",
    "melier-col": "server/loras/melier_col-000032.safetensors",
    "nature-bw": "server/loras/nature_bw-000052.safetensors",
    "nature-water": "server/loras/nature_water-000072.safetensors",
    "robwood": "server/loras/robwood-000060.safetensors",
    "sweet-vicious": "server/loras/sweet_vicious-000072.safetensors",
    "liquid-love": "server/loras/liquid_love-000032.safetensors"
}

# Default LoRAs to use - can be a single LoRA or a list of LoRAs to fuse
DEFAULT_CURATION_INDEX = 3

lora_curation = {
    "glitch_abstract": ["full-body-glitch-reddish", "abstract-monochrome"],
    "melier": ["melier-bw", "melier-col"],
    "liquid_nature": ["nature-bw", "nature-water"],
    "sweet_robwood": ["sweet-vicious", "robwood"]
}

curation_keys = ["glitch_abstract", "melier", "liquid_nature", "sweet_robwood"]

# Define adapter weights sets - one set per lora_curation element
adapter_weights_set_curation = {
    "glitch_abstract": [  # ["full-body-glitch-reddish", "abstract-monochrome"]
        [1.0, 0.0],    # Full weight on first LoRA
        [0.75, 0.25],  # More weight on first LoRA
        [0.5, 0.5],    # Equal weights
        [0.25, 0.75],  # More weight on second LoRA
        [0.0, 1.0]     # Full weight on second LoRA
    ],
    "melier": [  
        [1.0, 0.0],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.1, 0.9]
    ],
    "liquid_nature": [ 
        [0.6, 0.4],
        [0.6, 0.7],
        [0.4, 0.9],
        [0.3, 0.9],
        [0.2, 0.9]
    ],
    "sweet_robwood": [
        [0.9, 0.1],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.4, 0.6],
        [0.3, 0.7]
    ]
}

adapter_weights_sets = adapter_weights_set_curation[curation_keys[DEFAULT_CURATION_INDEX]]

# adapter_weights_sets = [
#     [1.0, 0.0],    # First set: full weight on first LoRA
#     [0.75, 0.25],
#     [0.5, 0.5],    # Middle set: equal weights
#     [0.25, 0.75],
#     [0.0, 1.0]     # Last set: full weight on second LoRA
# ]


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
default_prompt = prompt_prefix + "a red monkey"
default_target_prompt = prompt_prefix + "a blue dog"

page_content = """
<h1 class="text-3xl font-bold">Real-Time SDv2.1 Turbo</h1>
<h3 class="text-xl font-bold">Image-to-Image ControlNet</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD Turbo</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl_turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Diffusers</a
    > with a MJPEG stream server.
</p>
<p class="text-sm text-gray-500">

    Change the prompt to generate different images, accepts <a
    href="https://github.com/damian0815/compel/blob/main/doc/syntax.md"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">Compel</a
    > syntax.
</p>
"""
# openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
# midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+sd21Turbo"
        title: str = "SDv2.1 Turbo + Controlnet"
        description: str = "Generates an image from a text prompt"
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
            default_target_prompt,
            title="Target Prompt",
            field="textarea",
            id="target_prompt",
            hide=True,
        )
        curation_index: int = Field(
            DEFAULT_CURATION_INDEX,
            min=0,
            max=len(curation_keys) - 1,
            title="Curation Index",
            field="range",
            id="curation_index",
            description=f"Select which LoRA pair to use: {', '.join(curation_keys)}"
        )
        pipe_index: int = Field(
            0,
            min=0,
            max=4,
            step=1,
            title="Pipe Index",
            field="range",
            id="pipe_index",
            description="Select which weight combination to use for the selected LoRA pair"
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
        lora_models: List[str] = Field(
            [lora_curation[curation_keys[DEFAULT_CURATION_INDEX]][0]] if lora_curation[curation_keys[DEFAULT_CURATION_INDEX]] else [],
            title="LoRA Models",
            field="multiselect",
            id="lora_models",
            options=list(lora_models.keys()),
        )
        fuse_loras: bool = Field(
            False,
            title="Fuse LoRAs",
            field="checkbox",
            id="fuse_loras",
        )
        lora_scale: float = Field(
            1.0,
            min=0.0,
            max=2.0,
            step=0.01,
            title="LoRA Scale",
            field="range",
            id="lora_scale",
        )
        seed: int = Field(
            4402026899276587, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        target_seed: Optional[int] = Field(
            None, min=0, title="Target Seed", field="seed", hide=True, id="target_seed"
        )
        steps: int = Field(
            1, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            640, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            480, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.21,
            min=0,
            max=10,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.8,
            min=0.10,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            0.325,
            min=0,
            max=1.0,
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
        debug_controlnet: bool = Field(
            False,
            title="Debug ControlNet",
            field="checkbox",
            hide=True,
            id="debug_controlnet",
        )
        use_output_bg_removal: bool = Field(
            False,
            title="Use Output Background Removal",
            field="checkbox",
            id="use_output_bg_removal",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # Add current_curation_index to track changes
        self.current_curation_index = None
        
        self.pipes = []
        self.pipe_states = []
        
        # Initialize upscaler processor if enabled
        self.use_upscaler = getattr(args, 'use_upscaler', False)
        if self.use_upscaler:
            upscaler_type = getattr(args, 'upscaler_type', 'pil')
            self.upscaler_processor = get_upscaler_processor(device=device.type, upscaler_type=upscaler_type)
            print(f"[controlnetSDTurbot2i.py] {upscaler_type.upper()} upscaler processor initialized")
        
        for adapter_weights in adapter_weights_sets:
            # Create pipeline with ControlNet model
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model,
                controlnet=ControlNetModel.from_pretrained(
                    controlnet_model, torch_dtype=torch_dtype
                ).to(device),
                safety_checker=None,
                torch_dtype=torch_dtype,
            )

            if args.taesd:
                pipe.vae = AutoencoderTiny.from_pretrained(
                    taesd_model, torch_dtype=torch_dtype, use_safetensors=True
                ).to(device)

            if args.sfast:
                print("Using sfast compile\n")
                from sfast.compilers.stable_diffusion_pipeline_compiler import (
                    compile,
                    CompilationConfig,
                )

                config = CompilationConfig.Default()
                config.enable_xformers = True
                config.enable_triton = True
                config.enable_cuda_graph = True
                config.enable_jit = True
                pipe = compile(pipe, config=config)

                print("\nRunning with sfast compile\n")

            if args.onediff:
                print("\nRunning onediff compile\n")
                from onediff.infer_compiler import oneflow_compile

                pipe.unet = oneflow_compile(pipe.unet)
                pipe.vae.encoder = oneflow_compile(pipe.vae.encoder)
                pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
                pipe.controlnet = oneflow_compile(pipe.controlnet)

            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            pipe.to(device=device, dtype=torch_dtype)
            if device.type != "mps":
                pipe.unet.to(memory_format=torch.channels_last)

            if args.compel:
                from compel import Compel

                pipe.compel_proc = Compel(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    truncate_long_prompts=True,
                )

            if args.taesd:
                pipe.vae = AutoencoderTiny.from_pretrained(
                    taesd_model, torch_dtype=torch_dtype, use_safetensors=True
                ).to(device)

            # NOTE: torch compile temp ENABLED
            if args.torch_compile:
                print("\nRunning torch compile\n")
                pipe.unet = torch.compile(
                    pipe.unet, mode="reduce-overhead", fullgraph=True
                )
                pipe.vae = torch.compile(
                    pipe.vae, mode="reduce-overhead", fullgraph=True
                )

            # Initialize PromptTravel for both text and latent interpolation
            pipe.prompt_travel = PromptTravel(
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
            )
            
            # Create a state dictionary for this pipe
            pipe_state = {
                'current_lora_models': [],
                'fuse_loras': False,
                'lora_scale': 1.0,
                'lora_weights_loaded': False,
                'current_adapter_weights': [],
                'stored_result_latents': None
            }
            
            # Load default LoRA(s) during initialization with specific adapter weights
            print(f"Loading default LoRA(s): {lora_curation[curation_keys[DEFAULT_CURATION_INDEX]]} with weights {adapter_weights}")
            # If there's only one LoRA, don't fuse. If there are multiple, fuse them automatically
            should_fuse = True
            self.load_loras_for_pipe(pipe, pipe_state, lora_curation[curation_keys[DEFAULT_CURATION_INDEX]], fuse_loras=should_fuse, lora_scale=1.0, adapter_weights=adapter_weights)
            
            # Pass the upscaler processor to the pipeline if enabled
            if self.use_upscaler:
                pipe.upscaler = self.upscaler_processor
            
            self.pipes.append(pipe)
            self.pipe_states.append(pipe_state)

        # Store the current pipe index
        self.current_pipe_idx = 0

    def load_loras_for_pipe(self, pipe, pipe_state, lora_models_list: List[str], fuse_loras: bool = False, lora_scale: float = 1.0, adapter_weights: Optional[List[float]] = None) -> None:
        """
        Load or update LoRA models for a specific pipe.
        
        Args:
            pipe: The pipeline to load LoRAs into
            pipe_state: Dictionary containing pipe-specific state
            lora_models_list: List of LoRA model names to load
            fuse_loras: Whether to fuse multiple LoRAs
            lora_scale: Scale factor for LoRA weights
            adapter_weights: Optional list of weights for each LoRA adapter
        """
        start_time = time.time()
        
        # Filter out "None" from the selected LoRAs
        selected_loras = [lora for lora in lora_models_list if lora != "None"]
        
        # If no LoRAs are selected, unload any previously loaded LoRAs
        if not selected_loras:
            if pipe_state['lora_weights_loaded']:
                unload_start = time.time()
                pipe.unload_lora_weights()
                print(f"Unloading LoRA weights took: {time.time() - unload_start:.2f} seconds")
                pipe_state['lora_weights_loaded'] = False
            pipe_state['current_lora_models'] = []
            pipe_state['fuse_loras'] = False
            pipe_state['lora_scale'] = 1.0
            pipe_state['current_adapter_weights'] = []
            print(f"Total LoRA unloading time: {time.time() - start_time:.2f} seconds")
            return
        
        # Ensure adapter_weights is a list of the same length as selected_loras
        if adapter_weights is None:
            adapter_weights = [1.0] * len(selected_loras)
        elif len(adapter_weights) < len(selected_loras):
            # Pad with 1.0 if not enough weights provided
            adapter_weights = adapter_weights + [1.0] * (len(selected_loras) - len(adapter_weights))
        elif len(adapter_weights) > len(selected_loras):
            # Truncate if too many weights provided
            adapter_weights = adapter_weights[:len(selected_loras)]
        
        # Check if we need to reload LoRAs
        if (selected_loras != pipe_state['current_lora_models']) or (fuse_loras != pipe_state['fuse_loras']) or (lora_scale != pipe_state['lora_scale']) or (adapter_weights != pipe_state['current_adapter_weights']):
            # Unload any previously loaded LoRAs
            if pipe_state['lora_weights_loaded']:
                unload_start = time.time()
                pipe.unload_lora_weights()
                print(f"Unloading previous LoRA weights took: {time.time() - unload_start:.2f} seconds")
                pipe_state['lora_weights_loaded'] = False
            
            # Load all selected LoRAs
            load_start = time.time()
            for i, lora_id in enumerate(selected_loras):
                adapter_name = f"lora_{i}"
                lora_load_start = time.time()
                print(f"Loading LoRA: {lora_id} as {adapter_name}")
                pipe.load_lora_weights(lora_models[lora_id], adapter_name=adapter_name)
                print(f"Loading {lora_id} took: {time.time() - lora_load_start:.2f} seconds")
            print(f"Total LoRA loading time: {time.time() - load_start:.2f} seconds")
            
            if fuse_loras and len(selected_loras) > 1:
                # Fuse multiple LoRAs
                fuse_start = time.time()
                print(f"Fusing LoRAs: {selected_loras} with scale {lora_scale}")
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                # First set the adapters with their respective weights
                pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                # Then fuse them with the global scale
                pipe.fuse_lora(adapter_names=adapter_names, lora_scale=lora_scale)
                # Unload the individual LoRAs after fusing
                pipe.unload_lora_weights()
                pipe_state['lora_weights_loaded'] = False
                print(f"Fusing LoRAs took: {time.time() - fuse_start:.2f} seconds")
            else:
                # Set the LoRAs as active with their respective weights
                set_start = time.time()
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                print(f"Setting adapters took: {time.time() - set_start:.2f} seconds")
                pipe_state['lora_weights_loaded'] = True
            
            # Update current LoRA state
            pipe_state['current_lora_models'] = selected_loras
            pipe_state['fuse_loras'] = fuse_loras
            pipe_state['lora_scale'] = lora_scale
            pipe_state['current_adapter_weights'] = adapter_weights
            
        print(f"Total LoRA operation time: {time.time() - start_time:.2f} seconds")

    def update_lora_set(self, pipe, pipe_state, curation_index: int) -> None:
        """Helper function to update LoRAs if curation_index has changed"""
        if self.current_curation_index != curation_index:
            print(f"Updating LoRAs for curation index {curation_index}")
            
            # First unfuse and unload any existing LoRAs
            if pipe_state['lora_weights_loaded'] or hasattr(pipe, 'fused_lora_weights'):
                print("Unfusing and unloading existing LoRAs")
                if hasattr(pipe, 'fused_lora_weights'):
                    print("Unfusing LoRAs")
                    pipe.unfuse_lora()
                pipe.unload_lora_weights()
                print("Unloading LoRA weights took: {time.time() - unload_start:.2f} seconds")
                pipe_state['lora_weights_loaded'] = False
                pipe_state['current_lora_models'] = []
                pipe_state['fuse_loras'] = False
                pipe_state['lora_scale'] = 1.0
                pipe_state['current_adapter_weights'] = []
            
            # Now load the new LoRA set
            default_loras = lora_curation[curation_keys[curation_index]]
            adapter_weights = adapter_weights_set_curation[curation_keys[curation_index]][0]  # Start with first weight set
            self.load_loras_for_pipe(pipe, pipe_state, default_loras, fuse_loras=True, lora_scale=1.0, adapter_weights=adapter_weights)
            self.current_curation_index = curation_index

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Use the pipe index from params
        self.current_pipe_idx = params.pipe_index
        pipe = self.pipes[self.current_pipe_idx]
        pipe_state = self.pipe_states[self.current_pipe_idx]
        
        # Update LoRAs if curation_index has changed
        self.update_lora_set(pipe, pipe_state, params.curation_index)
        
        # Get current adapter weights based on both curation_index and pipe_index
        adapter_weights = adapter_weights_set_curation[curation_keys[params.curation_index]][params.pipe_index]
        print(f"Using pipe {self.current_pipe_idx} with adapter weights {adapter_weights}")

        generator = torch.manual_seed(params.seed)
        
        # Use target_seed if available, otherwise use seed+1
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = params.seed + 1
        target_generator = torch.Generator(device=pipe.device).manual_seed(target_seed)

        prompt = params.prompt
        prompt_embeds = None
        negative_prompt_embeds = None

        control_image = getattr(params, 'control_image', None)

        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        # Use provided prompt embeddings if available - with safer attribute check
        has_prompt_embeds = hasattr(params, "prompt_embeds") and params.prompt_embeds is not None

        print("[controlnetSDTurbot2i.py] PIPELINE has_prompt_embeds: ", has_prompt_embeds)

        if has_prompt_embeds:
            prompt_embeds = params.prompt_embeds
            if hasattr(prompt_embeds, 'device') and prompt_embeds.device != pipe.device:
                prompt_embeds = prompt_embeds.to(pipe.device)
            prompt = None
            
            if hasattr(params, "negative_prompt_embeds") and params.negative_prompt_embeds is not None:
                negative_prompt_embeds = params.negative_prompt_embeds
                if hasattr(negative_prompt_embeds, 'device') and negative_prompt_embeds.device != pipe.device:
                    negative_prompt_embeds = negative_prompt_embeds.to(pipe.device)
        
        elif hasattr(pipe, "compel_proc"):
            prompt_embeds = pipe.compel_proc(
                [params.prompt, "human, humanoid, figurine, face"]
            )
            prompt = None

        # Generate latents for source and target if latent travel is enabled
        latents = None
        if getattr(params, "use_latent_travel", False):
            # Check if we have stored result_latents from a previous call
            if pipe_state['stored_result_latents'] is not None:
                result_latents = pipe_state['stored_result_latents']
            else:
                # First call, no stored latents yet
                result_latents = None

            print("result_latents: ", result_latents.shape if result_latents is not None else "None")
                
            # Generate source latents
            source_latents = pipe.prompt_travel.prepare_latents(
                latents=result_latents,
                batch_size=1,
                num_channels_latents=pipe.unet.config.in_channels,
                vae_scale_factor=pipe.vae_scale_factor,
                scheduler=pipe.scheduler,
                height=params.height,
                width=params.width,
                dtype=pipe.dtype,
                device=pipe.device,
                generator=generator,
            )
            
            # Generate target latents with a different seed
            target_latents = pipe.prompt_travel.prepare_latents(
                latents=result_latents,
                batch_size=1,
                num_channels_latents=pipe.unet.config.in_channels,
                vae_scale_factor=pipe.vae_scale_factor,
                scheduler=pipe.scheduler,
                height=params.height,
                width=params.width,
                dtype=pipe.dtype,
                device=pipe.device,
                generator=target_generator,
            )
            
            latents = pipe.prompt_travel.interpolate_latents(
                source_latents,
                target_latents,
                getattr(params, "latent_travel_factor", 0.5),
                getattr(params, "latent_travel_method", "slerp")
            )

            print("prompt travel factor: ", getattr(params, "prompt_travel_factor", 0.5))
            print("latent travel factor: ", getattr(params, "latent_travel_factor", 0.5))
            print("prompt: ", prompt)
            print("prompt_embeds: ", prompt_embeds.shape if prompt_embeds is not None else "None")
            print("negative_prompt_embeds: ", negative_prompt_embeds.shape if negative_prompt_embeds is not None else "None")

        results = pipe(
            image=control_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=1,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
            latents=latents,
        )
        result_image = results[0][0]
        result_latents = results[1]

        # Store result_latents for next call if latent travel is enabled
        if getattr(params, "use_latent_travel", False):
            pipe_state['stored_result_latents'] = result_latents

        if params.debug_controlnet:
            # paste control_image on top of result_image
            scale_factor = 4
            w0, h0 = (scale_factor * 200, scale_factor * 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image
