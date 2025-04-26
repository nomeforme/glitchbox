from diffusers import (
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
    OnnxRuntimeModel,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from pipelines.diffusers_pipelines.pipeline_controlnet_tensorrt import TensorRTStableDiffusionControlNetImg2ImgPipeline, TensorRTModel
from compel import Compel
import torch
from pipelines.utils.canny_gpu import SobelOperator
import os
import glob
from typing import List, Dict, Optional, Union
import gc
from pathlib import Path
import time
import atexit
import torch.cuda as cuda
from pycuda.tools import make_default_context


try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
from controlnet_aux import OpenposeDetector, MidasDetector
import uuid
import torchvision.transforms as T
from diffusers.utils.remote_utils import remote_decode
from typing import Optional
import torch
import numpy as np

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel
#

# cuda.init()
# context = make_default_context()
# device = context.get_device()
# atexit.register(context.pop)

taesd_model = "madebyollin/taesd"
# controlnet_model = "thibaud/controlnet-sd21-canny-diffusers"
# controlnet_model = "thibaud/controlnet-sd21-openpose-diffusers"
controlnet_model = "thibaud/controlnet-sd21-depth-diffusers"
controlnet_canny_model = "thibaud/controlnet-sd21-canny-diffusers"
controlnet_depth_model = "thibaud/controlnet-sd21-depth-diffusers"
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
    "dark":"server/loras/flowers-000022.safetensors"
}

# Default LoRAs to use - can be a single LoRA or a list of LoRAs to fuse
default_loras = ["FKATwigs_A1-000038"]
# Default adapter weights for each LoRA (in the same order as default_loras)
default_adapter_weights = [1.0]
# default_loras = ["pbarbarant/sd-sonio"]
# default_loras = ["FKATwigs_A1-000038", "pbarbarant/sd-sonio"]

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
<h1 class="text-3xl font-bold">Real-Time SDv2.1 Turbo with TensorRT</h1>
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
    > with TensorRT acceleration and a MJPEG stream server.
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
        name: str = "controlnet+sd15Turbo+TRT"
        title: str = "SDv1.5 Turbo + Controlnet + TensorRT"
        description: str = "Generates an image from a text prompt using TensorRT acceleration"
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
            [default_loras[0]] if default_loras else [],
            title="LoRA Models",
            field="multiselect",
            id="lora_models",
            options=list(lora_models.keys()),
        )
        adapter_weights: List[float] = Field(
            default_adapter_weights,
            title="LoRA Adapter Weights",
            field="multiselect",
            id="adapter_weights",
            hide=True,
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
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
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
        canny_low_threshold: float = Field(
            0.31,
            min=0,
            max=1.0,
            step=0.001,
            title="Canny Low Threshold",
            field="range",
            hide=True,
            id="canny_low_threshold",
        )
        canny_high_threshold: float = Field(
            0.125,
            min=0,
            max=1.0,
            step=0.001,
            title="Canny High Threshold",
            field="range",
            hide=True,
            id="canny_high_threshold",
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
        controlnet_type: str = Field(
            "depth",
            title="ControlNet Type",
            field="select",
            id="controlnet_type",
            options=["depth", "canny"],
        )
        trt_engine_path: str = Field(
            "server/tensorrt_convert/onnx_models/unet/unet.engine",
            title="TensorRT Engine Path",
            field="text",
            id="trt_engine_path",
            hide=True,
        )
        onnx_model_dir: str = Field(
            "server/tensorrt_convert/onnx_models",
            title="ONNX Model Directory",
            field="text",
            id="onnx_model_dir",
            hide=True,
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # Initialize CUDA context first
        # print(f"[TIMING] Initializing CUDA context")
        # cuda_init_start = time.time()
        # cuda.init()
        # self.cuda_context = make_default_context()
        # device = self.cuda_context.get_device()
        # atexit.register(self.cuda_context.pop)
        # print(f"[TIMING] CUDA context initialized: {time.time() - cuda_init_start:.4f} seconds")

        # Initialize base pipeline first
        print(f"[TIMING] Initializing base pipeline")
        pipeline_start = time.time()
        base_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(base_model)
        base_pipeline.scheduler = UniPCMultistepScheduler.from_config(base_pipeline.scheduler.config)
        
        # Move VAE to CUDA and set dtype
        base_pipeline.vae = base_pipeline.vae.to(device, dtype=torch_dtype)
        print(f"[TIMING] Base pipeline initialized: {time.time() - pipeline_start:.4f} seconds")

        # Store base model's config for later use
        self.unet_config = base_pipeline.unet.config
        self.vae_config = base_pipeline.vae.config
        
        # Get TensorRT and ONNX paths
        # Get the absolute path to the workspace root
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Construct absolute paths for TensorRT engine and ONNX models
        trt_engine_path = getattr(args, "trt_engine_path", os.path.join(workspace_root, "server/tensorrt_convert/onnx_models/unet/unet.engine"))
        onnx_model_dir = getattr(args, "onnx_model_dir", os.path.join(workspace_root, "server/tensorrt_convert/onnx_models"))
        
        print(f"[DEBUG] Workspace root: {workspace_root}")
        print(f"[DEBUG] TensorRT engine path: {trt_engine_path}")
        print(f"[DEBUG] ONNX model dir: {onnx_model_dir}")
        
        # Initialize TensorRT pipeline
        print(f"[TIMING] Setting up TensorRT pipeline")
        trt_setup_start = time.time()
        provider = "CUDAExecutionProvider"
        
        # Load ONNX text encoder from local path
        text_encoder_path = os.path.join(onnx_model_dir, "text_encoder")
        if not os.path.exists(text_encoder_path):
            raise ValueError(f"ONNX text encoder not found at {text_encoder_path}")
            
        print(f"[DEBUG] Loading ONNX text encoder from: {text_encoder_path}")
        text_encoder = OnnxRuntimeModel.from_pretrained(
            text_encoder_path,
            provider=provider,
            local_files_only=True  # Ensure we only look for local files
        )
        
        # Load TensorRT UNet
        if not os.path.exists(trt_engine_path):
            raise ValueError(f"TensorRT engine not found at {trt_engine_path}")
            
        print(f"[DEBUG] Loading TensorRT UNet from: {trt_engine_path}")
        unet = TensorRTModel(trt_engine_path)
        
        # Create TensorRT pipeline
        self.pipe = TensorRTStableDiffusionControlNetImg2ImgPipeline(
            vae=base_pipeline.vae,
            text_encoder=text_encoder,
            tokenizer=base_pipeline.tokenizer,
            unet=unet,
            scheduler=base_pipeline.scheduler,
        )
        self.pipe = self.pipe.to(device)
        print(f"[TIMING] TensorRT pipeline setup: {time.time() - trt_setup_start:.4f} seconds")

        # Load ControlNet models
        print(f"[TIMING] Loading ControlNet models")
        controlnet_start = time.time()
        self.controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_canny_model, torch_dtype=torch_dtype
        )
        self.controlnet_depth = ControlNetModel.from_pretrained(
            controlnet_depth_model, torch_dtype=torch_dtype
        )
        self.current_controlnet = "depth"
        print(f"[TIMING] ControlNet models loaded: {time.time() - controlnet_start:.4f} seconds")

        # Initialize Canny edge detector
        self.canny_torch = SobelOperator(device=device)

        # Set up scheduler and other configurations
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

        # Optional TAESD setup
        if args.taesd:
            print(f"[TIMING] Loading TAESD")
            taesd_start = time.time()
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)
            print(f"[TIMING] TAESD loaded: {time.time() - taesd_start:.4f} seconds")

        # Set up Compel if enabled
        if args.compel:
            print(f"[TIMING] Setting up Compel")
            compel_start = time.time()
            self.pipe.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=True,
            )
            print(f"[TIMING] Compel setup: {time.time() - compel_start:.4f} seconds")

        # Initialize PromptTravel
        print(f"[TIMING] Setting up PromptTravel")
        prompt_travel_start = time.time()
        self.pipe.prompt_travel = PromptTravel(
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        print(f"[TIMING] PromptTravel setup: {time.time() - prompt_travel_start:.4f} seconds")
        
        # Initialize LoRA-related attributes
        self.current_lora_models = []
        self.fuse_loras = False
        self.lora_scale = 1.0
        self.lora_weights_loaded = False
        self.current_adapter_weights = []
        
        # Load default LoRA(s)
        print(f"[TIMING] Loading default LoRAs")
        lora_start = time.time()
        should_fuse = len(default_loras) > 1
        # self.load_loras(default_loras, fuse_loras=should_fuse, lora_scale=1.0, adapter_weights=default_adapter_weights)
        print(f"[TIMING] LoRAs loaded: {time.time() - lora_start:.4f} seconds")

    def load_loras(self, lora_models_list: List[str], fuse_loras: bool = False, lora_scale: float = 1.0, adapter_weights: Optional[List[float]] = None) -> None:
        """
        Load or update LoRA models for the pipeline.
        
        Args:
            lora_models_list: List of LoRA model names to load
            fuse_loras: Whether to fuse multiple LoRAs
            lora_scale: Scale factor for LoRA weights
            adapter_weights: Optional list of weights for each LoRA adapter
        """
        # Filter out "None" from the selected LoRAs
        selected_loras = [lora for lora in lora_models_list if lora != "None"]
        
        # If no LoRAs are selected, unload any previously loaded LoRAs
        if not selected_loras:
            if self.lora_weights_loaded:
                self.pipe.unload_lora_weights()
                self.lora_weights_loaded = False
            self.current_lora_models = []
            self.fuse_loras = False
            self.lora_scale = 1.0
            self.current_adapter_weights = []
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
        if (selected_loras != self.current_lora_models) or (fuse_loras != self.fuse_loras) or (lora_scale != self.lora_scale) or (adapter_weights != self.current_adapter_weights):
            # Unload any previously loaded LoRAs
            if self.lora_weights_loaded:
                self.pipe.unload_lora_weights()
                self.lora_weights_loaded = False
            
            # Load all selected LoRAs
            for i, lora_id in enumerate(selected_loras):
                adapter_name = f"lora_{i}"
                print(f"Loading LoRA: {lora_id} as {adapter_name}")
                self.pipe.load_lora_weights(lora_models[lora_id], adapter_name=adapter_name)
            
            if fuse_loras and len(selected_loras) > 1:
                # Fuse multiple LoRAs
                print(f"Fusing LoRAs: {selected_loras} with scale {lora_scale}")
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                # First set the adapters with their respective weights
                self.pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                # Then fuse them with the global scale
                self.pipe.fuse_lora(adapter_names=adapter_names, lora_scale=lora_scale)
                # Unload the individual LoRAs after fusing
                self.pipe.unload_lora_weights()
                self.lora_weights_loaded = False
            else:
                # Set the LoRAs as active with their respective weights
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                self.pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                self.lora_weights_loaded = True
            
            # Update current LoRA state
            self.current_lora_models = selected_loras
            self.fuse_loras = fuse_loras
            self.lora_scale = lora_scale
            self.current_adapter_weights = adapter_weights

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        
        # Use target_seed if available, otherwise use seed+1
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = params.seed + 1
        target_generator = torch.Generator(device=self.pipe.device).manual_seed(target_seed)

        prompt = params.prompt
        prompt_embeds = None

        if self.current_controlnet == "depth":
            control_image = getattr(params, 'control_image', None)
        if self.current_controlnet == "canny":
            control_image = self.canny_torch(
                params.image, params.canny_low_threshold, params.canny_high_threshold
            )

        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        # Use provided prompt embeddings if available - with safer attribute check
        has_prompt_embeds = hasattr(params, "prompt_embeds") and params.prompt_embeds is not None

        if has_prompt_embeds:
            prompt_embeds = params.prompt_embeds
            if hasattr(prompt_embeds, 'device') and prompt_embeds.device != self.pipe.device:
                prompt_embeds = prompt_embeds.to(self.pipe.device)
            prompt = None
            
            if hasattr(params, "negative_prompt_embeds") and params.negative_prompt_embeds is not None:
                negative_prompt_embeds = params.negative_prompt_embeds
                if hasattr(negative_prompt_embeds, 'device') and negative_prompt_embeds.device != self.pipe.device:
                    negative_prompt_embeds = negative_prompt_embeds.to(self.pipe.device)
        
        elif hasattr(self.pipe, "compel_proc"):
            prompt_embeds = self.pipe.compel_proc(
                [params.prompt, "human, humanoid, figurine, face"]
            )
            prompt = None

        # # Generate latents for source and target if latent travel is enabled
        # latents = None
        # if getattr(params, "use_latent_travel", False):
        #     # Check if we have stored result_latents from a previous call
        #     if hasattr(self, "stored_result_latents") and self.stored_result_latents is not None:
        #         result_latents = self.stored_result_latents
        #     else:
        #         # First call, no stored latents yet
        #         result_latents = None

        #     print("result_latents: ", result_latents.shape if result_latents is not None else "None")
                
        #     # Generate source latents
        #     source_latents = self.pipe.prompt_travel.prepare_latents(
        #         # processed_image,
        #         # self.pipe.scheduler.timesteps[:1],
        #         latents=result_latents,
        #         batch_size=1,
        #         # num_images_per_prompt=1,
        #         num_channels_latents=self.unet_config.in_channels,
        #         vae_scale_factor=self.vae_config.vae_scale_factor,
        #         scheduler=self.pipe.scheduler,
        #         height=params.height,
        #         width=params.width,
        #         dtype=self.pipe.dtype,
        #         device=self.pipe.device,
        #         generator=generator,
        #     )
            
        #     # Generate target latents with a different seed
        #     target_latents = self.pipe.prompt_travel.prepare_latents(
        #         # processed_image,
        #         # self.pipe.scheduler.timesteps[:1],
        #         latents=result_latents,
        #         batch_size=1,
        #         # num_images_per_prompt=1,
        #         num_channels_latents=self.unet_config.in_channels,
        #         vae_scale_factor=self.vae_config.vae_scale_factor,
        #         scheduler=self.pipe.scheduler,
        #         height=params.height,
        #         width=params.width,
        #         dtype=self.pipe.dtype,
        #         device=self.pipe.device,
        #         generator=target_generator,
        #     )
            
        #     latents = self.pipe.prompt_travel.interpolate_latents(
        #         source_latents,
        #         target_latents,
        #         getattr(params, "latent_travel_factor", 0.5), # NOTE: should belatent travel factor latent_travel_factor
        #         getattr(params, "latent_travel_method", "slerp")
        #     )

            print("prompt travel factor: ", getattr(params, "prompt_travel_factor", 0.5))
            print("latent travel factor: ", getattr(params, "latent_travel_factor", 0.5))

        # Get TensorRT engine path from params or use default
        trt_engine_path = getattr(params, "trt_engine_path", "server/tensorrt_convert/onnx_models/unet/unet.engine")

        # print("control image shape: ", control_image.shape)
        print("input height: ", params.height)
        print("input width: ", params.width)
        
        # Call the TensorRT pipeline
        results = self.pipe(
            num_controlnet=1,  # We're using a single ControlNet
            fp16=False,  # Use FP16 for better performance
            control_image=control_image,
            prompt=prompt,
            # prompt_embeds=prompt_embeds,
            # generator=generator,
            strength=strength,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
            # latents=latents,
        )
        
        # Extract the result image
        result_image = results.images[0]
        
        # Store result_latents for next call if latent travel is enabled
        if getattr(params, "use_latent_travel", False):
            # For TensorRT pipeline, we need to extract latents from the output
            # This might need adjustment based on the actual output format
            if hasattr(results, "latents"):
                setattr(self, "stored_result_latents", results.latents)

        if params.debug_controlnet:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image 