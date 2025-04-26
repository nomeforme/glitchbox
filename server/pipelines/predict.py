import argparse
import os
import time
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers import OnnxRuntimeModel
from diffusers import AutoencoderKL
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import the PromptTravel module
from server.modules.prompt_travel.prompt_travel import PromptTravel

from server.pipelines.diffusers_pipelines.pipeline_controlnet_tensorrt import TensorRTStableDiffusionControlNetImg2ImgPipeline, TensorRTModel

# Default values and constants
taesd_model = "madebyollin/taesd"
base_model = "stabilityai/sd-turbo"

onnx_model_dir = "server/tensorrt_convert/onnx_models"
unet_engine_path = "server/tensorrt_convert/onnx_models/unet/unet.engine"

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

# Default prompt
default_prompt = "a red dog and a blue monkey"
default_target_prompt = "a blue monkey and a red dog"

# Page content for UI
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

class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+sd15Turbo"
        title: str = "SDv1.5 Turbo + Controlnet"
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
            1.0,
            min=0.10,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            0.8,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Scale",
            field="range",
            hide=True,
            id="controlnet_scale",
        )
        controlnet_start: float = Field(
            0.3,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Start",
            field="range",
            hide=True,
            id="controlnet_start",
        )
        controlnet_end: float = Field(
            0.9,
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

    def __init__(self, args, device: torch.device, torch_dtype: torch.dtype):
        # Initialize the base pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(base_model)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # Move VAE to CUDA and set dtype
        self.pipe.vae = self.pipe.vae.to(device, dtype=torch_dtype)
        
        # Store device and dtype
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Initialize ONNX pipeline
        provider = "CUDAExecutionProvider"
        self.onnx_pipeline = TensorRTStableDiffusionControlNetImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=OnnxRuntimeModel.from_pretrained(
                os.path.join(onnx_model_dir, "text_encoder"), provider=provider
            ),
            tokenizer=self.pipe.tokenizer,
            unet=TensorRTModel(
                unet_engine_path,
                in_channels=4,  # Standard value for Stable Diffusion models
                out_channels=4,  # Standard value for Stable Diffusion models
                sample_size=64,  # Standard value for Stable Diffusion models
            ),
            scheduler=self.pipe.scheduler,
        )
        self.onnx_pipeline = self.onnx_pipeline.to(device)
        
        # Initialize PromptTravel for both text and latent interpolation
        self.onnx_pipeline.prompt_travel = PromptTravel(
            text_encoder=self.onnx_pipeline.text_encoder,
            tokenizer=self.onnx_pipeline.tokenizer,
        )
        
        # Initialize LoRA-related attributes
        self.current_lora_models = []
        self.fuse_loras = False
        self.lora_scale = 1.0
        self.lora_weights_loaded = False
        self.current_adapter_weights = []
        
        # Load default LoRA(s) during initialization
        print(f"Loading default LoRA(s): {default_loras}")
        # If there's only one LoRA, don't fuse. If there are multiple, fuse them automatically
        should_fuse = len(default_loras) > 1
        self.load_loras(default_loras, fuse_loras=should_fuse, lora_scale=1.0, adapter_weights=default_adapter_weights)

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
                self.onnx_pipeline.unload_lora_weights()
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
                self.onnx_pipeline.unload_lora_weights()
                self.lora_weights_loaded = False
            
            # Load all selected LoRAs
            for i, lora_id in enumerate(selected_loras):
                adapter_name = f"lora_{i}"
                print(f"Loading LoRA: {lora_id} as {adapter_name}")
                self.onnx_pipeline.load_lora_weights(lora_models[lora_id], adapter_name=adapter_name)
            
            if fuse_loras and len(selected_loras) > 1:
                # Fuse multiple LoRAs
                print(f"Fusing LoRAs: {selected_loras} with scale {lora_scale}")
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                # First set the adapters with their respective weights
                self.onnx_pipeline.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                # Then fuse them with the global scale
                self.onnx_pipeline.fuse_lora(adapter_names=adapter_names, lora_scale=lora_scale)
                # Unload the individual LoRAs after fusing
                self.onnx_pipeline.unload_lora_weights()
                self.lora_weights_loaded = False
            else:
                # Set the LoRAs as active with their respective weights
                adapter_names = [f"lora_{i}" for i in range(len(selected_loras))]
                self.onnx_pipeline.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
                self.lora_weights_loaded = True
            
            # Update current LoRA state
            self.current_lora_models = selected_loras
            self.fuse_loras = fuse_loras
            self.lora_scale = lora_scale
            self.current_adapter_weights = adapter_weights

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        print("PREDICT -params", params)
        generator = torch.manual_seed(params.seed)
        
        # Use target_seed if available, otherwise use seed+1
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = params.seed + 1
        target_generator = torch.Generator(device=self.device).manual_seed(target_seed)

        prompt = params.prompt
        prompt_embeds = None

        # Get control image from params (e.g. from depth anything etc.)
        control_image = getattr(params, 'control_image', None)

        
        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = 1  # Ensure at least one step

        # Use provided prompt embeddings if available - with safer attribute check
        has_prompt_embeds = hasattr(params, "prompt_embeds") and params.prompt_embeds is not None

        if has_prompt_embeds:
            prompt_embeds = params.prompt_embeds
            if hasattr(prompt_embeds, 'device') and prompt_embeds.device != self.device:
                prompt_embeds = prompt_embeds.to(self.device)
            prompt = None
            
            if hasattr(params, "negative_prompt_embeds") and params.negative_prompt_embeds is not None:
                negative_prompt_embeds = params.negative_prompt_embeds
                if hasattr(negative_prompt_embeds, 'device') and negative_prompt_embeds.device != self.device:
                    negative_prompt_embeds = negative_prompt_embeds.to(self.device)

        # Generate latents for source and target if latent travel is enabled
        latents = None
        if getattr(params, "use_latent_travel", False):
            # Check if we have stored result_latents from a previous call
            if hasattr(self, "stored_result_latents") and self.stored_result_latents is not None:
                result_latents = self.stored_result_latents
            else:
                # First call, no stored latents yet
                result_latents = None

            print("result_latents: ", result_latents.shape if result_latents is not None else "None")
                
            # Generate source latents
            source_latents = self.onnx_pipeline.prompt_travel.prepare_latents(
                latents=result_latents,
                batch_size=1,
                num_channels_latents=self.onnx_pipeline.unet.config.in_channels,
                vae_scale_factor=self.onnx_pipeline.vae_scale_factor,
                scheduler=self.onnx_pipeline.scheduler,
                height=params.height,
                width=params.width,
                dtype=self.torch_dtype,
                device=self.device,
                generator=generator,
            )
            
            # Generate target latents with a different seed
            target_latents = self.onnx_pipeline.prompt_travel.prepare_latents(
                latents=result_latents,
                batch_size=1,
                num_channels_latents=self.onnx_pipeline.unet.config.in_channels,
                vae_scale_factor=self.onnx_pipeline.vae_scale_factor,
                scheduler=self.onnx_pipeline.scheduler,
                height=params.height,
                width=params.width,
                dtype=self.torch_dtype,
                device=self.device,
                generator=target_generator,
            )
            
            # Interpolate between source and target latents
            latents = self.onnx_pipeline.prompt_travel.interpolate_latents(
                source_latents,
                target_latents,
                getattr(params, "latent_travel_factor", 0.5),
                getattr(params, "latent_travel_method", "slerp")
            )

            print("prompt travel factor: ", getattr(params, "prompt_travel_factor", 0.5))
            print("latent travel factor: ", getattr(params, "latent_travel_factor", 0.5))

        # print("PREDICT -control_image", control_image)

        # Run the pipeline
        results = self.onnx_pipeline(
            num_controlnet=1,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            # negative_prompt_embeds=negative_prompt_embeds,
            # generator=generator,
            control_image=control_image,
            width=params.width,
            height=params.height,
            strength=strength,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=params.controlnet_scale,
            control_guidance_start=params.controlnet_start,
            control_guidance_end=params.controlnet_end,
            guidance_scale=params.guidance_scale,
            latents=latents,
        )
        
        result_image = results[0][0]
        result_latents = results[1]

        # Store result_latents for next call if latent travel is enabled
        if getattr(params, "use_latent_travel", False):
            setattr(self, "stored_result_latents", result_latents)

        if params.debug_controlnet and control_image is not None:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image

# Main execution block for command-line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_model",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        required=True,
        help="Path to the ONNX directory",
    )

    parser.add_argument(
        "--unet_engine_path",
        type=str,
        required=True,
        help="Path to the unet + controlnet tensorrt model",
    )

    parser.add_argument("--qr_img_path", type=str, required=True, help="Path to the qr code image")

    args = parser.parse_args()

    print(f"[TIMING] Starting application")
    app_start_time = time.time()
    
    print(f"[TIMING] Loading QR image")
    qr_load_start = time.time()
    qr_image = Image.open(args.qr_img_path)
    qr_image = qr_image.resize((512, 512))
    print(f"[TIMING] QR image loaded: {time.time() - qr_load_start:.4f} seconds")

    # Create a simple Args class for compatibility
    class Args:
        def __init__(self, **kwargs):
            self.pipeline = kwargs.get("pipeline", "predict")
            self.model_id = kwargs.get("model_id", base_model)
            self.device = kwargs.get("device", "cuda")
            self.torch_dtype = kwargs.get("torch_dtype", torch.float16)
            self.use_compel = kwargs.get("use_compel", False)
            self.use_onnx = kwargs.get("use_onnx", True)
            self.use_tensorrt = kwargs.get("use_tensorrt", True)
            self.onnx_model_dir = kwargs.get("onnx_model_dir", onnx_model_dir)
            self.unet_engine_path = kwargs.get("unet_engine_path", unet_engine_path)
            self.taesd_model = kwargs.get("taesd_model", taesd_model)
            self.lora_models = kwargs.get("lora_models", lora_models)
            self.default_loras = kwargs.get("default_loras", default_loras)
            self.default_adapter_weights = kwargs.get("default_adapter_weights", default_adapter_weights)
            self.fuse_loras = kwargs.get("fuse_loras", False)
            self.lora_scale = kwargs.get("lora_scale", 1.0)
            self.use_prompt_travel = kwargs.get("use_prompt_travel", True)
            self.use_latent_travel = kwargs.get("use_latent_travel", True)
            self.prompt_travel_factor = kwargs.get("prompt_travel_factor", 0.5)
            self.latent_travel_factor = kwargs.get("latent_travel_factor", 0.5)
            self.latent_travel_method = kwargs.get("latent_travel_method", "slerp")
            self.debug_controlnet = kwargs.get("debug_controlnet", False)
            self.use_output_bg_removal = kwargs.get("use_output_bg_removal", False)
            self.controlnet_type = kwargs.get("controlnet_type", "depth")
            self.controlnet_scale = kwargs.get("controlnet_scale", 0.8)
            self.controlnet_start = kwargs.get("controlnet_start", 0.3)
            self.controlnet_end = kwargs.get("controlnet_end", 0.9)
            self.canny_low_threshold = kwargs.get("canny_low_threshold", 0.31)
            self.canny_high_threshold = kwargs.get("canny_high_threshold", 0.125)
    
    # Initialize the pipeline
    device = torch.device("cuda")
    torch_dtype = torch.float16
    pipeline = Pipeline(Args(sd_model=args.sd_model, onnx_model_dir=args.onnx_model_dir, unet_engine_path=args.unet_engine_path), device, torch_dtype)
    
    # Create input parameters
    params = Pipeline.InputParams(
        prompt="a red dog and a blue monkey",
        width=512,
        height=512,
        strength=1.0,
        steps=1,
        guidance_scale=1.1,
        controlnet_scale=0.8,
        controlnet_start=0.3,
        controlnet_end=0.9,
        use_latent_travel=True,
        use_prompt_travel=True,
    )

    print(f"[TIMING] Starting image generation loop")
    loop_start_time = time.time()
    total_iteration_time = 0
    
    for i in range(10):
        iteration_start = time.time()
        print(f"\n[TIMING] Starting iteration {i+1}/10")
        
        image = pipeline.predict(params, control_image=qr_image)
        
        iteration_time = time.time() - iteration_start
        total_iteration_time += iteration_time
        print(f"[TIMING] Iteration {i+1} completed in {iteration_time:.4f} seconds")
        
        save_start = time.time()
        image.save(f"server/assets/output_qr_code_{i}.png")
        print(f"[TIMING] Image saved: {time.time() - save_start:.4f} seconds")
    
    avg_iteration_time = total_iteration_time / 10
    print(f"\n[TIMING] All iterations completed in {time.time() - loop_start_time:.4f} seconds")
    print(f"[TIMING] Average iteration time: {avg_iteration_time:.4f} seconds")
    print(f"[TIMING] Total application time: {time.time() - app_start_time:.4f} seconds") 