from diffusers import (
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
)
from pipelines.diffusers_pipelines.pipeline_controlnet import StableDiffusionControlNetPipeline
from compel import Compel
import torch
from pipelines.utils.canny_gpu import SobelOperator
import os
import glob
from typing import List, Dict, Optional, Union

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

# NOTE: this is a custom prompt travel module
from modules.prompt_travel.prompt_travel import PromptTravel
#
taesd_model = "madebyollin/taesd"
controlnet_model = "thibaud/controlnet-sd21-canny-diffusers"
# controlnet_model = "thibaud/controlnet-sd21-openpose-diffusers"
# controlnet_model = "thibaud/controlnet-sd21-depth-diffusers"
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
    "FKATwigs_A1-000038": "server/loras/FKATwigs_A1-000038.safetensors"
}

# Default LoRA to use
default_lora = "FKATwigs_A1-000038"
# default_lora = "pbarbarant/sd-sonio"

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
            [default_lora],
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
        debug_canny: bool = Field(
            False,
            title="Debug Canny",
            field="checkbox",
            hide=True,
            id="debug_canny",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype
        )
        self.pipes = {}

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet_canny,
            safety_checker=None,
            torch_dtype=torch_dtype,
        )

        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        if args.sfast:
            print("\nRunning sfast compile\n")
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.pipe = compile(self.pipe, config=config)

        if args.onediff:
            print("\nRunning onediff compile\n")
            from onediff.infer_compiler import oneflow_compile

            self.pipe.unet = oneflow_compile(self.pipe.unet)
            self.pipe.vae.encoder = oneflow_compile(self.pipe.vae.encoder)
            self.pipe.vae.decoder = oneflow_compile(self.pipe.vae.decoder)
            self.pipe.controlnet = oneflow_compile(self.pipe.controlnet)

        self.canny_torch = SobelOperator(device=device)

        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.compel:
            from compel import Compel

            self.pipe.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=True,
            )

        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        if args.torch_compile:
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )
            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (768, 768))],
                control_image=[Image.new("RGB", (768, 768))],
            )

        # Initialize PromptTravel for both text and latent interpolation
        self.pipe.prompt_travel = PromptTravel(
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        
        # Initialize LoRA-related attributes
        self.current_lora_models = []
        self.fuse_loras = False
        self.lora_scale = 1.0
        self.lora_weights_loaded = False
        
        # Load default LoRA during initialization
        print(f"Loading default LoRA: {default_lora}")
        self.pipe.load_lora_weights(lora_models[default_lora], adapter_name="default_lora")
        self.pipe.set_adapters(adapter_names=["default_lora"])
        self.current_lora_models = [default_lora]
        self.lora_weights_loaded = True

    def load_loras(self, lora_models_list: List[str], fuse_loras: bool = False, lora_scale: float = 1.0) -> None:
        """
        Load or update LoRA models for the pipeline.
        
        Args:
            lora_models_list: List of LoRA model names to load
            fuse_loras: Whether to fuse multiple LoRAs
            lora_scale: Scale factor for LoRA weights
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
            return
        
        # Check if we need to reload LoRAs
        if (selected_loras != self.current_lora_models) or (fuse_loras != self.fuse_loras) or (lora_scale != self.lora_scale):
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
                self.pipe.fuse_lora(adapter_names=adapter_names, lora_scale=lora_scale)
                # Unload the individual LoRAs after fusing
                self.pipe.unload_lora_weights()
                self.lora_weights_loaded = False
            else:
                # Set the first LoRA as active
                self.pipe.set_adapters(adapter_names=[f"lora_0"])
                self.lora_weights_loaded = True
            
            # Update current LoRA state
            self.current_lora_models = selected_loras
            self.fuse_loras = fuse_loras
            self.lora_scale = lora_scale

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        
        # Use target_seed if available, otherwise use seed+1
        target_seed = getattr(params, 'target_seed', None)
        if target_seed is None:
            target_seed = params.seed + 1
        target_generator = torch.Generator(device=self.pipe.device).manual_seed(target_seed)

        prompt = params.prompt
        prompt_embeds = None

        # Update LoRA models if needed
        self.load_loras(params.lora_models, params.fuse_loras, params.lora_scale)

        control_image = self.canny_torch(
            params.image, params.canny_low_threshold, params.canny_high_threshold
        )
        # control_image = openpose(params.image)
        # control_image = midas(params.image)
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
            source_latents = self.pipe.prompt_travel.prepare_latents(
                # processed_image,
                # self.pipe.scheduler.timesteps[:1],
                latents=result_latents,
                batch_size=1,
                # num_images_per_prompt=1,
                num_channels_latents=self.pipe.unet.config.in_channels,
                vae_scale_factor=self.pipe.vae_scale_factor,
                scheduler=self.pipe.scheduler,
                height=params.height,
                width=params.width,
                dtype=self.pipe.dtype,
                device=self.pipe.device,
                generator=generator,
            )
            
            # Generate target latents with a different seed
            target_latents = self.pipe.prompt_travel.prepare_latents(
                # processed_image,
                # self.pipe.scheduler.timesteps[:1],
                latents=result_latents,
                batch_size=1,
                # num_images_per_prompt=1,
                num_channels_latents=self.pipe.unet.config.in_channels,
                vae_scale_factor=self.pipe.vae_scale_factor,
                scheduler=self.pipe.scheduler,
                height=params.height,
                width=params.width,
                dtype=self.pipe.dtype,
                device=self.pipe.device,
                generator=target_generator,
            )
            
            latents = self.pipe.prompt_travel.interpolate_latents(
                source_latents,
                target_latents,
                getattr(params, "latent_travel_factor", 0.5), # NOTE: should belatent travel factor latent_travel_factor
                getattr(params, "latent_travel_method", "slerp")
            )

            print("prompt travel factor: ", getattr(params, "prompt_travel_factor", 0.5))
            print("latent travel factor: ", getattr(params, "latent_travel_factor", 0.5))

        results = self.pipe(
            #image=params.image,
            image=control_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
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

        # # print("result_latents: ", result_latents)
        # image = remote_decode(
        #     endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
        #     tensor=result_latents,
        #     scaling_factor=0.18215,
        # )
        # image.save(f"result_latents_{uuid.uuid4()}.png")
        # Store result_latents for next call if latent travel is enabled
        if getattr(params, "use_latent_travel", False):
            setattr(self, "stored_result_latents", result_latents)

        if params.debug_canny:
            # paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image = control_image.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image, (w1 - w0, h1 - h0))

        return result_image
