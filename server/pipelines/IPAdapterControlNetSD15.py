from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    AutoencoderTiny,
    ControlNetModel,
    DiffusionPipeline,
    TCDScheduler,
)
from compel import Compel
import torch
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download
from pipelines.utils.canny_gpu import SobelOperator

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
from diffusers.utils import load_image
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Model paths
# controlnet_model = "thibaud/controlnet-sd21-openpose-diffusers"
# controlnet_model = "thibaud/controlnet-sd21-depth-diffusers"
base_model = "stabilityai/sd-turbo"
#base_model = "runwayml/stable-diffusion-v1-5"
taesd_model = "madebyollin/taesd"
# controlnet_model = "thibaud/controlnet-sd21-canny-diffusers"
controlnet_model = "lllyasviel/control_v11p_sd15_canny"
ip_adapter_model = "ostris/ip-composition-adapter"
ip_adapter_file_name = "ip_plus_composition_sd15.safetensors"

# Default values
default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"

# Example reference image for IP Adapter
ip_adapter_reference_image = load_image("/home/dream/Desktop/AI/Real-Time-Latent-Consistency-Model/server/assets/mountain.png").resize((512, 512))

page_content = """
<h1 class="text-3xl font-bold">ControlNet + IP-Adapter Combined Pipeline</h1>
<h3 class="text-xl font-bold">Edge Detection + Reference Image Guidance</h3>
<p class="text-sm">
    This combined pipeline uses both ControlNet Canny for edge detection and IP-Adapter for reference image composition,
    allowing for both structural guidance and visual reference in the generation process.
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
        name: str = "controlnet_ip_adapter"
        title: str = "ControlNet + IP Adapter"
        description: str = "Generates an image using both edge detection and reference image guidance"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            hide=True,
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            4, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.0,
            min=0,
            max=10,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            0.5,
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
        ip_adapter_scale: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.001,
            title="IP Adapter Scale",
            field="range",
            hide=True,
            id="ip_adapter_scale",
        )
        eta: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Eta",
            field="range",
            hide=True,
            id="eta",
        )
        debug_canny: bool = Field(
            False,
            title="Debug Canny",
            field="checkbox",
            hide=True,
            id="debug_canny",
        )
        use_reference_image: bool = Field(
            False,
            title="Use Default Reference",
            field="checkbox",
            hide=True,
            id="use_reference_image",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # Initialize ControlNet model
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype
        ).to(device)

        # Initialize Image encoder for IP-Adapter
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch_dtype,
        ).to(device)

        # Initialize the combined pipeline
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model,
            safety_checker=None,
            controlnet=controlnet_canny,
            torch_dtype=torch_dtype,
            variant="fp16",
            image_encoder=image_encoder,
        )

        # Load IP-Adapter weights
        self.pipe.load_ip_adapter(
            ip_adapter_model,
            subfolder="",
            weight_name=[ip_adapter_file_name],
            image_encoder_folder=None,
        )

        # Load Hyper-SD for faster inference
        # self.pipe.load_lora_weights(
        #     hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors")
        # )
        # self.pipe.fuse_lora()

        # Use tiny autoencoder if enabled
        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        # Use TCD scheduler for better results with fewer steps
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

        # Set up Canny edge detection with GPU acceleration
        self.canny_torch = SobelOperator(device=device)
        
        # Default IP-Adapter scale
        self.pipe.set_ip_adapter_scale([0.5])

        # Apply sfast compile if enabled
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

        # Apply onediff compile if enabled
        if args.onediff:
            print("\nRunning onediff compile\n")
            from onediff.infer_compiler import oneflow_compile

            self.pipe.unet = oneflow_compile(self.pipe.unet)
            self.pipe.vae.encoder = oneflow_compile(self.pipe.vae.encoder)
            self.pipe.vae.decoder = oneflow_compile(self.pipe.vae.decoder)
            self.pipe.controlnet = oneflow_compile(self.pipe.controlnet)

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device)
        
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Set up Compel for better prompt understanding
        if args.compel:
            self.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=False,
            )

        # Apply torch compile if enabled
        if args.torch_compile:
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )
            
            # Warmup
            self.pipe(
                prompt="warmup",
                image=Image.new("RGB", (512, 512)),
                control_image=Image.new("RGB", (512, 512)),
                ip_adapter_image=[Image.new("RGB", (512, 512))],
            )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.manual_seed(params.seed)
        
        # Process prompt with Compel if available
        prompt_embeds = None
        prompt = params.prompt
        negative_prompt = params.negative_prompt
        
        if hasattr(self, "compel_proc"):
            prompt_embeds = self.compel_proc(prompt)
            prompt = None

        # Generate Canny edge detection image
        control_image = self.canny_torch(
            params.image, params.canny_low_threshold, params.canny_high_threshold
        )
        
        # Adjust steps based on strength to ensure enough denoising
        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))
        
        # Set IP-Adapter scale
        self.pipe.set_ip_adapter_scale([params.ip_adapter_scale])
        
        # Select reference image - either use the input image or the default reference
        ip_adapter_image = ip_adapter_reference_image if params.use_reference_image else params.image

        # Run the combined pipeline
        results = self.pipe(
            image=params.image,
            control_image=control_image,
            prompt_embeds=prompt_embeds,
            prompt=prompt,
            negative_prompt=negative_prompt,
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
            ip_adapter_image=[ip_adapter_image],
            eta=params.eta,
        )
        
        result_image = results.images[0]
        
        # Optionally show debug view of Canny edges
        if params.debug_canny:
            # Paste control_image on top of result_image
            w0, h0 = (200, 200)
            control_image_pil = Image.fromarray((control_image.squeeze().cpu().numpy() * 255).astype('uint8'))
            control_image_pil = control_image_pil.resize((w0, h0))
            w1, h1 = result_image.size
            result_image.paste(control_image_pil, (w1 - w0, h1 - h0))

        return result_image