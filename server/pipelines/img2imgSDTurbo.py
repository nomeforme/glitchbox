from diffusers import (
    AutoPipelineForImage2Image,
    AutoencoderTiny,
)
import torch

try:
    import intel_extension_for_pytorch as ipex  # type: ignore
except:
    pass

import psutil
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
import numpy as np


base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
default_negative_prompt = "blurry, low quality, render, 3D, oversaturated"
default_target_prompt = "a blue dog"

page_content = """
<h1 class="text-3xl font-bold">Real-Time SD-Turbo</h1>
<h3 class="text-xl font-bold">Image-to-Image</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://huggingface.co/stabilityai/sdxl-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SDXL Turbo</a>
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
        name: str = "img2img"
        title: str = "Image-to-Image SDXL"
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
        negative_prompt: str = Field(
            default_negative_prompt,
            title="Negative Prompt",
            field="textarea",
            id="negative_prompt",
            hide=True,
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
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
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

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model,
            safety_checker=None,
        )
        if args.taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            ).to(device)

        if args.sfast:
            from sfast.compilers.stable_diffusion_pipeline_compiler import (
                compile,
                CompilationConfig,
            )

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

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        if args.torch_compile:
            print("Running torch compile")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )

            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (768, 768))],
            )
        if args.compel:
            from compel import Compel

            self.pipe.compel_proc = Compel(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                truncate_long_prompts=True,
            )

        # Initialize PromptTravel for both text and latent interpolation
        from modules.prompt_travel.prompt_travel import PromptTravel
        self.pipe.prompt_travel = PromptTravel(
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        generator = torch.Generator(device=self.pipe.device).manual_seed(params.seed)
        target_generator = torch.Generator(device=self.pipe.device).manual_seed(params.seed + 1)

        steps = params.steps
        strength = params.strength
        if int(steps * strength) < 1:
            steps = math.ceil(1 / max(0.10, strength))

        prompt = params.prompt
        prompt_embeds = None
        negative_prompt_embeds = None

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
            # Convert PIL Image to tensor if needed
            if isinstance(params.image, Image.Image):
                # Convert PIL Image to tensor
                image_tensor = torch.from_numpy(np.array(params.image)).permute(2, 0, 1).unsqueeze(0).to(
                    device=self.pipe.device, 
                    dtype=self.pipe.dtype
                )
            else:
                image_tensor = params.image
                
            # Preprocess the image using the pipeline's image processor
            processed_image = self.pipe.image_processor.preprocess(
                image_tensor, 
                height=params.height, 
                width=params.width
            ).to(dtype=self.pipe.dtype)
                
            # Generate source latents
            source_latents = self.pipe.prompt_travel.prepare_latents(
                # processed_image,
                # self.pipe.scheduler.timesteps[:1],
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
            
            # # Interpolate between latents using the specified method
            # if hasattr(self.pipe, "prompt_travel") and self.pipe.prompt_travel is not None:
            #     latents = self.pipe.prompt_travel.interpolate_latents(
            #         source_latents,
            #         target_latents,
            #         getattr(params, "prompt_travel_factor", 0.5),
            #         getattr(params, "latent_travel_method", "slerp")
            #     )

        ######################################333

        # source_latents = torch.randn(
        #     (1, self.pipe.unet.config.in_channels, params.height // 8, params.width // 8),
        #     generator=generator,
        #     device=self.pipe.device,
        #     dtype=self.pipe.dtype,
        # )

        # target_latents = torch.randn(
        #     (1, self.pipe.unet.config.in_channels, params.height // 8, params.width // 8),
        #     generator=target_generator,
        #     device=self.pipe.device,
        #     dtype=self.pipe.dtype,
        # )


            latents = self.pipe.prompt_travel.interpolate_latents(
                source_latents,
                target_latents,
                getattr(params, "prompt_travel_factor", 0.5), # NOTE: should belatent travel factor latent_travel_factor
                getattr(params, "latent_travel_method", "slerp")
            )

            print("prompt travel factor: ", getattr(params, "prompt_travel_factor", 0.5))
            print("latent travel factor: ", getattr(params, "latent_travel_factor", 0.5))

        results = self.pipe(
            image=params.image,
            prompt_embeds=prompt_embeds,
            prompt=prompt,
            negative_prompt=params.negative_prompt,
            generator=generator,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=1.1,
            width=params.width,
            height=params.height,
            output_type="pil",
            latents=source_latents,
        )

        return results.images[0]
