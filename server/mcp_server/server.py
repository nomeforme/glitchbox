"""MCP server for Glitchbox generation control.

Exposes all GenerationControl gRPC RPCs as MCP tools and current state as a resource.
"""

import json
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .grpc_client import GlitchboxGRPCClient

client = GlitchboxGRPCClient()


@asynccontextmanager
async def lifespan(server):
    await client.connect()
    try:
        yield
    finally:
        await client.close()


mcp = FastMCP("glitchbox", lifespan=lifespan)


# ── Resource ────────────────────────────────────────────────────────────────


@mcp.resource("glitchbox://state/current")
async def current_generation_state() -> str:
    """Current state of all generation parameters (prompt, seed, guidance, acid effects, etc.)."""
    state = await client.get_current_state()
    return json.dumps(state, indent=2)


# ── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool()
async def set_prompt(
    prompt: str,
    target_prompt: str | None = None,
    prompt_travel_factor: float | None = None,
    transition_frames: int | None = None,
) -> str:
    """Set the generation prompt. Optionally specify a target_prompt for prompt travel,
    a prompt_travel_factor (0.0-1.0) for interpolation, and transition_frames (0-100)
    for smooth blending from the old prompt to the new one."""
    result = await client.set_prompt(
        prompt=prompt,
        target_prompt=target_prompt,
        prompt_travel_factor=prompt_travel_factor,
        transition_frames=transition_frames,
    )
    return json.dumps(result)


@mcp.tool()
async def set_generation_params(
    seed: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    strength: float | None = None,
    width: int | None = None,
    height: int | None = None,
    temporal_coherence: float | None = None,
    temporal_coherence_latent: float | None = None,
) -> str:
    """Adjust core generation parameters. Only provided fields are updated.
    - seed: random seed for generation
    - num_inference_steps: number of denoising steps
    - guidance_scale: classifier-free guidance scale
    - strength: denoising strength (0.0-1.0)
    - width/height: output resolution in pixels
    - temporal_coherence: noise blending for frame stability (0.0-1.0)
    - temporal_coherence_latent: latent feedback blending (0.0-1.0)"""
    result = await client.set_generation_params(
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        width=width,
        height=height,
        temporal_coherence=temporal_coherence,
        temporal_coherence_latent=temporal_coherence_latent,
    )
    return json.dumps(result)


@mcp.tool()
async def set_controlnet_params(
    controlnet_scale: float | None = None,
    controlnet_start: float | None = None,
    controlnet_end: float | None = None,
) -> str:
    """Adjust ControlNet conditioning parameters. Only provided fields are updated.
    - controlnet_scale: conditioning strength
    - controlnet_start: start point in the denoising process (0.0-1.0)
    - controlnet_end: end point in the denoising process (0.0-1.0)"""
    result = await client.set_controlnet_params(
        controlnet_scale=controlnet_scale,
        controlnet_start=controlnet_start,
        controlnet_end=controlnet_end,
    )
    return json.dumps(result)


@mcp.tool()
async def set_lora_params(
    lora_scale: float | None = None,
    pipe_index: int | None = None,
) -> str:
    """Adjust LoRA parameters. Only provided fields are updated.
    - lora_scale: LoRA weight scale
    - pipe_index: active pipeline/LoRA configuration index"""
    result = await client.set_lora_params(
        lora_scale=lora_scale,
        pipe_index=pipe_index,
    )
    return json.dumps(result)


@mcp.tool()
async def set_prompt_travel_params(
    enabled: bool | None = None,
    min_factor: float | None = None,
    max_factor: float | None = None,
    factor_increment: float | None = None,
    stabilize_duration: int | None = None,
    oscillate: bool | None = None,
    use_prompt_scheduler: bool | None = None,
    loop_prompts: bool | None = None,
) -> str:
    """Configure prompt travel/interpolation behavior. Only provided fields are updated.
    - enabled: toggle prompt travel on/off
    - min_factor/max_factor: interpolation range (0.0-1.0)
    - factor_increment: step size per frame
    - stabilize_duration: frames to hold at endpoints
    - oscillate: ping-pong between prompts vs one-way
    - use_prompt_scheduler: enable automatic prompt scheduling
    - loop_prompts: loop through prompt sequence"""
    result = await client.set_prompt_travel_params(
        enabled=enabled,
        min_factor=min_factor,
        max_factor=max_factor,
        factor_increment=factor_increment,
        stabilize_duration=stabilize_duration,
        oscillate=oscillate,
        use_prompt_scheduler=use_prompt_scheduler,
        loop_prompts=loop_prompts,
    )
    return json.dumps(result)


@mcp.tool()
async def set_acid_params(
    acid_strength: float | None = None,
    zoom_factor: float | None = None,
    x_shift: int | None = None,
    y_shift: int | None = None,
    coef_noise: float | None = None,
    do_acid_tracers: bool | None = None,
    acid_strength_foreground: float | None = None,
    do_acid_wobblers: bool | None = None,
    color_matching: float | None = None,
    do_human_seg: bool | None = None,
    do_blur: bool | None = None,
    brightness: float | None = None,
) -> str:
    """Configure acid visual effect processor. Only provided fields are updated.
    - acid_strength: overall effect intensity
    - zoom_factor: zoom amount per frame
    - x_shift/y_shift: horizontal/vertical pixel shift per frame
    - coef_noise: noise coefficient
    - do_acid_tracers: enable motion tracers
    - acid_strength_foreground: foreground effect intensity
    - do_acid_wobblers: enable wobble distortion
    - color_matching: color consistency (0.0-1.0)
    - do_human_seg: enable human segmentation masking
    - do_blur: enable blur effect
    - brightness: brightness adjustment"""
    result = await client.set_acid_params(
        acid_strength=acid_strength,
        zoom_factor=zoom_factor,
        x_shift=x_shift,
        y_shift=y_shift,
        coef_noise=coef_noise,
        do_acid_tracers=do_acid_tracers,
        acid_strength_foreground=acid_strength_foreground,
        do_acid_wobblers=do_acid_wobblers,
        color_matching=color_matching,
        do_human_seg=do_human_seg,
        do_blur=do_blur,
        brightness=brightness,
    )
    return json.dumps(result)


@mcp.tool()
async def switch_curation(curation_index: int) -> str:
    """Switch the active LoRA curation preset by index. This reloads the pipeline
    with the specified curation configuration (model, LoRA weights, default params)."""
    result = await client.switch_curation(curation_index)
    return json.dumps(result)


@mcp.tool()
async def get_current_state() -> str:
    """Get the full current state of all generation parameters including prompt,
    seed, guidance scale, ControlNet, LoRA, acid effects, prompt travel settings,
    and transition progress."""
    state = await client.get_current_state()
    return json.dumps(state, indent=2)


@mcp.tool()
async def batch_update(
    prompt_params: dict | None = None,
    generation_params: dict | None = None,
    controlnet_params: dict | None = None,
    lora_params: dict | None = None,
    prompt_travel_params: dict | None = None,
    acid_params: dict | None = None,
) -> str:
    """Update multiple parameter groups atomically in a single call.
    Each parameter group is a dict with the same fields as the corresponding
    individual set_* tool. Only provided groups are updated.

    Example: batch_update(
        prompt_params={"prompt": "a cosmic nebula"},
        generation_params={"guidance_scale": 1.5, "strength": 0.7},
        acid_params={"acid_strength": 0.3, "zoom_factor": 1.02}
    )"""
    result = await client.batch_update(
        prompt_params=prompt_params,
        generation_params=generation_params,
        controlnet_params=controlnet_params,
        lora_params=lora_params,
        prompt_travel_params=prompt_travel_params,
        acid_params=acid_params,
    )
    return json.dumps(result)
