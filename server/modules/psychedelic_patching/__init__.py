"""
Psychedelic UNet Activation Patching Module

This module provides activation function patching for Stable Diffusion UNets,
enabling psychedelic-style transformations of the denoising network's activation functions.
"""

from .patching import (
    PsyAct,
    PatchCfg,
    patch_denoiser,
    restore_denoiser,
    make_base_act,
)

__all__ = [
    "PsyAct",
    "PatchCfg",
    "patch_denoiser",
    "restore_denoiser",
    "make_base_act",
]
