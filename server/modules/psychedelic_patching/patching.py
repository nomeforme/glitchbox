"""
Psychedelic UNet Activation Patching

This module provides tools to patch activation functions in Stable Diffusion UNets
with psychedelic-style transformations.

Core formula:
    mix = (1-gamma)*SiLU(x/tau) + gamma*new_act(x/tau)
    y   = (1-beta)*mix + beta*x
    out = gain*y + bias

Parameters:
    - act: Activation function (silu, gelu, relu, leakyrelu, mish, etc.)
    - tau: Temperature scaling (>1.0 flattens, <1.0 sharpens)
    - beta: Identity blend (0=pure activation, 1=passthrough)
    - gamma: Blend SiLU→new activation (0=keep SiLU, 1=full replacement)
    - stages: Which UNet stages to patch ("down", "mid", "up")
    - start_idx/end_idx: Block range to patch
    - patch_mlp: Also patch attention MLP activations
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------- Activation Library ----------------------

def make_base_act(kind: str) -> nn.Module:
    """Create an activation function module by name."""
    kind = kind.lower()
    if kind == "silu":        return nn.SiLU()
    if kind == "gelu":        return nn.GELU()  # erf approx
    if kind == "gelu_tanh":   return nn.GELU(approximate="tanh")
    if kind == "relu":        return nn.ReLU()
    if kind == "leakyrelu":   return nn.LeakyReLU(0.05)
    if kind == "mish":        return nn.Mish()
    if kind == "hswish":
        class HSwish(nn.Module):
            def forward(self, x): return x * (torch.clamp(x + 3, 0, 6) / 6)
        return HSwish()
    if kind == "softsign":    return nn.Softsign()
    if kind == "softplus":    return nn.Softplus(beta=1.0, threshold=20.0)
    if kind == "hardtanh":    return nn.Hardtanh(min_val=-1.0, max_val=1.0)
    raise ValueError(f"Unknown act kind: {kind}")


# ---------------------- Psychedelic Activation ----------------------

class PsyAct(nn.Module):
    """
    Psychedelic-style activation wrapper with 3 knobs + optional affine correction:
      mix = (1-gamma)*SiLU(x/tau) + gamma*new_act(x/tau)
      y   = (1-beta)*mix + beta*x
      out = gain*y + bias
    """
    def __init__(self, baseline: nn.Module, new_act: nn.Module,
                 tau: float = 1.0, beta: float = 0.0, gamma: float = 1.0,
                 gain: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.baseline = baseline
        self.new_act  = new_act
        self.tau      = float(tau)
        self.beta     = float(beta)
        self.gamma    = float(gamma)
        self.register_buffer("gain_buf", torch.tensor(float(gain)), persistent=False)
        self.register_buffer("bias_buf", torch.tensor(float(bias)), persistent=False)
        self._active = True

    def set_active(self, active: bool) -> None:
        self._active = bool(active)

    @property
    def is_active(self) -> bool:
        return self._active

    def forward(self, x):
        if not self._active:
            return self.baseline(x)
        z = x / self.tau
        mix = (1.0 - self.gamma) * self.baseline(z) + self.gamma * self.new_act(z)
        y = (1.0 - self.beta) * mix + self.beta * x
        return self.gain_buf * y + self.bias_buf


# ---------------------- Patching Configuration ----------------------

@dataclass
class PatchCfg:
    """Configuration for psychedelic UNet patching."""
    stages: List[str]
    start_idx: int
    start_idx_per_stage: Dict[str, int]
    end_idx: Optional[int]
    end_idx_per_stage: Dict[str, Optional[int]]
    act_kind: str
    tau: float
    beta: float
    gamma: float
    patch_attn_mlp: bool
    calibrate: bool = False


# ---------------------- UNet Patching Functions ----------------------

def stage_of_name(name: str) -> Optional[str]:
    """Determine which UNet stage a module belongs to based on its name."""
    if name.startswith("down_blocks.") or ".down_blocks." in name:
        return "down"
    if name.startswith("up_blocks.")   or ".up_blocks."   in name:
        return "up"
    if name.startswith("mid_block.")   or ".mid_block."   in name:
        return "mid"
    return None


def _iter_leaf_acts(module: nn.Module):
    """Yield (parent, attr, child, fullname) for all leaf activation modules."""
    for fullname, m in module.named_modules():
        parent_name = fullname.rsplit('.', 1)[0] if '.' in fullname else ''
        parent = dict(module.named_modules()).get(parent_name, module if parent_name == '' else None)
        if parent is None:
            continue
        for attr, child in list(parent._modules.items()):
            if child is m and len(list(child.children())) == 0:
                yield parent, attr, child, fullname


def build_psy_factory(act_kind: str, tau: float, beta: float, gamma: float) -> Callable[[], nn.Module]:
    """Factory function to create PsyAct modules with given parameters."""
    def factory():
        baseline = nn.SiLU()
        new_act  = make_base_act(act_kind)
        return PsyAct(baseline, new_act, tau=tau, beta=beta, gamma=gamma)
    return factory


def patch_denoiser(
    denoiser: nn.Module,
    cfg: PatchCfg
) -> Tuple[int, Dict[str, PsyAct], List[Tuple[nn.Module, str, nn.Module]], Dict[str, int]]:
    """
    Patch activation functions in a UNet denoiser module.

    Args:
        denoiser: The UNet module to patch
        cfg: Patching configuration

    Returns:
        Tuple of:
        - count: Number of modules patched
        - patched: Dict mapping module names to PsyAct instances
        - replacements: List of (parent, attr, original) for restoration
        - stage_depths: Dict mapping stage names to their depths
    """
    try:
        from diffusers.models.resnet import ResnetBlock2D
    except ImportError:
        # Fallback for older diffusers versions
        try:
            from diffusers.models.unet_2d_blocks import ResnetBlock2D
        except ImportError:
            ResnetBlock2D = None

    count = 0
    patched: Dict[str, PsyAct] = {}
    idx_by_stage = {"down": -1, "mid": -1, "up": -1}
    resnet_index_map: Dict[str, Dict[str, int]] = {"down": {}, "mid": {}, "up": {}}
    replacements: List[Tuple[nn.Module, str, nn.Module]] = []
    patched_fullnames = set()
    stage_depths: Dict[str, int] = {"down": 0, "mid": 0, "up": 0}

    def stage_allowed(stage: Optional[str]) -> bool:
        if "all" in cfg.stages:
            return True
        return stage in cfg.stages

    # Count ResNet blocks and replace SiLU inside them
    if ResnetBlock2D is not None:
        for fullname, m in denoiser.named_modules():
            st = stage_of_name(fullname)
            if isinstance(m, ResnetBlock2D) and st in stage_depths:
                stage_depths[st] += 1
            if isinstance(m, ResnetBlock2D) and stage_allowed(st):
                stage = st or "all"
                if stage in idx_by_stage:
                    idx_by_stage[stage] += 1
                    resnet_index_map[stage][fullname] = idx_by_stage[stage]
                    start_threshold = cfg.start_idx_per_stage.get(stage, cfg.start_idx)
                    end_threshold = cfg.end_idx_per_stage.get(stage, cfg.end_idx)
                    if stage in cfg.stages and idx_by_stage[stage] < start_threshold:
                        continue
                    if stage in cfg.stages and end_threshold is not None and idx_by_stage[stage] > end_threshold:
                        continue
                for attr, child in list(m._modules.items()):
                    if isinstance(child, nn.SiLU):
                        psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                        m._modules[attr] = psy
                        fullname_attr = f"{fullname}.{attr}" if fullname else attr
                        patched[fullname_attr] = psy
                        replacements.append((m, attr, child))
                        patched_fullnames.add(fullname_attr)
                        count += 1

    def resolve_resnet_index(stage: Optional[str], path: str) -> Optional[int]:
        if stage not in resnet_index_map:
            return None
        current = path
        while current:
            if current in resnet_index_map[stage]:
                return resnet_index_map[stage][current]
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        return None

    # General fallback: patch remaining leaf SiLU/GELU modules within allowed scopes
    for parent, attr, child, fullname in _iter_leaf_acts(denoiser):
        st = stage_of_name(fullname)
        if not stage_allowed(st):
            continue
        fullname_attr = fullname
        if fullname_attr in patched_fullnames:
            continue
        if isinstance(child, (nn.GELU, nn.SiLU)):
            start_threshold = cfg.start_idx_per_stage.get(st, cfg.start_idx)
            end_threshold = cfg.end_idx_per_stage.get(st, cfg.end_idx)
            if st in cfg.stages:
                res_idx = resolve_resnet_index(st, fullname)
                if res_idx is not None:
                    if res_idx < start_threshold:
                        continue
                    if end_threshold is not None and res_idx > end_threshold:
                        continue
                elif start_threshold > 0:
                    continue
            psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
            parent._modules[attr] = psy
            patched[fullname_attr] = psy
            replacements.append((parent, attr, child))
            patched_fullnames.add(fullname_attr)
            count += 1

    # Optionally patch attention MLP activations (often GELU/SiLU)
    if cfg.patch_attn_mlp:
        for parent, attr, child, fullname in _iter_leaf_acts(denoiser):
            if fullname in patched_fullnames:
                continue
            st = stage_of_name(fullname)
            if not stage_allowed(st):
                continue
            if isinstance(child, (nn.GELU, nn.SiLU)):
                start_threshold = cfg.start_idx_per_stage.get(st, cfg.start_idx)
                end_threshold = cfg.end_idx_per_stage.get(st, cfg.end_idx)
                if st in cfg.stages:
                    res_idx = resolve_resnet_index(st, fullname)
                    if res_idx is not None:
                        if res_idx < start_threshold:
                            continue
                        if end_threshold is not None and res_idx > end_threshold:
                            continue
                    elif start_threshold > 0:
                        continue
                psy = build_psy_factory(cfg.act_kind, cfg.tau, cfg.beta, cfg.gamma)()
                parent._modules[attr] = psy
                patched[fullname] = psy
                replacements.append((parent, attr, child))
                patched_fullnames.add(fullname)
                count += 1

    return count, patched, replacements, stage_depths


def restore_denoiser(replacements: List[Tuple[nn.Module, str, nn.Module]]):
    """Restore modules swapped by patch_denoiser back to their original instances."""
    for parent, attr, original in replacements:
        parent._modules[attr] = original
