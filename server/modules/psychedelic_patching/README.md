# Psychedelic Activation Patching Module

This module provides tools to patch activation functions in Stable Diffusion UNets with psychedelic-style transformations.

## Overview

The psychedelic patching system replaces standard activation functions (like SiLU/GELU) in the UNet with modified versions that can create interesting visual effects.

## Formula

```python
mix = (1-gamma)*SiLU(x/tau) + gamma*new_act(x/tau)
y   = (1-beta)*mix + beta*x
out = gain*y + bias
```

## Parameters

### Main Controls

- **`act`** (str): Activation function to use
  - Options: `"silu"`, `"gelu"`, `"gelu_tanh"`, `"relu"`, `"leakyrelu"`, `"mish"`, `"hswish"`, `"softsign"`, `"softplus"`, `"hardtanh"`
  - Default: `"silu"`

- **`tau`** (float): Temperature scaling
  - `> 1.0` flattens the activation curve (more linear)
  - `< 1.0` sharpens the activation curve (more nonlinear)
  - Default: `1.0` (no scaling)

- **`beta`** (float): Identity blend
  - `0.0` = pure activation function
  - `1.0` = pure identity (passthrough)
  - Default: `0.0`

- **`gamma`** (float): Activation blend
  - `0.0` = keep baseline SiLU
  - `1.0` = full replacement with new activation
  - Default: `1.0`

### Targeting Controls

- **`stages`** (List[str]): Which UNet stages to patch
  - Options: `["down"]`, `["mid"]`, `["up"]`, or any combination
  - Default: `["down", "mid", "up"]`

- **`start_idx`** (int): First resblock index to patch
  - Default: `0`

- **`end_idx`** (Optional[int]): Last resblock index to patch (inclusive)
  - Default: `None` (patch all blocks)

- **`patch_mlp`** (bool): Also patch attention MLP activations
  - Default: `False`

## Default Settings for "No Effect"

To ensure the patching has no visual effect (useful for testing):
- `tau=1.0, beta=0.0, gamma=1.0, act="silu"` (default settings)
- OR `beta=1.0` (identity passthrough regardless of other settings)

## Usage in img2imgStreamDiffusionXL

The module is integrated into the `img2imgStreamDiffusionXL` pipeline with all parameters hidden by default.

### Parameters in InputParams

- `use_psy_patching` (bool): Enable/disable psychedelic patching
- `psy_act` (str): Activation function
- `psy_tau` (float): Temperature scaling
- `psy_beta` (float): Identity blend
- `psy_gamma` (float): Activation blend
- `psy_stages` (str): Comma-separated stages (e.g., "down,mid,up")
- `psy_start_idx` (int): Start block index
- `psy_end_idx` (int): End block index (-1 = all)
- `psy_patch_mlp` (bool): Patch MLPs

### Example Usage

The patching is applied automatically based on parameters:

1. Enable patching by setting `use_psy_patching=True`
2. Configure parameters (all have safe defaults)
3. The UNet will be patched on the first frame
4. Setting `use_psy_patching=False` will restore the original UNet

## Implementation Details

- Patching is done per-pipe (supports multiple pipes with different patches)
- Patches are cached and reused until disabled or parameters change
- No performance overhead when disabled
- Thread-safe for real-time streaming

## Advanced: Direct API Usage

```python
from modules.psychedelic_patching import PatchCfg, patch_denoiser, restore_denoiser

# Create configuration
cfg = PatchCfg(
    stages=["mid", "up"],
    start_idx=0,
    start_idx_per_stage={'down': 0, 'mid': 0, 'up': 0},
    end_idx=None,
    end_idx_per_stage={'down': None, 'mid': None, 'up': None},
    act_kind="gelu",
    tau=1.5,
    beta=0.0,
    gamma=0.8,
    patch_attn_mlp=False,
    calibrate=False,
)

# Patch the UNet
unet = pipe.unet
count, patched, replacements, stage_depths = patch_denoiser(unet, cfg)
print(f"Patched {count} modules")

# ... use the pipeline ...

# Restore when done
restore_denoiser(replacements)
```

## Credits

Original implementation from [SD-Psychedelic-Lab](https://github.com/simpolism/SD-Psychedelic-Lab) by the glitchbox team.
