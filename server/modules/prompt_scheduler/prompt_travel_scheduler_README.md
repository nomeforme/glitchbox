# Prompt Travel Scheduler

The Prompt Travel Scheduler is a module that automates the interpolation between source and target prompts by providing a deterministic schedule for the prompt travel factor.

## Overview

This scheduler works similarly to the oscillators for zoom and shift effects but applies to prompt interpolation. It allows for smooth transitions between prompts based on a configurable schedule, making it useful for creating animation-like effects with prompt travel.

## Features

- Automatic oscillation between minimum and maximum prompt travel factors
- Configurable increment amount for smooth or rapid transitions
- Stabilization periods at minimum and maximum values
- Support for one-way transitions (non-oscillating mode)
- Debug output for monitoring factor changes

## Usage

### Command Line Arguments

The scheduler can be configured through command line arguments when starting the server:

```bash
python main.py \
  --use-prompt-travel \
  --use-prompt-travel-scheduler \
  --prompt-travel-min-factor 0.0 \
  --prompt-travel-max-factor 1.0 \
  --prompt-travel-factor-increment 0.025 \
  --prompt-travel-stabilize-duration 3 \
  --prompt-travel-oscillate
```

### Frontend Settings

The scheduler can also be controlled from the frontend by sending the following parameters in the `acid_settings` object:

```javascript
{
  "acid_settings": {
    "use_prompt_travel_scheduler": true,
    "prompt_travel_factor_increment": 0.025,
    "prompt_travel_min_factor": 0.0,
    "prompt_travel_max_factor": 1.0,
    "prompt_travel_oscillate": true
  }
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_prompt_travel_scheduler` | boolean | `false` | Enable/disable the scheduler |
| `prompt_travel_min_factor` | float | `0.0` | Minimum prompt travel factor (0.0-1.0) |
| `prompt_travel_max_factor` | float | `1.0` | Maximum prompt travel factor (0.0-1.0) |
| `prompt_travel_factor_increment` | float | `0.025` | Amount to change factor per update |
| `prompt_travel_stabilize_duration` | int | `3` | Number of iterations to pause at min/max |
| `prompt_travel_oscillate` | boolean | `true` | Whether to oscillate or go one-way |

## Integration with LCM Model

The scheduler automatically updates the `prompt_travel_factor` parameter used by the embeddings service for prompt interpolation. It provides a hands-free way to create dynamic prompt transitions without manually adjusting the factor.

## Example Use Case

1. Set your base prompt (e.g., "a majestic mountain")
2. Set your target prompt (e.g., "a tranquil beach")
3. Enable the prompt travel scheduler
4. Watch as the model automatically transitions back and forth between the two prompts

This creates an animation-like effect as the model smoothly interpolates between the two concepts. 