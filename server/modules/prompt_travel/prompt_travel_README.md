# Prompt Travel Module

This module provides functionality for interpolating between text prompts in the embedding space, allowing for smooth transitions between different prompts. It can be used to create animations or to explore the latent space between different textual descriptions.

## Features

- Linear interpolation between two text prompts
- Custom scheduling of multiple waypoint prompts
- Support for negative prompts
- Configurable number of interpolation steps
- Compatible with Stable Diffusion models and other models using CLIP text encoders

## Usage

### Basic Usage

```python
from prompt_travel import PromptTravel
from transformers import CLIPTextModel, CLIPTokenizer
import torch

# Initialize the tokenizer and text encoder
model_id = "runwayml/stable-diffusion-v1-5"  # Or any other SD model
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

# Initialize the PromptTravel module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_travel = PromptTravel(text_encoder, tokenizer)

# Interpolate between two prompts
prompt_from = "A serene landscape with mountains"
prompt_to = "A vibrant cityscape at night with neon lights"
negative_prompt = "blurry, low quality"

interpolated_embeds = prompt_travel.travel_between_prompts(
    prompt_from=prompt_from,
    prompt_to=prompt_to,
    device=device,
    num_steps=10,  # Number of interpolation steps
    negative_prompt=negative_prompt,
    do_classifier_free_guidance=True,
)

# The result is a list of (prompt_embeds, negative_prompt_embeds) tuples
# that can be used directly with a diffusion model
```

### Using with Multiple Waypoints

You can also create a sequence of embeddings that follow multiple waypoint prompts:

```python
# Define multiple waypoint prompts
prompts = [
    "A serene landscape with mountains",
    "A peaceful forest with a river",
    "A coastal scene with waves", 
    "A vibrant cityscape at night with neon lights"
]

# Define a schedule of positions (0.0 to 1.0) to sample from the prompts
# The schedule doesn't have to be evenly spaced
schedule = [0.0, 0.2, 0.3, 0.6, 1.0]

scheduled_embeds = prompt_travel.get_scheduled_prompts(
    prompts=prompts,
    schedule=schedule,
    device=device,
    negative_prompt=negative_prompt,
)
```

### Integration with Diffusion Models

This module is designed to be used with diffusion models. After generating the interpolated embeddings, you can use them directly with the UNet model in your diffusion pipeline:

```python
# Example of using the embeddings with a diffusion model
for prompt_embeds, negative_prompt_embeds in interpolated_embeds:
    # Combine the embeddings for classifier-free guidance
    combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    
    # Run the diffusion process with these embeddings
    # ... (your diffusion code here)
```

## Methods

### `encode_prompt`

Encodes a text prompt into embeddings compatible with the diffusion model.

### `interpolate_embeddings`

Performs linear interpolation between two sets of embeddings.

### `travel_between_prompts`

Creates a sequence of interpolated embeddings between two text prompts.

### `get_scheduled_prompts`

Creates embeddings based on a list of waypoint prompts and a schedule of positions.

## Example

See `prompt_travel_example.py` for a complete example of using this module. 