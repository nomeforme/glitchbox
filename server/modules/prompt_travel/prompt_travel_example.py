import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from prompt_travel import PromptTravel

def main():
    """
    Example of using the PromptTravel module to create interpolated prompt embeddings.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load text encoder and tokenizer from a model like stable diffusion
    model_id = "runwayml/stable-diffusion-v1-5"  # Or any other SD model
    
    # Initialize the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    
    # Initialize the PromptTravel module
    prompt_travel = PromptTravel(text_encoder, tokenizer)
    
    # Define test prompts
    prompt_from = "A serene landscape with mountains"
    prompt_to = "A vibrant cityscape at night with neon lights"
    
    # Set parameters
    num_steps = 5
    negative_prompt = "blurry, low quality"
    
    print(f"Creating {num_steps} interpolation steps between the prompts:")
    print(f"From: '{prompt_from}'")
    print(f"To: '{prompt_to}'")
    
    # Generate interpolated prompt embeddings
    interpolated_embeds = prompt_travel.travel_between_prompts(
        prompt_from=prompt_from,
        prompt_to=prompt_to,
        device=device,
        num_steps=num_steps,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
    )
    
    # Print information about the resulting embeddings
    print(f"\nGenerated {len(interpolated_embeds)} sets of embeddings:")
    for i, (embed, neg_embed) in enumerate(interpolated_embeds):
        print(f"Step {i}: Embedding shape = {embed.shape}, Negative embedding shape = {neg_embed.shape}")
    
    # Example of using the get_scheduled_prompts method
    print("\nTesting scheduled prompts with multiple waypoints:")
    prompts = [
        "A serene landscape with mountains",
        "A peaceful forest with a river",
        "A coastal scene with waves",
        "A vibrant cityscape at night with neon lights"
    ]
    
    # Create a schedule that's not evenly spaced
    schedule = [0.0, 0.2, 0.3, 0.6, 1.0]
    
    print(f"Using {len(prompts)} prompts with a custom schedule: {schedule}")
    
    scheduled_embeds = prompt_travel.get_scheduled_prompts(
        prompts=prompts,
        schedule=schedule,
        device=device,
        negative_prompt=negative_prompt,
    )
    
    print(f"\nGenerated {len(scheduled_embeds)} sets of embeddings:")
    for i, (embed, neg_embed) in enumerate(scheduled_embeds):
        print(f"Step {i}: Embedding shape = {embed.shape}, Negative embedding shape = {neg_embed.shape}")
    
if __name__ == "__main__":
    main() 