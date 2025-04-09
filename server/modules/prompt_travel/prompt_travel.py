import torch
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer


class PromptTravel:
    """
    A module for interpolating between two text prompts in the embedding space.
    
    This allows for smooth transitions between different prompts by linearly 
    interpolating their embeddings.
    """
    
    def __init__(
        self,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        """
        Initialize the PromptTravel module.
        
        Args:
            text_encoder: The CLIP text encoder model
            tokenizer: The CLIP tokenizer
        """
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the prompt into text encoder hidden states.
        
        This is adapted from the StableDiffusionControlNetImg2ImgPipeline.encode_prompt method.
        
        Args:
            prompt: The prompt to encode
            device: The device to use
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            negative_prompt: The negative prompt to use
            prompt_embeds: Pre-generated text embeddings
            negative_prompt_embeds: Pre-generated negative text embeddings
            lora_scale: LoRA scale factor
            clip_skip: Number of layers to skip in CLIP
            
        Returns:
            A tuple of (prompt_embeds, negative_prompt_embeds)
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if prompt_embeds is None:
            # Tokenize the text
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            
            # Check if truncation occurred and potentially warn (omitted for brevity)
            
            # Get attention mask if needed
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
                
            # Process with or without clip_skip
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                # Get the hidden state from the desired layer
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # Apply the final layer norm
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)
                
        # Ensure correct dtype
        prompt_embeds_dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        
        # Duplicate embeddings for each image per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        # Get unconditional embeddings for classifier-free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt
                
            # Tokenize negative prompt
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
                
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            
        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings for each generation per prompt
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            
        return prompt_embeds, negative_prompt_embeds
    
    def interpolate_embeddings(
        self,
        embeds_from: torch.Tensor,
        embeds_to: torch.Tensor,
        factor: float,
    ) -> torch.Tensor:
        """
        Linearly interpolate between two embeddings.
        
        Args:
            embeds_from: Source embeddings
            embeds_to: Target embeddings
            factor: Interpolation factor (0.0 = source, 1.0 = target)
            
        Returns:
            Interpolated embeddings
        """
        # Ensure factor is within [0, 1]
        factor = max(0.0, min(1.0, factor))
        
        # Linear interpolation: (1-t)*A + t*B
        return (1 - factor) * embeds_from + factor * embeds_to
    
    def travel_between_prompts(
        self,
        prompt_from: str,
        prompt_to: str,
        device: torch.device,
        num_steps: int = 10,
        interpolation_method: str = "linear",
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        clip_skip: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Create a series of prompt embeddings that interpolate between two prompts.
        
        Args:
            prompt_from: The source prompt
            prompt_to: The target prompt
            device: The device to use
            num_steps: Number of interpolation steps
            interpolation_method: Method of interpolation (currently only 'linear' supported)
            negative_prompt: Negative prompt to use (same for all steps)
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            clip_skip: Number of layers to skip in CLIP
            
        Returns:
            A list of tuples containing (prompt_embeds, negative_prompt_embeds) for each step
        """
        # Encode the source and target prompts
        embeds_from, neg_embeds = self.encode_prompt(
            prompt_from,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            clip_skip=clip_skip,
        )
        
        embeds_to, _ = self.encode_prompt(
            prompt_to,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            clip_skip=clip_skip,
        )
        
        # Create interpolated embeddings
        result = []
        
        for i in range(num_steps):
            if num_steps == 1:
                factor = 1.0
            else:
                factor = i / (num_steps - 1)
                
            if interpolation_method == "linear":
                interpolated_embeds = self.interpolate_embeddings(embeds_from, embeds_to, factor)
            else:
                raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
                
            result.append((interpolated_embeds, neg_embeds))
            
        return result
    
    def get_scheduled_prompts(
        self,
        prompts: List[str],
        schedule: List[float],
        device: torch.device,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        clip_skip: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Create prompt embeddings according to a schedule of prompts.
        
        Args:
            prompts: List of prompts to transition between
            schedule: List of positions (0.0 to 1.0) to sample from the prompts
            device: The device to use
            negative_prompt: Negative prompt to use (same for all steps)
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            clip_skip: Number of layers to skip in CLIP
            
        Returns:
            A list of tuples containing (prompt_embeds, negative_prompt_embeds) for each step
        """
        if len(prompts) < 2:
            raise ValueError("At least two prompts are needed for scheduling")
            
        # Encode all prompts
        all_embeds = []
        neg_embeds = None
        
        for prompt in prompts:
            embeds, neg = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                clip_skip=clip_skip,
            )
            all_embeds.append(embeds)
            if neg_embeds is None:
                neg_embeds = neg
                
        # Generate embeddings according to schedule
        result = []
        
        for pos in schedule:
            # Ensure pos is within [0, 1]
            pos = max(0.0, min(1.0, pos))
            
            # Calculate which segment we're in
            segment_length = 1.0 / (len(prompts) - 1)
            segment_idx = int(pos / segment_length)
            
            # Handle edge case for pos = 1.0
            if segment_idx == len(prompts) - 1:
                segment_idx = len(prompts) - 2
                local_pos = 1.0
            else:
                local_pos = (pos - segment_idx * segment_length) / segment_length
                
            # Interpolate within this segment
            interpolated_embeds = self.interpolate_embeddings(
                all_embeds[segment_idx],
                all_embeds[segment_idx + 1],
                local_pos
            )
            
            result.append((interpolated_embeds, neg_embeds))
            
        return result

    def slerp(self, x0: torch.Tensor, x1: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Perform spherical linear interpolation (SLERP) between two latent vectors.
        
        Args:
            x0: Source latent tensor
            x1: Target latent tensor
            factor: Interpolation factor (0.0 = source, 1.0 = target)
            
        Returns:
            Interpolated latent tensor
        """
        # Ensure factor is within [0, 1]
        factor = max(0.0, min(1.0, factor))
        
        # If tensors are identical or factor is at extremes, return early
        if torch.allclose(x0, x1) or factor == 0.0:
            return x0
        if factor == 1.0:
            return x1
        
        # Normalize the vectors
        x0_norm = x0 / torch.norm(x0, dim=-1, keepdim=True)
        x1_norm = x1 / torch.norm(x1, dim=-1, keepdim=True)
        
        # Compute the cosine of the angle between the vectors
        dot_product = torch.sum(x0_norm * x1_norm, dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot_product)
        
        # Handle edge cases where vectors are nearly parallel
        if torch.allclose(omega, torch.zeros_like(omega)):
            return x0 * (1.0 - factor) + x1 * factor
        
        # Perform SLERP
        sin_omega = torch.sin(omega)
        x0_factor = torch.sin((1.0 - factor) * omega) / sin_omega
        x1_factor = torch.sin(factor * omega) / sin_omega
        
        return x0 * x0_factor + x1 * x1_factor

    def interpolate_latents(
        self,
        latents_from: torch.Tensor,
        latents_to: torch.Tensor,
        factor: float,
        interpolation_method: str = "slerp"
    ) -> torch.Tensor:
        """
        Interpolate between two latent vectors.
        
        Args:
            latents_from: Source latent tensor
            latents_to: Target latent tensor
            factor: Interpolation factor (0.0 = source, 1.0 = target)
            interpolation_method: Method of interpolation ('linear' or 'slerp')
            
        Returns:
            Interpolated latent tensor
        """
        if interpolation_method == "slerp":
            # Reshape latents to 2D for SLERP
            original_shape = latents_from.shape
            latents_from_2d = latents_from.reshape(latents_from.shape[0], -1)
            latents_to_2d = latents_to.reshape(latents_to.shape[0], -1)
            
            # Apply SLERP
            interpolated = self.slerp(latents_from_2d, latents_to_2d, factor)
            
            # Restore original shape
            return interpolated.reshape(original_shape)
        else:
            # Linear interpolation
            return self.interpolate_embeddings(latents_from, latents_to, factor) 