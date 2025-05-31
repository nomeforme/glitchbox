import torch
import torch.nn.functional as F


class PixelateProcessor:
    @staticmethod
    def apply(tensor, pixel_size=16, noise_strength=0.2, color_shift=0.001, scan_lines=True, add_glitch=False, use_pixelate=True, use_noise=True, use_color_shift=False):
        """
        Apply artistic glitch pixelation effect to a batch of tensors in range [-1, 1].
        Outputs a 4x larger image with pixelation and glitch effects.

        Parameters:
            tensor: torch.Tensor of shape (B, C, H, W) in range [-1, 1]
            pixel_size: int, pixel block size (larger = more pixelated)
            noise_strength: float, noise strength in [0, 1]
            color_shift: float, RGB channel shift amount in [0, 1]
            scan_lines: bool, whether to add scan line effect
            add_glitch: bool, whether to add random block shifts
            use_pixelate: bool, whether to apply pixelation effect

        Returns:
            torch.Tensor of shape (B, C, H*4, W*4) in range [-1, 1]
        """
        B, C, H, W = tensor.shape

        # Create pixelation effect if enabled
        if use_pixelate:
            downsampled = F.interpolate(
                tensor, 
                size=(H * 4 // pixel_size, W * 4 // pixel_size),
                mode='nearest'
            )
            pixelated = F.interpolate(
                downsampled,
                size=(H , W),
                mode='nearest'
            )
        else:
            pixelated = tensor

        # Add noise with random seed
        if use_noise:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(torch.randint(0, 1000000, (1,)).item())
            noise = (torch.rand(pixelated.shape, device=pixelated.device, generator=generator) - 0.5) * 2 * noise_strength
            glitched = pixelated + noise
        else:
            glitched = pixelated

        # Add color shift effect
        if use_color_shift:
            # Shift RGB channels slightly
            shift = int(W * 4 * color_shift)
            r, g, b = glitched.chunk(3, dim=1)
            r = torch.roll(r, shifts=shift, dims=3)
            b = torch.roll(b, shifts=-shift, dims=3)
            glitched = torch.cat([r, g, b], dim=1)

        # Add scan lines if enabled
        if scan_lines:
            scan_line = torch.ones_like(glitched)
            scan_line[:, :, ::2, :] = 0.95  # Create alternating dark lines
            glitched = glitched * scan_line

        # Add some random block shifts for glitch effect
        if add_glitch and pixel_size > 4:
            block_size = pixel_size * 2
            for i in range(0, H * 4, block_size):
                for j in range(0, W * 4, block_size):
                    if torch.rand(1, generator=generator) < 0.1:  # 10% chance to shift a block
                        shift = int(block_size * 0.5)
                        glitched[:, :, i:i+block_size, j:j+block_size] = torch.roll(
                            glitched[:, :, i:i+block_size, j:j+block_size],
                            shifts=shift,
                            dims=3
                        )

        # Clamp to [-1, 1]
        return torch.clamp(glitched, -1.0, 1.0) 