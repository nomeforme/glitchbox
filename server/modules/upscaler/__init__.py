from .processor import PilUpscalerProcessor, get_processor as get_pil_processor
from .fast_srgan_processor import FastSRGANProcessor, get_processor as get_fast_srgan_processor
from .factory import create_upscaler

# Alias for create_upscaler for backward compatibility
def get_processor(device=None, **kwargs):
    """
    Alias for create_upscaler for backward compatibility.
    
    Args:
        device (str, optional): Device to run the processor on ('cuda', 'mps', or 'cpu').
            If None, will automatically select the best available device.
        **kwargs: Additional arguments to pass to the upscaler processor.
            
    Returns:
        An instance of the upscaler processor.
    """
    return create_upscaler(device=device, **kwargs)

__all__ = [
    'PilUpscalerProcessor', 
    'get_pil_processor',
    'FastSRGANProcessor',
    'get_fast_srgan_processor',
    'create_upscaler',
    'get_processor'
] 