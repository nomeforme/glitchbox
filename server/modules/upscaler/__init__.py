from .processors.pil_processor import PilUpscalerProcessor, get_processor as get_pil_processor
from .processors.fast_srgan_processor import FastSRGANProcessor, get_processor as get_fast_srgan_processor
from .processors.omni_sr_processor import OmniSRProcessor, get_processor as get_omni_sr_processor
from .processors.rvsr_processor import RvsrUpscalerProcessor, get_processor as get_rvsr_processor
from .processors.dscf_sr_processor import DscfEfdnUpscalerProcessor, get_processor as get_dscf_sr_processor
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
    'OmniSRProcessor',
    'get_omni_sr_processor',
    'RvsrUpscalerProcessor',
    'get_rvsr_processor',
    'DscfEfdnUpscalerProcessor',
    'get_dscf_sr_processor',
    'create_upscaler',
    'get_processor'
] 