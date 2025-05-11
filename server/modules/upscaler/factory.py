from .processors.pil_processor import get_processor as get_pil_processor
from .processors.fast_srgan_processor import get_processor as get_fast_srgan_processor
from .processors.omni_sr_processor import get_processor as get_omni_sr_processor
from .processors.rvsr_processor import get_processor as get_rvsr_processor
from .processors.dscf_sr_processor import get_processor as get_dscf_sr_processor

def create_upscaler(upscaler_type="pil", device=None, **kwargs):
    """
    Factory function to create an upscaler processor.
    
    Args:
        upscaler_type (str): Type of upscaler to create. Options are:
            - "pil": Uses PIL for basic upscaling (default)
            - "fast_srgan": Uses Fast-SRGAN for high-quality upscaling
            - "omni_sr": Uses Omni-SR for high-quality upscaling
            - "rvsr": Uses RVSR for high-quality upscaling
            - "dscf_sr": Uses DSCF-SR for high-quality upscaling
        device (str, optional): Device to run the processor on ('cuda', 'mps', or 'cpu').
            If None, will automatically select the best available device.
        **kwargs: Additional arguments to pass to the upscaler processor.
            For Fast-SRGAN, these include:
            - model_path (str, optional): Path to the model weights file.
            - config_path (str, optional): Path to the model configuration file.
            For Omni-SR, these include:
            - model_path (str, optional): Path to the model weights file.
            For RVSR and DSCF-SR, these include:
            - model_path (str, optional): Path to the model weights file.
            
    Returns:
        An instance of the requested upscaler processor.
    """
    if upscaler_type.lower() == "pil":
        print("Using PIL upscaler")
        return get_pil_processor(device=device)
    elif upscaler_type.lower() == "fast_srgan":
        print("Using Fast-SRGAN upscaler")
        return get_fast_srgan_processor(device=device, **kwargs)
    elif upscaler_type.lower() == "omni_sr":
        print("Using Omni-SR upscaler")
        return get_omni_sr_processor(device=device, **kwargs)
    elif upscaler_type.lower() == "rvsr":
        print("Using RVSR upscaler")
        return get_rvsr_processor(device=device, **kwargs)
    elif upscaler_type.lower() == "dscf_sr":
        print("Using DSCF-SR upscaler")
        return get_dscf_sr_processor(device=device, **kwargs)
    else:
        print(f"Unknown upscaler type: {upscaler_type}. Using default: PIL")
        return get_pil_processor(device=device) 