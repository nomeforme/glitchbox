from .processor import get_processor as get_pil_processor
from .fast_srgan_processor import get_processor as get_fast_srgan_processor

def create_upscaler(upscaler_type="fast_srgan", device=None, **kwargs):
    """
    Factory function to create an upscaler processor.
    
    Args:
        upscaler_type (str): Type of upscaler to create. Options are:
            - "pil": Uses PIL for basic upscaling
            - "fast_srgan": Uses Fast-SRGAN for high-quality upscaling (default)
        device (str, optional): Device to run the processor on ('cuda', 'mps', or 'cpu').
            If None, will automatically select the best available device.
        **kwargs: Additional arguments to pass to the upscaler processor.
            For Fast-SRGAN, these include:
            - model_path (str, optional): Path to the model weights file.
            - config_path (str, optional): Path to the model configuration file.
            
    Returns:
        An instance of the requested upscaler processor.
    """
    if upscaler_type.lower() == "pil":
        print("Using PIL upscaler")
        return get_pil_processor(device=device)
    elif upscaler_type.lower() == "fast_srgan":
        print("Using Fast-SRGAN upscaler")
        return get_fast_srgan_processor(device=device, **kwargs)
    else:
        print(f"Unknown upscaler type: {upscaler_type}. Using default: Fast-SRGAN")
        return get_fast_srgan_processor(device=device, **kwargs) 