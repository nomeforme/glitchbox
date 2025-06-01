import os
import argparse
from pathlib import Path
from onnx_to_trt import convert_models

def convert_all_models(base_dir: str, num_controlnet: int, fp16: bool = False, sd_xl: bool = False, text_hidden_size: int = None):
    """
    Convert all ONNX models in the specified directory to TensorRT format.
    
    Args:
        base_dir: Base directory containing the ONNX models
        num_controlnet: Number of controlnet models
        fp16: Whether to use FP16 precision
        sd_xl: Whether the models are from SD XL
        text_hidden_size: Hidden size of the text encoder
    """
    # Convert to absolute path
    base_dir = Path(base_dir).resolve()
    print(f"Using base directory: {base_dir}")
    
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")
    
    # Define the model components and their expected ONNX file names
    model_components = {
        'unet': 'model.onnx',
        'vae_decoder': 'model.onnx',
        'vae_encoder': 'model.onnx',
        'text_encoder': 'model.onnx'
    }
    
    for component, onnx_file in model_components.items():
        onnx_path = base_dir / component / onnx_file
        print(f"\nProcessing {component}...")
        print(f"Looking for file at: {onnx_path}")
        
        if not onnx_path.exists():
            print(f"Warning: {onnx_path} does not exist, skipping...")
            # List contents of the component directory to help debug
            component_dir = base_dir / component
            if component_dir.exists():
                print(f"Contents of {component_dir}:")
                for file in component_dir.iterdir():
                    print(f"  - {file.name}")
            continue
            
        if not onnx_path.is_file():
            print(f"Warning: {onnx_path} exists but is not a file, skipping...")
            continue
            
        output_path = onnx_path.parent / f"{component}.engine"
        print(f"Input: {onnx_path}")
        print(f"Output: {output_path}")
        
        try:
            convert_models(
                str(onnx_path),
                num_controlnet,
                str(output_path),
                fp16=fp16,
                sd_xl=sd_xl,
                text_hidden_size=text_hidden_size
            )
            print(f"Successfully converted {component}")
        except Exception as e:
            print(f"Error converting {component}: {str(e)}")
            print(f"File permissions: {oct(onnx_path.stat().st_mode)[-3:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all ONNX models to TensorRT format")
    parser.add_argument("--base_dir", type=str, default="tensorrt_convert/onnx_models",
                      help="Base directory containing the ONNX models")
    parser.add_argument("--num_controlnet", type=int, default=1,
                      help="Number of controlnet models")
    parser.add_argument("--fp16", action="store_true", default=False,
                      help="Use FP16 precision")
    parser.add_argument("--sd_xl", action="store_true", default=False,
                      help="Use SD XL configuration")
    parser.add_argument("--text_hidden_size", type=int, default=None,
                      help="Hidden size of the text encoder (default: 768 for SD, 2048 for SD XL)")
    
    args = parser.parse_args()
    
    try:
        convert_all_models(
            args.base_dir,
            args.num_controlnet,
            args.fp16,
            args.sd_xl,
            args.text_hidden_size
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1) 