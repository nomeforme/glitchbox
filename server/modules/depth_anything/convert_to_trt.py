import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
    """
    Build a TensorRT engine from an ONNX model.
    
    Args:
        onnx_file_path (str): Path to the ONNX model file
        engine_file_path (str): Path to save the TensorRT engine
        fp16_mode (bool): Whether to use FP16 precision
    """
    # Create logger
    logger = trt.Logger()
    
    # Create builder
    builder = trt.Builder(logger)
    
    # Create network definition
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Enable FP16 if requested and supported
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build and serialize engine
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Save engine
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_file_path}")

if __name__ == "__main__":
    # Paths
    onnx_path = "models/depth_anything_v2_vits.onnx"
    engine_path = "models/depth_anything_v2_vits.trt"
    
    # Convert to absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, onnx_path)
    engine_path = os.path.join(current_dir, engine_path)
    
    # Build engine
    build_engine(onnx_path, engine_path, fp16_mode=True) 