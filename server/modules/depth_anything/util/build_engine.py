import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path, precision="fp32"):
    """
    Build TensorRT engine from ONNX file using TensorRT 10 API
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
    
    # Set precision mode
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build and serialize engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return None
    
    # Save engine to file
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    # Create runtime and load engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    return engine

# Example usage
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Construct absolute paths
    onnx_path = os.path.join(parent_dir, "models", "depth_anything_v2_vits.onnx")
    engine_path = os.path.join(parent_dir, "models", "depth_anything_v2_vits.trt")
    
    build_engine(onnx_path, engine_path, precision="fp16")