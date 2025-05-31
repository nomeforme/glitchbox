import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorrt as trt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TensorRT_Builder')

def build_engine(onnx_file_path, engine_file_path, precision="fp32"):
    """
    Build TensorRT engine from ONNX file using TensorRT 10 API
    """
    logger.info(f"Starting TensorRT engine build process for {onnx_file_path}")
    logger.info(f"Target precision: {precision}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    logger.info("Creating TensorRT builder and network")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure builder
    logger.info("Configuring builder settings")
    config = builder.create_builder_config()
    workspace_size = 1 << 30  # 1GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    logger.info(f"Set workspace size to {workspace_size / (1024*1024*1024):.1f} GB")
    
    # Set precision mode
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            logger.info("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 precision requested but not supported on this platform")
    
    # Parse ONNX file
    logger.info("Parsing ONNX model")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                logger.error(f"Parser Error {error}: {parser.get_error(error)}")
            return None
    
    # Build and serialize engine
    logger.info("Building and serializing TensorRT engine")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None
    
    # Save engine to file
    logger.info(f"Saving TensorRT engine to {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    # Create runtime and load engine
    logger.info("Creating runtime and loading engine")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    if engine is not None:
        logger.info("Successfully built and loaded TensorRT engine")
    else:
        logger.error("Failed to load TensorRT engine")
    
    return engine

# Example usage
if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Construct absolute paths
    onnx_path = os.path.join(parent_dir, "models", "depth_anything_v2_vits.onnx")
    engine_path = os.path.join(parent_dir, "models", "depth_anything_v2_vits.trt")
    
    logger.info("Starting example usage of build_engine")
    build_engine(onnx_path, engine_path, precision="fp16")