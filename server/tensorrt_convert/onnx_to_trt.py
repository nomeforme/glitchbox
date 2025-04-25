import argparse
import sys
import os

import tensorrt as trt


def convert_models(onnx_path: str, num_controlnet: int, output_path: str, fp16: bool = False, sd_xl: bool = False, text_hidden_size: int = None):
    """
    Function to convert models in stable diffusion controlnet pipeline into TensorRT format
    Example:
    python convert_stable_diffusion_controlnet_to_tensorrt.py
    --onnx_path path-to-models-stable_diffusion/RevAnimated-v1-2-2/unet/model.onnx
    --output_path path-to-models-stable_diffusion/RevAnimated-v1-2-2/unet/model.engine
    --fp16
    --num_controlnet 2
    Example for SD XL:
    python convert_stable_diffusion_controlnet_to_tensorrt.py
    --onnx_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.onnx
    --output_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.engine
    --fp16
    --num_controlnet 1
    --sd_xl
    Returns:
        unet/model.engine
        run test script in diffusers/examples/community
        python test_onnx_controlnet.py
        --sd_model danbrown/RevAnimated-v1-2-2
        --onnx_model_dir path-to-models-stable_diffusion/RevAnimated-v1-2-2
        --unet_engine_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.engine
        --qr_img_path path-to-qr-code-image
    """
    # Determine model type based on the path
    model_type = "unknown"
    if "unet" in onnx_path.lower():
        model_type = "unet"
    elif "vae_encoder" in onnx_path.lower():
        model_type = "vae_encoder"
    elif "vae_decoder" in onnx_path.lower():
        model_type = "vae_decoder"
    elif "text_encoder" in onnx_path.lower():
        model_type = "text_encoder"
    
    print(f"Detected model type: {model_type}")
    
    # UNET
    if sd_xl:
        batch_size = 1
        unet_in_channels = 4
        unet_sample_size = 64
        num_tokens = 77
        if text_hidden_size is None:
            text_hidden_size = 2048
        img_size = 512

        text_embeds_shape = (2 * batch_size, 1280)
        time_ids_shape = (2 * batch_size, 6)
    else:
        batch_size = 1
        unet_in_channels = 4
        unet_sample_size = 64
        num_tokens = 77
        if text_hidden_size is None:
            text_hidden_size = 768
        img_size = 512
        batch_size = 1

    print(f"Using text_hidden_size: {text_hidden_size}")

    latents_shape = (2 * batch_size, unet_in_channels, unet_sample_size, unet_sample_size)
    embed_shape = (2 * batch_size, num_tokens, text_hidden_size)
    controlnet_conds_shape = (num_controlnet, 2 * batch_size, 3, img_size, img_size)
    
    # VAE shapes
    vae_batch_size = 1
    vae_in_channels = 3
    vae_sample_size = 512
    vae_latent_channels = 4
    vae_latent_size = 64

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    TRT_RUNTIME = trt.Runtime(TRT_LOGGER)

    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    parse_success = onnx_parser.parse_from_file(onnx_path)
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit("ONNX model parsing failed")
    print("Load Onnx model done")

    # Print network input shapes for debugging
    print("Network inputs:")
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  {input_tensor.name}: {input_tensor.shape}")
        
        # If this is the encoder_hidden_states tensor, use its actual dimension
        if input_tensor.name == "encoder_hidden_states":
            # Get the actual dimension from the ONNX model
            actual_shape = input_tensor.shape
            if len(actual_shape) >= 3:
                actual_text_hidden_size = actual_shape[2]
                print(f"Actual text_hidden_size from ONNX model: {actual_text_hidden_size}")
                
                # Always use the actual dimension from the ONNX model
                text_hidden_size = actual_text_hidden_size
                print(f"Using text_hidden_size from ONNX model: {text_hidden_size}")
                
                # Update the embed_shape with the actual dimension
                embed_shape = (2 * batch_size, num_tokens, text_hidden_size)
                print(f"Updated embed_shape: {embed_shape}")
        
        # If this is the controlnet_conds tensor, use its actual dimension
        if input_tensor.name == "controlnet_conds":
            # Get the actual dimension from the ONNX model
            actual_shape = input_tensor.shape
            if len(actual_shape) >= 1:
                actual_num_controlnet = actual_shape[0]
                print(f"Actual num_controlnet from ONNX model: {actual_num_controlnet}")
                
                # Always use the actual dimension from the ONNX model
                num_controlnet = actual_num_controlnet
                print(f"Using num_controlnet from ONNX model: {num_controlnet}")
                
                # Update the controlnet_conds_shape with the actual dimension
                controlnet_conds_shape = (num_controlnet, 2 * batch_size, 3, img_size, img_size)
                print(f"Updated controlnet_conds_shape: {controlnet_conds_shape}")

    profile = TRT_BUILDER.create_optimization_profile()

    # Set shapes based on model type
    if model_type == "unet":
        print(f"Setting shape for 'sample': {latents_shape}")
        profile.set_shape("sample", latents_shape, latents_shape, latents_shape)
        
        print(f"Setting shape for 'encoder_hidden_states': {embed_shape}")
        profile.set_shape("encoder_hidden_states", embed_shape, embed_shape, embed_shape)
        
        print(f"Setting shape for 'controlnet_conds': {controlnet_conds_shape}")
        profile.set_shape("controlnet_conds", controlnet_conds_shape, controlnet_conds_shape, controlnet_conds_shape)
        
        if sd_xl:
            print(f"Setting shape for 'text_embeds': {text_embeds_shape}")
            profile.set_shape("text_embeds", text_embeds_shape, text_embeds_shape, text_embeds_shape)
            
            print(f"Setting shape for 'time_ids': {time_ids_shape}")
            profile.set_shape("time_ids", time_ids_shape, time_ids_shape, time_ids_shape)
    
    elif model_type == "vae_encoder":
        print(f"Setting shape for 'sample': ({vae_batch_size}, {vae_in_channels}, {vae_sample_size}, {vae_sample_size})")
        profile.set_shape("sample", 
                         (vae_batch_size, vae_in_channels, vae_sample_size, vae_sample_size),
                         (vae_batch_size, vae_in_channels, vae_sample_size, vae_sample_size),
                         (vae_batch_size, vae_in_channels, vae_sample_size, vae_sample_size))
    
    elif model_type == "vae_decoder":
        print(f"Setting shape for 'latent_sample': ({vae_batch_size}, {vae_latent_channels}, {vae_latent_size}, {vae_latent_size})")
        profile.set_shape("latent_sample", 
                         (vae_batch_size, vae_latent_channels, vae_latent_size, vae_latent_size),
                         (vae_batch_size, vae_latent_channels, vae_latent_size, vae_latent_size),
                         (vae_batch_size, vae_latent_channels, vae_latent_size, vae_latent_size))
    
    elif model_type == "text_encoder":
        print(f"Setting shape for 'input_ids': ({vae_batch_size}, 77)")
        profile.set_shape("input_ids", 
                         (vae_batch_size, 77),
                         (vae_batch_size, 77),
                         (vae_batch_size, 77))
    
    else:
        # For unknown model types, try to set shapes for all inputs
        print("Unknown model type, attempting to set shapes for all inputs")
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_name = input_tensor.name
            input_shape = input_tensor.shape
            
            # Create a concrete shape by replacing -1 with 1
            concrete_shape = tuple(1 if dim == -1 else dim for dim in input_shape)
            print(f"Setting shape for '{input_name}': {concrete_shape}")
            profile.set_shape(input_name, concrete_shape, concrete_shape, concrete_shape)

    config = TRT_BUILDER.create_builder_config()
    config.add_optimization_profile(profile)
    
    # TensorRT 10 compatibility - removed deprecated PreviewFeature
    # config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, True)
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Print TensorRT version for debugging
    print(f"TensorRT version: {trt.__version__}")

    plan = TRT_BUILDER.build_serialized_network(network, config)
    if plan is None:
        sys.exit("Failed building engine")
    print("Succeeded building engine")

    engine = TRT_RUNTIME.deserialize_cuda_engine(plan)

    ## save TRT engine
    with open(output_path, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the onnx checkpoint to convert",
    )

    parser.add_argument("--num_controlnet", type=int)

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")
    
    parser.add_argument("--text_hidden_size", type=int, default=None, 
                      help="Hidden size of the text encoder (default: 768 for SD, 2048 for SD XL)")

    args = parser.parse_args()

    convert_models(args.onnx_path, args.num_controlnet, args.output_path, args.fp16, args.sd_xl, args.text_hidden_size)
