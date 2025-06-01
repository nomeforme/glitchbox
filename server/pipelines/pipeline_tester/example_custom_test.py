#!/usr/bin/env python3
"""
Example Custom Pipeline Test

This script demonstrates how to customize the pipeline tester for specific use cases,
such as testing specific prompts, using custom control images, or testing specific
parameter combinations.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from types import SimpleNamespace

# Add the server directory to the Python path
server_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(server_dir))

# Import the base tester
from pipeline_tester.test_controlnet_pipeline import (
    create_test_config, 
    Pipeline, 
    device, 
    torch_dtype
)


def create_custom_control_image(width=640, height=480):
    """Create a custom control image - example with geometric patterns"""
    # Create a checkerboard pattern
    tile_size = 40
    pattern = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                pattern[i:i+tile_size, j:j+tile_size] = 255
    
    # Add some noise for more interesting depth
    noise = np.random.normal(0, 20, (height, width))
    pattern = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    
    # Convert to RGB
    control_image = np.stack([pattern] * 3, axis=-1)
    return Image.fromarray(control_image)


def create_custom_params(pipeline, control_image, test_case):
    """Create custom parameters for specific test cases"""
    default_params = pipeline.InputParams()
    params = SimpleNamespace()
    
    # Copy defaults
    for field_name, field_value in default_params.__dict__.items():
        setattr(params, field_name, field_value)
    
    # Define test cases
    test_cases = {
        "high_quality": {
            "prompt": "a majestic dragon flying over a medieval castle, highly detailed, 8k",
            "steps": 8,
            "guidance_scale": 2.0,
            "strength": 0.9,
            "controlnet_scale": 0.8,
            "lora_scale": 1.0,
        },
        "fast_generation": {
            "prompt": "a simple landscape with mountains and trees",
            "steps": 1,
            "guidance_scale": 1.0,
            "strength": 0.6,
            "controlnet_scale": 0.4,
            "lora_scale": 0.5,
        },
        "creative_prompt": {
            "prompt": "a surreal dreamscape with floating islands and purple skies, abstract art style",
            "steps": 4,
            "guidance_scale": 1.5,
            "strength": 0.8,
            "controlnet_scale": 0.6,
            "lora_scale": 0.8,
        },
        "latent_travel_test": {
            "prompt": "a cyberpunk city at night",
            "target_prompt": "the same city during a bright sunny day",
            "use_latent_travel": True,
            "latent_travel_factor": 0.7,
            "steps": 4,
            "guidance_scale": 1.2,
            "strength": 0.7,
        }
    }
    
    # Apply test case parameters
    if test_case in test_cases:
        case_params = test_cases[test_case]
        for key, value in case_params.items():
            setattr(params, key, value)
    
    # Common settings
    params.width = 640
    params.height = 480
    params.seed = 42  # Fixed seed for reproducible results
    params.control_image = control_image
    
    return params


def run_custom_test():
    """Run custom test scenarios"""
    print("Running Custom Pipeline Tests")
    print("=" * 40)
    
    # Initialize pipeline
    config = create_test_config()
    pipeline = Pipeline(config, device, torch_dtype)
    
    # Create custom control image
    control_image = create_custom_control_image()
    
    # Test cases to run
    test_cases = [
        "high_quality",
        "fast_generation", 
        "creative_prompt",
        "latent_travel_test"
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case} ---")
        
        # Create parameters for this test case
        params = create_custom_params(pipeline, control_image, test_case)
        
        print(f"Prompt: {params.prompt}")
        print(f"Steps: {params.steps}")
        print(f"Guidance Scale: {params.guidance_scale}")
        print(f"Strength: {params.strength}")
        print(f"ControlNet Scale: {params.controlnet_scale}")
        
        try:
            import time
            start_time = time.time()
            result_image = pipeline.predict(params)
            inference_time = time.time() - start_time
            
            print(f"✓ Success! Inference time: {inference_time:.2f}s")
            print(f"  Image size: {result_image.size}")
            
            # Save the result
            output_filename = f"custom_test_{test_case}.png"
            result_image.save(output_filename)
            print(f"  Saved: {output_filename}")
            
            results.append({
                'test_case': test_case,
                'success': True,
                'inference_time': inference_time
            })
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'test_case': test_case,
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*40}")
    print("CUSTOM TEST SUMMARY")
    print(f"{'='*40}")
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"Successful tests: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r['inference_time'] for r in results if r['success']) / successful
        print(f"Average inference time: {avg_time:.2f}s")
    
    print("\nDetailed results:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        if result['success']:
            print(f"  {status} {result['test_case']}: {result['inference_time']:.2f}s")
        else:
            print(f"  {status} {result['test_case']}: {result['error']}")


def test_parameter_sweep():
    """Test a sweep of different parameter values"""
    print("\n" + "="*40)
    print("PARAMETER SWEEP TEST")
    print("="*40)
    
    config = create_test_config()
    pipeline = Pipeline(config, device, torch_dtype)
    control_image = create_custom_control_image()
    
    # Test different strength values
    strength_values = [0.3, 0.5, 0.7, 0.9]
    
    for strength in strength_values:
        print(f"\nTesting strength = {strength}")
        
        params = create_custom_params(pipeline, control_image, "fast_generation")
        params.strength = strength
        params.prompt = f"a beautiful landscape, strength={strength}"
        
        try:
            import time
            start_time = time.time()
            result_image = pipeline.predict(params)
            inference_time = time.time() - start_time
            
            print(f"  ✓ Success: {inference_time:.2f}s")
            
            # Save result
            filename = f"strength_test_{strength:.1f}.png"
            result_image.save(filename)
            print(f"  Saved: {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    try:
        # Run custom test scenarios
        run_custom_test()
        
        # Run parameter sweep
        test_parameter_sweep()
        
        print(f"\nAll custom tests completed!")
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc() 