#!/usr/bin/env python3
"""
Standalone Pipeline Tester for controlnetSDTurbot2i

This script instantiates and tests the controlnetSDTurbot2i pipeline with offline inputs,
allowing for repeated calls to the predict() method as opposed to the dynamic client/server
streaming setup in main.py.

Usage:
    python test_controlnet_pipeline.py [--num-iterations N] [--save-outputs] [--session-name NAME]
    
Default mode: If no session-name is provided, uses synthetic control images.
Session mode: If session-name is provided, looks for depth images in input/sessions/session_name/
"""

import os
import sys
import argparse
import time
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from types import SimpleNamespace

# Add the server directory to the Python path
server_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(server_dir))

# Import required modules
from config import Args
from device import device, torch_dtype
from util import get_pipeline_class
from pipelines.controlnetSDTurbot2i import Pipeline


def find_latest_session():
    """Find the latest session directory based on directory name (assumes timestamp format)"""
    sessions_dir = server_dir / "input" / "sessions"
    
    if not sessions_dir.exists():
        return None
    
    available_sessions = [d for d in sessions_dir.iterdir() if d.is_dir()]
    if not available_sessions:
        return None
    
    # Sort by name (which should be timestamp-based) and get the latest
    latest_session = sorted(available_sessions, key=lambda x: x.name)[-1]
    return latest_session.name


def find_depth_images(session_name):
    """Find all depth images in the specified session directory"""
    session_path = server_dir / "input" / "sessions" / session_name
    
    if not session_path.exists():
        raise FileNotFoundError(f"Session directory not found: {session_path}")
    
    # Find all depth images (files ending with _depth.png)
    depth_pattern = str(session_path / "*_depth.png")
    depth_files = glob.glob(depth_pattern)
    depth_files.sort()  # Sort for consistent ordering
    
    if not depth_files:
        raise FileNotFoundError(f"No depth images found in session: {session_name}")
    
    print(f"Found {len(depth_files)} depth images in session: {session_name}")
    return depth_files


def load_depth_image(depth_file_path):
    """Load a depth image and return as PIL Image"""
    try:
        image = Image.open(depth_file_path)
        # Ensure it's RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading depth image {depth_file_path}: {e}")
        return None


def create_sample_control_image(width=640, height=480):
    """Create a sample control image for testing (depth-like pattern)"""
    # Create a simple depth-like gradient pattern
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create a radial gradient pattern
    center_x, center_y = width // 2, height // 2
    distance = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    depth_pattern = (1 - distance) * 255
    depth_pattern = np.clip(depth_pattern, 0, 255).astype(np.uint8)
    
    # Convert to RGB
    depth_image = np.stack([depth_pattern] * 3, axis=-1)
    return Image.fromarray(depth_image)


def parse_metadata_file(metadata_file_path):
    """Parse a metadata file and return a dictionary of key-value pairs"""
    metadata = {}
    
    try:
        with open(metadata_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert numeric values
                    if key in ['seed', 'pipe_index']:
                        try:
                            metadata[key] = int(value)
                        except ValueError:
                            metadata[key] = value
                    elif key in ['guidance_scale', 'strength', 'timestamp']:
                        try:
                            metadata[key] = float(value)
                        except ValueError:
                            metadata[key] = value
                    else:
                        metadata[key] = value
        
        return metadata
        
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_file_path}")
        return {}
    except Exception as e:
        print(f"Error parsing metadata file {metadata_file_path}: {e}")
        return {}


def get_depth_metadata(depth_file_path):
    """Get metadata for a depth image by finding its corresponding metadata file"""
    depth_path = Path(depth_file_path)
    
    # Convert depth image filename to metadata filename
    # e.g., image_000052_034623_357_depth.png -> image_000052_034623_357_depth_metadata.txt
    depth_stem = depth_path.stem  # removes .png extension
    metadata_filename = f"{depth_stem}_metadata.txt"
    metadata_path = depth_path.parent / metadata_filename
    
    if metadata_path.exists():
        metadata = parse_metadata_file(metadata_path)
        print(f"Loaded metadata for {depth_path.name}: pipe_index={metadata.get('pipe_index', 'not found')}")
        return metadata
    else:
        print(f"No metadata file found for {depth_path.name}: {metadata_path}")
        return {}


def create_test_config():
    """Create a test configuration similar to the one used in main.py"""
    return Args(
        host="localhost",
        port=7860,
        reload=False,
        max_queue_size=0,
        timeout=0.0,
        safety_checker=False,
        torch_compile=False,
        taesd=True,
        pipeline="controlnetSDTurbot2i",
        ssl_certfile=None,
        ssl_keyfile=None,
        sfast=False,
        tensorrt=False,
        onediff=False,
        compel=True,
        debug=True,
        use_acid_processor=False,
        use_depth_estimator=False,
        depth_engine_path="modules/depth_anything/depth_anything_small.engine",
        depth_grayscale=False,
        use_prompt_travel=False,
        use_latent_travel=True,
        use_prompt_travel_scheduler=False,
        prompt_travel_min_factor=0.0,
        prompt_travel_max_factor=1.0,
        prompt_travel_factor_increment=0.025,
        prompt_travel_stabilize_duration=3,
        prompt_travel_oscillate=True,
        use_seed_travel=False,
        use_prompt_scheduler=False,
        prompts_dir="prompts",
        prompt_file_pattern="*.txt",
        loop_prompts=True,
        acid_strength=0.4,
        acid_coef_noise=0.15,
        acid_tracers=False,
        acid_strength_foreground=0.4,
        acid_zoom_factor=1.10,
        acid_x_shift=0,
        acid_y_shift=0,
        acid_wobblers=False,
        acid_color_matching=0.5,
        acid_human_seg=True,
        acid_blur=False,
        acid_brightness=1.0,
        acid_infrared_colorize=False,
        acid_low_bin_sensitivity=0.1,
        acid_high_bin_sensitivity=0.1,
        use_frequency_zoom=False,
        mic_index=0,
        use_lora_sound_control=False,
        use_test_zoom=False,
        use_test_shift=False,
        test_min_zoom=0.5,
        test_max_zoom=1.5,
        test_zoom_increment=0.03,
        test_zoom_stabilize_duration=3,
        test_x_shift_increment=0,
        test_y_shift_increment=0,
        test_x_max=50,
        test_y_max=50,
        use_background_removal=False,
        use_upscaler=False,
        upscaler_type="fast_srgan",
        upscaler_scale_factor=2.0,
        upscaler_resample_method="lanczos",
        use_pixelate_processor=False,
        default_curation_index=0,
        lora_model_name="glitch",
        use_image_saver=False,
        image_save_dir="output",
        image_save_format="png",
        image_save_quality=95,
        image_save_queue_size=100,
    )


def create_test_params(pipeline, control_image, iteration=0, depth_file_path=None, depth_metadata=None):
    """Create test parameters for the pipeline"""
    # Get default parameters from the pipeline
    default_params = pipeline.InputParams()
    
    # Create a SimpleNamespace object similar to how it's done in main.py
    params = SimpleNamespace()
    
    # Copy all default parameters
    for field_name, field_value in default_params.__dict__.items():
        setattr(params, field_name, field_value)
    
    # Override with test-specific values
    if depth_file_path:
        # Extract some info from the depth file name for more interesting prompts
        filename = Path(depth_file_path).stem
        
        # Use metadata prompt if available, otherwise generate one
        if depth_metadata and 'prompt' in depth_metadata:
            params.prompt = depth_metadata['prompt']
        else:
            params.prompt = f"a futuristic robot in a cyberpunk city, neon lights, depth frame {filename}"
        
        params.target_prompt = "a blue mechanical dog in the same cyberpunk scene"
    else:
        params.prompt = f"a futuristic robot in a cyberpunk city, neon lights, iteration {iteration}"
        params.target_prompt = "a blue mechanical dog in the same scene"
    
    # Use pipe_index from metadata if available, otherwise use iteration-based cycling
    if depth_metadata and 'pipe_index' in depth_metadata:
        params.pipe_index = depth_metadata['pipe_index']
        print(f"Using pipe_index from metadata: {params.pipe_index}")
    else:
        params.pipe_index = iteration % 5  # Cycle through pipe indices
        print(f"Using iteration-based pipe_index: {params.pipe_index}")
    
    params.use_prompt_travel = False
    params.prompt_travel_factor = 0.5
    params.use_latent_travel = True
    params.latent_travel_method = "slerp"
    params.latent_travel_factor = 0.3 + (iteration * 0.1) % 0.7  # Vary latent travel factor
    params.lora_scale = 0.8
    
    # Use seed from metadata if available
    if depth_metadata and 'seed' in depth_metadata:
        params.seed = depth_metadata['seed']
        print(f"Using seed from metadata: {params.seed}")
    else:
        params.seed = 4402026899276587 + iteration  # Vary seed for each iteration
        print(f"Using iteration-based seed: {params.seed}")
    
    params.target_seed = params.seed + 1
    params.steps = 2  # Keep low for fast testing
    params.width = 640
    params.height = 480
    
    # Use guidance_scale and strength from metadata if available
    if depth_metadata and 'guidance_scale' in depth_metadata:
        params.guidance_scale = depth_metadata['guidance_scale']
    else:
        params.guidance_scale = 1.0
    
    if depth_metadata and 'strength' in depth_metadata:
        params.strength = depth_metadata['strength']
    else:
        params.strength = 0.8
    
    params.controlnet_scale = 0.55
    params.controlnet_start = 0.0
    params.controlnet_end = 1.0
    params.debug_controlnet = False
    params.use_output_bg_removal = False
    
    # Set the control image
    params.control_image = control_image
    
    return params


def run_pipeline_test(num_iterations=5, save_outputs=False, output_dir="test_outputs", session_name=None, passthrough=False):
    """Run the pipeline test with specified number of iterations"""
    
    # Determine session to use
    if session_name is None:
        # Try to find the latest session automatically
        latest_session = find_latest_session()
        if latest_session:
            session_name = latest_session
            print(f"No session specified, using latest session: {session_name}")
        else:
            print("No sessions found, using synthetic mode")
    
    # Determine if we're using session mode or synthetic mode
    use_session_mode = session_name is not None
    depth_files = []
    
    if use_session_mode:
        print(f"Running in session mode with session: {session_name}")
        try:
            depth_files = find_depth_images(session_name)
            # Set num_iterations to match number of depth images found
            num_iterations = len(depth_files)
            print(f"Found {num_iterations} depth images, setting iterations to match")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Falling back to synthetic mode...")
            use_session_mode = False
            session_name = None
    else:
        print("Running in synthetic mode with generated control images")
    
    print(f"Starting pipeline test with {num_iterations} iterations...")
    print(f"Mode: {'Passthrough' if passthrough else 'Full Pipeline'}")
    print(f"Device: {device}")
    print(f"Torch dtype: {torch_dtype}")
    
    # Create test configuration
    config = create_test_config()
    
    # Initialize the pipeline
    print("Initializing pipeline...")
    start_time = time.time()
    pipeline = Pipeline(config, device, torch_dtype)
    init_time = time.time() - start_time
    print(f"Pipeline initialization took: {init_time:.2f} seconds")
    
    # Create output directory if saving outputs
    if save_outputs:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"Outputs will be saved to: {output_path.absolute()}")
    
    # Run test iterations
    total_inference_time = 0
    results = []
    
    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")
        
        # Get control image for this iteration
        if use_session_mode and i < len(depth_files):
            depth_file_path = depth_files[i]
            print(f"Loading depth image: {Path(depth_file_path).name}")
            control_image = load_depth_image(depth_file_path)
            
            if control_image is None:
                print(f"Failed to load depth image, skipping iteration {i+1}")
                continue
                
            # Save the control image for reference if saving outputs
            if save_outputs:
                control_filename = f"control_image_iteration_{i+1:03d}_{Path(depth_file_path).name}"
                control_image.save(output_path / control_filename)
        else:
            # Create synthetic control image
            print("Creating synthetic control image...")
            control_image = create_sample_control_image(width=640, height=480)
            depth_file_path = None
            
            # Save the control image for reference if saving outputs
            if save_outputs and i == 0:  # Only save once for synthetic mode
                control_image.save(output_path / "control_image_synthetic.png")
                print("Saved synthetic control image for reference")
        
        # Create test parameters for this iteration
        depth_metadata = get_depth_metadata(depth_file_path) if depth_file_path else None
        params = create_test_params(pipeline, control_image, i, depth_file_path, depth_metadata)
        
        print(f"Prompt: {params.prompt}")
        print(f"Pipe index: {params.pipe_index}")
        print(f"Latent travel factor: {params.latent_travel_factor:.2f}")
        print(f"Seed: {params.seed}")
        
        # Run prediction or passthrough
        start_time = time.time()
        try:
            if passthrough:
                # Passthrough mode: just return the control image
                result_image = control_image
                print("Passthrough mode: using control image as result")
            else:
                # Normal mode: run the pipeline
                result_image = pipeline.predict(params)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            print(f"{'Passthrough' if passthrough else 'Inference'} time: {inference_time:.2f} seconds")
            print(f"Result image size: {result_image.size}")
            
            # Save output if requested
            if save_outputs and result_image:
                if use_session_mode:
                    depth_name = Path(depth_files[i]).stem.replace('_depth', '')
                    output_filename = f"output_{depth_name}_iteration_{i+1:03d}.png"
                else:
                    output_filename = f"output_iteration_{i+1:03d}.png"
                result_image.save(output_path / output_filename)
                print(f"Saved: {output_filename}")
            
            results.append({
                'iteration': i+1,
                'inference_time': inference_time,
                'success': True,
                'image_size': result_image.size if result_image else None,
                'depth_file': depth_file_path if use_session_mode else None,
                'passthrough': passthrough
            })
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'iteration': i+1,
                'inference_time': 0,
                'success': False,
                'error': str(e),
                'depth_file': depth_file_path if use_session_mode else None,
                'passthrough': passthrough
            })
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Mode: {'Session' if use_session_mode else 'Synthetic'}")
    if use_session_mode:
        print(f"Session: {session_name}")
    print(f"Pipeline mode: {'Passthrough' if passthrough else 'Full inference'}")
    print(f"Total iterations: {num_iterations}")
    successful_runs = sum(1 for r in results if r['success'])
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {num_iterations - successful_runs}")
    
    if successful_runs > 0:
        avg_inference_time = total_inference_time / successful_runs
        print(f"Average {'passthrough' if passthrough else 'inference'} time: {avg_inference_time:.2f} seconds")
        print(f"Total {'passthrough' if passthrough else 'inference'} time: {total_inference_time:.2f} seconds")
        
        # Print per-iteration results
        print(f"\nPer-iteration results:")
        for result in results:
            status = "✓" if result['success'] else "✗"
            if result['success']:
                depth_info = f" ({Path(result['depth_file']).name})" if result.get('depth_file') else ""
                print(f"  {status} Iteration {result['iteration']}: {result['inference_time']:.2f}s{depth_info}")
            else:
                print(f"  {status} Iteration {result['iteration']}: FAILED - {result.get('error', 'Unknown error')}")
    
    if save_outputs:
        print(f"\nOutputs saved to: {output_path.absolute()}")
    
    return results


def get_session_info():
    """Get information about available sessions and their depth image counts"""
    sessions_dir = server_dir / "input" / "sessions"
    session_info = []
    
    if not sessions_dir.exists():
        return session_info
    
    available_sessions = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
    for session in sorted(available_sessions):
        session_path = sessions_dir / session
        depth_count = len(glob.glob(str(session_path / "*_depth.png")))
        session_info.append({
            'name': session,
            'depth_count': depth_count,
            'path': session_path
        })
    
    return session_info


def display_session_info(session_info, specified_session=None):
    """Display information about available sessions and which one will be used"""
    if not session_info:
        print("No sessions found. Will use synthetic mode.")
        return None
    
    print("Available sessions:")
    for session in session_info:
        print(f"  - {session['name']} ({session['depth_count']} depth images)")
    
    if specified_session is None:
        latest_session = find_latest_session()
        if latest_session:
            print(f"\nWill use latest session: {latest_session}")
            print("To use a different session, add --session-name <session_name>")
            return latest_session
        else:
            print("\nNo valid sessions found, will use synthetic mode.")
            return None
    else:
        print(f"\nWill use specified session: {specified_session}")
        return specified_session


def process_arguments(args):
    """Process command line arguments and determine execution parameters"""
    # Get session information
    session_info = get_session_info()
    
    # Display session info and determine which session to use
    if not session_info:
        print("Sessions directory not found. Will use synthetic mode.")
        selected_session = None
    else:
        selected_session = display_session_info(session_info, args.session_name)
    
    # Determine num_iterations default
    if args.num_iterations is None:
        # Will be auto-detected in run_pipeline_test based on session mode
        default_iterations = 5  # Only used for synthetic mode
    else:
        default_iterations = args.num_iterations
    
    return {
        'session_name': selected_session,
        'num_iterations': default_iterations,
        'save_outputs': args.save_outputs,
        'output_dir': args.output_dir,
        'passthrough': args.passthrough
    }


def run_test_with_error_handling(test_params):
    """Run the pipeline test with proper error handling and exit codes"""
    try:
        results = run_pipeline_test(**test_params)
        
        # Exit with error code if any tests failed
        failed_tests = sum(1 for r in results if not r['success'])
        if failed_tests > 0:
            print(f"\n{failed_tests} test(s) failed!")
            sys.exit(1)
        else:
            print(f"\nAll {len(results)} tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Test the controlnetSDTurbot2i pipeline")
    parser.add_argument(
        "--num-iterations", "-n",
        type=int,
        default=None,
        help="Number of test iterations to run (default: auto-detect from session depth images, or 5 for synthetic mode)"
    )
    parser.add_argument(
        "--save-outputs", "-s",
        action="store_true",
        help="Save output images to disk"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_outputs",
        help="Directory to save outputs (default: test_outputs)"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Session name to load depth images from (e.g., 'session_20250601_034606'). If not provided, uses latest session automatically."
    )
    parser.add_argument(
        "--passthrough", "-p",
        action="store_true",
        help="Passthrough mode: skip pipeline inference and return control image as result (useful for testing flow)"
    )
    
    args = parser.parse_args()
    
    # Process arguments and determine execution parameters
    test_params = process_arguments(args)
    
    # Run the test with error handling
    run_test_with_error_handling(test_params)


if __name__ == "__main__":
    main() 