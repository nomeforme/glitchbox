# Pipeline Tester

This directory contains standalone scripts for testing pipelines with offline inputs, as opposed to the dynamic client/server streaming setup in `main.py`.

## controlnetSDTurbot2i Pipeline Tester

### Overview

The `test_controlnet_pipeline.py` script allows you to test the `controlnetSDTurbot2i` pipeline in isolation with configurable parameters and multiple iterations.

### Features

- **Standalone Testing**: No need for the full server/client setup
- **Configurable Parameters**: Easy to modify test parameters
- **Multiple Iterations**: Run multiple tests with varying parameters
- **Performance Metrics**: Tracks inference times and success rates
- **Output Saving**: Optionally save generated images to disk
- **Error Handling**: Comprehensive error reporting and recovery

### Usage

#### Basic Usage

```bash
# Run 5 iterations with default settings
python test_controlnet_pipeline.py

# Run 10 iterations
python test_controlnet_pipeline.py --num-iterations 10

# Run 3 iterations and save outputs
python test_controlnet_pipeline.py -n 3 --save-outputs

# Save outputs to custom directory
python test_controlnet_pipeline.py -s -o my_test_outputs
```

#### Command Line Arguments

- `--num-iterations, -n`: Number of test iterations to run (default: 5)
- `--save-outputs, -s`: Save output images to disk
- `--output-dir, -o`: Directory to save outputs (default: test_outputs)

### What the Script Does

1. **Pipeline Initialization**: Creates and initializes the controlnetSDTurbot2i pipeline with test configuration
2. **Control Image Generation**: Creates a sample depth-like control image for testing
3. **Parameter Variation**: For each iteration, varies:
   - Prompt content (includes iteration number)
   - Pipe index (cycles through 0-4)
   - Latent travel factor
   - Random seed
4. **Performance Tracking**: Measures and reports:
   - Pipeline initialization time
   - Per-iteration inference time
   - Average inference time
   - Success/failure rates
5. **Output Management**: Optionally saves generated images with descriptive filenames

### Test Configuration

The script uses a test configuration that mirrors the production setup but with optimizations for testing:

- **TAESD**: Enabled for faster VAE operations
- **Compel**: Enabled for prompt processing
- **Debug**: Enabled for detailed logging
- **Steps**: Set to 2 for faster inference
- **Latent Travel**: Enabled to test interpolation features
- **Background Processing**: Disabled for simpler testing

### Sample Output

```
Starting pipeline test with 5 iterations...
Device: cuda:0
Torch dtype: torch.float16
Initializing pipeline...
Pipeline initialization took: 12.34 seconds
Creating sample control image...
Outputs will be saved to: /path/to/test_outputs

--- Iteration 1/5 ---
Prompt: a futuristic robot in a cyberpunk city, neon lights, iteration 0
Pipe index: 0
Latent travel factor: 0.30
Seed: 4402026899276587
Inference time: 2.15 seconds
Result image size: (640, 480)
Saved: output_iteration_001.png

...

==================================================
TEST SUMMARY
==================================================
Total iterations: 5
Successful runs: 5
Failed runs: 0
Average inference time: 2.08 seconds
Total inference time: 10.42 seconds

Per-iteration results:
  ✓ Iteration 1: 2.15s
  ✓ Iteration 2: 2.03s
  ✓ Iteration 3: 2.11s
  ✓ Iteration 4: 2.05s
  ✓ Iteration 5: 2.08s

Outputs saved to: /path/to/test_outputs

All 5 tests passed!
```

### Customization

You can easily customize the test by modifying:

- **Test Configuration**: Edit `create_test_config()` to change pipeline settings
- **Test Parameters**: Modify `create_test_params()` to adjust generation parameters
- **Control Image**: Replace `create_sample_control_image()` with your own image generation logic
- **Prompts**: Change the prompt generation logic in `create_test_params()`

### Requirements

- All dependencies from the main project
- PIL (Pillow) for image handling
- NumPy for array operations
- PyTorch for tensor operations

### Troubleshooting

1. **Import Errors**: Make sure you're running from the correct directory and all dependencies are installed
2. **CUDA Errors**: Ensure your GPU has sufficient memory and CUDA is properly configured
3. **Model Loading Errors**: Check that all required model files are present and accessible
4. **Permission Errors**: Ensure write permissions for the output directory

### Adding New Pipeline Tests

To create tests for other pipelines:

1. Copy `test_controlnet_pipeline.py` as a template
2. Update the import statement to your target pipeline
3. Modify the configuration and parameters to match your pipeline's requirements
4. Adjust the control image generation if needed
5. Update the README with pipeline-specific information 