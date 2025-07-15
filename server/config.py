from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    reload: bool
    max_queue_size: int
    timeout: float
    safety_checker: bool
    torch_compile: bool
    taesd: bool
    pipeline: str
    ssl_certfile: str
    ssl_keyfile: str
    sfast: bool
    tensorrt: bool = False
    onediff: bool = False
    compel: bool = False
    debug: bool = False
    use_acid_processor: bool = False
    # Enable depth estimation
    use_depth_estimator: bool = False
    depth_engine_path: str = "modules/depth_anything/depth_anything_small.engine"
    depth_grayscale: bool = False
    # Depth threshold parameters
    depth_normalized_distance_threshold: float = 0.225
    depth_absolute_min: float = 0.0
    depth_absolute_max: float = 18.0
    # Use camera as control image
    use_camera_as_control: bool = False
    # Enable prompt travel
    use_prompt_travel: bool = False
    use_latent_travel: bool = False
    # Prompt travel scheduler settings
    use_prompt_travel_scheduler: bool = False
    prompt_travel_min_factor: float = 0.0
    prompt_travel_max_factor: float = 1.0
    prompt_travel_factor_increment: float = 0.025
    prompt_travel_stabilize_duration: int = 3
    prompt_travel_oscillate: bool = True
    use_seed_travel: bool = False
    # Prompt scheduler settings
    use_prompt_scheduler: bool = False
    prompts_dir: str = "prompts"
    prompt_file_pattern: str = "*.txt"
    loop_prompts: bool = True
    # Default acid processor settings
    acid_strength: float = 0.4
    acid_coef_noise: float = 0.15
    acid_tracers: bool = False
    acid_strength_foreground: float = 0.4
    acid_zoom_factor: float = 1.10
    acid_x_shift: int = 0
    acid_y_shift: int = 0
    acid_wobblers: bool = False
    acid_color_matching: float = 0.5
    acid_human_seg: bool = True
    acid_blur: bool = False
    acid_brightness: float = 1.0
    acid_infrared_colorize: bool = False
    # Audio frequency zoom controller settings
    acid_low_bin_sensitivity: float = 0.1
    acid_high_bin_sensitivity: float = 0.1
    # Enable frequency zoom controller
    use_frequency_zoom: bool = True # TODO: change to more generic acid controller flag
    mic_index: int = 0
    # Enable LoRA sound controller
    use_lora_sound_control: bool = False
    # LoRA sound controller treble boost factors for mel bins [bass, low_mids, mids, high_mids, treble]
    lora_treble_boost_factors: str = "1.0,1.0,1.5,2.0,2.5"
    # Test oscillator settings
    use_test_zoom: bool = False
    use_test_shift: bool = False
    test_min_zoom: float = 0.5
    test_max_zoom: float = 1.5
    test_zoom_increment: float = 0.03
    test_zoom_stabilize_duration: int = 3
    test_x_shift_increment: int = 0
    test_y_shift_increment: int = 0
    test_x_max: int = 50
    test_y_max: int = 50
    use_background_removal: bool = True
    # Enable upscaler
    use_upscaler: bool = False
    upscaler_type: str = "fast_srgan"
    upscaler_scale_factor: float = 2.0
    upscaler_resample_method: str = "lanczos"
    # Add pixelate processor argument
    use_pixelate_processor: bool = False
    default_curation_index: int = 0
    prompts_file_name: str = "glitch"
    # Image saver settings
    use_image_saver: bool = False
    image_save_dir: str = "output"
    image_save_format: str = "png"
    image_save_quality: int = 95
    image_save_queue_size: int = 100
    # Warmup settings
    warmup: bool = True
    # LoRA Configuration Directory
    lora_config_dir: str = "lora_config"

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", None) == "True"
USE_TAESD = os.environ.get("USE_TAESD", "False") == "True"
default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    dest="safety_checker",
    action="store_true",
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--torch-compile",
    dest="torch_compile",
    action="store_true",
    default=TORCH_COMPILE,
    help="Torch Compile",
)
parser.add_argument(
    "--taesd",
    dest="taesd",
    action="store_true",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--pipeline",
    type=str,
    default="txt2img",
    help="Pipeline to use",
)
parser.add_argument(
    "--ssl-certfile",
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=True,
    help="Debug",
)
parser.add_argument(
    "--compel",
    action="store_true",
    default=False,
    help="Compel",
)
parser.add_argument(
    "--sfast",
    action="store_true",
    default=False,
    help="Enable Stable Fast",
)
parser.add_argument(
    "--tensorrt",
    action="store_true",
    default=False,
    help="Enable TensorRT acceleration",
)
parser.add_argument(
    "--onediff",
    action="store_true",
    default=False,
    help="Enable OneDiff",
)
parser.add_argument(
    "--use-acid-processor",
    dest="use_acid_processor",
    action="store_true",
    default=True,
    help="Enable Acid Processor",
)
# Add acid processor arguments
parser.add_argument(
    "--acid-strength",
    dest="acid_strength",
    type=float,
    default=0.65,
    help="Acid effect strength",
)
parser.add_argument(
    "--acid-strength-foreground",
    dest="acid_strength_foreground",
    type=float,
    default=0.65,
    help="Acid effect strength for foreground",
)
parser.add_argument(
    "--acid-coef-noise",
    dest="acid_coef_noise",
    type=float,
    default=0.0,
    help="Coefficient for noise in acid effect",
)
parser.add_argument(
    "--acid-tracers",
    dest="acid_tracers",
    action="store_true",
    default=False,
    help="Enable acid tracers effect",
)
parser.add_argument(
    "--acid-zoom-factor",
    dest="acid_zoom_factor",
    type=float,
    default=1.0,
    help="Zoom factor for acid effect",
)
parser.add_argument(
    "--acid-x-shift",
    dest="acid_x_shift",
    type=int,
    default=0,
    help="X shift for acid effect",
)
parser.add_argument(
    "--acid-y-shift",
    dest="acid_y_shift",
    type=int,
    default=0,
    help="Y shift for acid effect",
)
parser.add_argument(
    "--acid-wobblers",
    dest="acid_wobblers",
    action="store_true",
    default=False,
    help="Enable acid wobblers effect",
)
parser.add_argument(
    "--acid-color-matching",
    dest="acid_color_matching",
    type=float,
    default=0.0,
    help="Color matching strength for acid effect",
)
parser.add_argument(
    "--acid-human-seg",
    dest="acid_human_seg",
    action="store_true",
    default=False,
    help="Enable human segmentation in acid effect",
)
parser.add_argument(
    "--acid-blur",
    dest="acid_blur",
    action="store_true",
    default=False,
    help="Enable blur in acid effect",
)
parser.add_argument(
    "--acid-brightness",
    dest="acid_brightness",
    type=float,
    default=1.0,
    help="Brightness adjustment in acid effect",
)
parser.add_argument(
    "--acid-infrared-colorize",
    dest="acid_infrared_colorize",
    action="store_true",
    default=False,
    help="Enable infrared colorization in acid effect",
)
# Add new arguments for frequency zoom controller
parser.add_argument(
    "--acid-low-bin-sensitivity",
    dest="acid_low_bin_sensitivity",
    type=float,
    default=0.1,
    help="Sensitivity of low frequency bins for zoom out effect (0.0-1.0)",
)
parser.add_argument(
    "--acid-high-bin-sensitivity",
    dest="acid_high_bin_sensitivity",
    type=float,
    default=0.1,
    help="Sensitivity of high frequency bins for zoom in effect (0.0-1.0)",
)

# Add frequency zoom controller arguments
parser.add_argument(
    "--use-frequency-zoom",
    dest="use_frequency_zoom",
    action="store_true",
    default=True,
    help="Enable frequency zoom control",
)
parser.add_argument(
    "--mic-index",
    dest="mic_index",
    type=int,
    default=8,
    help="Mic Device Index",
)
# Add test oscillator parameters
parser.add_argument(
    "--use-test-zoom",
    dest="use_test_zoom",
    action="store_true",
    default=False,
    help="Enable test zoom oscillation",
)
parser.add_argument(
    "--use-test-shift",
    dest="use_test_shift",
    action="store_true",
    default=False,
    help="Enable test shift oscillation",
)
parser.add_argument(
    "--test-min-zoom",
    dest="test_min_zoom",
    type=float,
    default=1.0,
    help="Minimum zoom value for test oscillation",
)
parser.add_argument(
    "--test-max-zoom",
    dest="test_max_zoom",
    type=float,
    default=2.0,
    help="Maximum zoom value for test oscillation",
)
parser.add_argument(
    "--test-zoom-increment",
    dest="test_zoom_increment",
    type=float,
    default=0.03,
    help="Zoom increment per frame for test oscillation",
)
parser.add_argument(
    "--test-zoom-stabilize-duration",
    dest="test_zoom_stabilize_duration",
    type=int,
    default=3,
    help="Number of frames to stabilize at zoom=1.0",
)
parser.add_argument(
    "--test-x-shift-increment",
    dest="test_x_shift_increment",
    type=int,
    default=0,
    help="X shift increment per frame for test oscillation",
)
parser.add_argument(
    "--test-y-shift-increment",
    dest="test_y_shift_increment",
    type=int,
    default=0,
    help="Y shift increment per frame for test oscillation",
)
parser.add_argument(
    "--test-x-max",
    dest="test_x_max",
    type=int,
    default=50,
    help="Maximum X shift for test oscillation",
)
parser.add_argument(
    "--test-y-max",
    dest="test_y_max",
    type=int,
    default=50,
    help="Maximum Y shift for test oscillation",
)
parser.add_argument(
    "--use-backround-removal",
    dest="use_background_removal",
    action="store_true",
    default=False,
    help="Remove the background from the image feed"
)
# Add prompt travel argument
parser.add_argument(
    "--use-prompt-travel",
    dest="use_prompt_travel",
    action="store_true",
    default=True,
    help="Enable Prompt Travel",
)
# Add latent travel argument
parser.add_argument(
    "--use-latent-travel",
    dest="use_latent_travel",
    action="store_true",
    default=True,
    help="Enable Latent Travel",
)
# Add prompt travel scheduler arguments
parser.add_argument(
    "--use-prompt-travel-scheduler",
    dest="use_prompt_travel_scheduler",
    action="store_true",
    default=True,
    help="Enable Prompt Travel Scheduler",
)
parser.add_argument(
    "--prompt-travel-min-factor",
    dest="prompt_travel_min_factor",
    type=float,
    default=0.0,
    help="Minimum prompt travel factor",
)
parser.add_argument(
    "--prompt-travel-max-factor",
    dest="prompt_travel_max_factor",
    type=float,
    default=1.0,
    help="Maximum prompt travel factor",
)
parser.add_argument(
    "--prompt-travel-factor-increment",
    dest="prompt_travel_factor_increment",
    type=float,
    default=0.025,
    help="Increment amount for prompt travel factor",
)
parser.add_argument(
    "--prompt-travel-stabilize-duration",
    dest="prompt_travel_stabilize_duration",
    type=int,
    default=3,
    help="Number of frames to stabilize at min/max factor",
)
parser.add_argument(
    "--prompt-travel-oscillate",
    dest="prompt_travel_oscillate",
    action="store_true",
    default=True,
    help="Oscillate between min and max factor",
)

# Add seed travel argument
parser.add_argument(
    "--use-seed-travel",
    dest="use_seed_travel",
    action="store_true",
    default=False,
    help="Use seed travel",
)

# Prompt scheduler settings
parser.add_argument(
    "--use-prompt-scheduler",
    dest="use_prompt_scheduler",
    action="store_true",
    default=True,
    help="Use prompt scheduler",
)
parser.add_argument(
    "--prompts-dir",
    dest="prompts_dir",
    type=str,
    default="prompts",
    help="Directory containing prompt files",
)
parser.add_argument(
    "--prompt-file-pattern",
    dest="prompt_file_pattern",
    type=str,
    default="*.txt",
    help="Pattern to match prompt files",
)
parser.add_argument(
    "--loop-prompts",
    dest="loop_prompts",
    action="store_true",
    default=True,
    help="Loop back to the beginning when reaching the end of prompts",
)
parser.add_argument(
    "--no-loop-prompts",
    dest="loop_prompts",
    action="store_false",
    help="Don't loop back to the beginning when reaching the end of prompts",
)

# Add depth estimation arguments
parser.add_argument(
    "--use-depth-estimator",
    dest="use_depth_estimator",
    action="store_true",
    default=True,
    help="Enable depth estimation using DepthAnything TensorRT",
)
parser.add_argument(
    "--depth-grayscale",
    dest="depth_grayscale",
    action="store_true",
    default=False,
    help="Save depth maps in grayscale",
)
parser.add_argument(
    "--use-camera-as-control",
    dest="use_camera_as_control",
    action="store_true",
    default=True,
    help="Use camera image directly as control image, bypassing depth estimation (useful for depth cameras)",
)

# Add depth threshold arguments
parser.add_argument(
    "--depth-normalized-distance-threshold",
    dest="depth_normalized_distance_threshold",
    type=float,
    default=0.225,
    help="Normalized distance threshold for depth estimation (0.0-1.0, default: 0.225)",
)
parser.add_argument(
    "--depth-absolute-min",
    dest="depth_absolute_min",
    type=float,
    default=0.0,
    help="Absolute minimum depth value for normalization (default: 0.0)",
)
parser.add_argument(
    "--depth-absolute-max",
    dest="depth_absolute_max",
    type=float,
    default=18.0,
    help="Absolute maximum depth value for normalization (default: 18.0)",
)

# Add pixelate processor argument
parser.add_argument(
    "--use-pixelate-processor",
    dest="use_pixelate_processor",
    action="store_true",
    default=True,
    help="Enable Pixelate Processor",
)
# Add upscaler arguments
parser.add_argument(
    "--use-upscaler",
    dest="use_upscaler",
    action="store_true",
    default=True,
    help="Enable upscaler for output images",
)
parser.add_argument(
    "--upscaler-type",
    dest="upscaler_type",
    type=str,
    default="rvsr",
    choices=["pil", "fast_srgan", "omni_sr", "rvsr", "dscf_sr"],
    help="Type of upscaler to use (default: pil). Options: pil (basic), fast_srgan (high quality), omni_sr (high quality), rvsr (high quality), dscf_sr (high quality)",
)
parser.add_argument(
    "--upscaler-scale-factor",
    dest="upscaler_scale_factor",
    type=float,
    default=4.0,
    help="Scale factor for upscaler (default: 2.0)",
)
parser.add_argument(
    "--upscaler-resample-method",
    dest="upscaler_resample_method",
    type=str,
    default="lanczos",
    choices=["nearest", "bilinear", "bicubic", "lanczos"],
    help="Resampling method for upscaler (default: lanczos)",
)

parser.add_argument(
    "--use-lora-sound-control",
    dest="use_lora_sound_control",
    action="store_true",
    default=True,
    help="Enable LoRA sound controller",
)

parser.add_argument(
    "--lora-treble-boost-factors",
    dest="lora_treble_boost_factors",
    type=str,
    default="1.0,1.0,1.0,1.25,1.5",
    help="Comma-separated list of treble boost factors for LoRA sound controller mel bins [bass, low_mids, mids, high_mids, treble]. Default: 1.0,1.0,1.5,2.0,2.5",
)

# Image saver arguments
parser.add_argument(
    "--use-image-saver",
    dest="use_image_saver",
    action="store_true",
    default=False,
    help="Enable image saving to disk",
)
parser.add_argument(
    "--image-save-dir",
    dest="image_save_dir",
    type=str,
    default="output",
    help="Directory to save images (default: output)",
)
parser.add_argument(
    "--image-save-format",
    dest="image_save_format",
    type=str,
    default="png",
    choices=["png", "jpg", "jpeg"],
    help="Image format for saving (default: png)",
)
parser.add_argument(
    "--image-save-quality",
    dest="image_save_quality",
    type=int,
    default=95,
    help="Image quality for JPEG format (1-100, default: 95)",
)
parser.add_argument(
    "--image-save-queue-size",
    dest="image_save_queue_size",
    type=int,
    default=1000,
    help="Maximum queue size for image saving (default: 1000)",
)


parser.add_argument(
    "--default-curation-index",
    dest="default_curation_index",
    type=int,
    default=5,
    help="Default index for curation selection",
)

parser.add_argument(
    "--prompts-file-name",
    dest="prompts_file_name",
    type=str,
    default="megamix",
    help="Name of the prompts file to use (without .txt extension), corresponds to a LoRA curation config.",
)

# Warmup settings
parser.add_argument(
    "--warmup",
    dest="warmup",
    action="store_true",
    default=True,
    help="Warmup all pipes during startup to pre-trace computational graphs",
)

# Add LoRA config directory argument
parser.add_argument(
    "--lora-config-dir",
    dest="lora_config_dir",
    type=str,
    default="lora_config",
    help="Directory containing LoRA curation JSON configuration files.",
)

parser.set_defaults(taesd=USE_TAESD)

config = Args(**vars(parser.parse_args()))
config.pretty_print()
