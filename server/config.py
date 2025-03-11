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
    onediff: bool = False
    compel: bool = False
    debug: bool = False
    use_acid_processor: bool = False
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
    "--onediff",
    action="store_true",
    default=False,
    help="Enable OneDiff",
)
parser.add_argument(
    "--use-acid-processor",
    dest="use_acid_processor",
    action="store_true",
    default=False,
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
    default=False,
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
    default=0.5,
    help="Minimum zoom value for test oscillation",
)
parser.add_argument(
    "--test-max-zoom",
    dest="test_max_zoom",
    type=float,
    default=1.5,
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

parser.set_defaults(taesd=USE_TAESD)

config = Args(**vars(parser.parse_args()))
config.pretty_print()
