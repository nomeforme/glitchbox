import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import markdown2
from pipelines.utils.safety_checker import SafetyChecker
from PIL import Image
import logging
from config import config, Args
from connection_manager import ConnectionManager, ServerFullException
import uuid
import time
from types import SimpleNamespace
from util import pil_to_frame, bytes_to_pil, is_firefox, get_pipeline_class
from device import device, torch_dtype
import asyncio
import os
import time
# Import the acid processor
from modules.acid_processor import AcidProcessor, InputImageProcessor
# Import the frequency zoom controller
from modules.audio_controller import BeatZoomController, LoraSoundController
# Import fft analyzer
from modules.fft.stream_analyzer import Stream_Analyzer
# Import test oscillators
from utils.test_oscillators import ZoomOscillator, ShiftOscillator
# Import the prompt travel scheduler
from modules.prompt_scheduler import PromptTravelScheduler
# Import background removal processor
from modules.bg_removal import get_processor as get_bg_removal_processor
# Import the depth estimator
from modules.depth_anything.depth_anything_trt import DepthAnythingTRT as DepthAnything
# Import the image saver
from modules.image_saver import get_image_saver
# Import LoRACurationConfig for curation management
from pipelines.config import LoRACurationConfig

import numpy as np
import zmq
import pycuda.driver as cuda

# # Print detailed CUDA device information
# print("\nDetailed CUDA Device Information:")
# cuda.init()

# # Get all devices and their properties
# devices = []
# for i in range(cuda.Device.count()):
#     devicea = cuda.Device(i)
#     devices.append({
#         'index': i,
#         'name': devicea.name(),
#         'memory': devicea.total_memory(),
#         'compute_capability': devicea.compute_capability()
#     })

# # Sort devices by compute capability (lower first) and then by memory
# devices.sort(key=lambda x: (x['compute_capability'][0], x['compute_capability'][1], x['memory']))

# # Print device information in sorted order
# for device_info in devices:
#     device = cuda.Device(device_info['index'])
#     print(f"\nDevice {device_info['index']}: {device.name()}")
#     print(f"  Make: NVIDIA")
#     print(f"  Model: {device.name()}")
#     print(f"  Compute Capability: {device.compute_capability()}")
#     print(f"  Total Memory: {device.total_memory() / 1024**2:.2f} MB")
#     print(f"  Multi Processor Count: {device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)}")
#     print(f"  Clock Rate: {device.get_attribute(cuda.device_attribute.CLOCK_RATE) / 1000:.2f} MHz")
#     print(f"  Max Threads Per Block: {device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)}")
#     print(f"  Max Block Dimensions: {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X)} x {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y)} x {device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z)}")
#     print(f"  Max Grid Dimensions: {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)} x {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)} x {device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z)}")
#     print(f"  Warp Size: {device.get_attribute(cuda.device_attribute.WARP_SIZE)}")
#     print(f"  Max Shared Memory Per Block: {device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK) / 1024:.2f} KB")
#     print(f"  Device Overlap: {'Yes' if device.get_attribute(cuda.device_attribute.ASYNC_ENGINE_COUNT) > 0 else 'No'}")
#     print(f"  Concurrent Kernels: {'Yes' if device.get_attribute(cuda.device_attribute.CONCURRENT_KERNELS) else 'No'}")



THROTTLE = 1.0 / 120


class App:
    def __init__(self, config: Args, pipeline, lora_config):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        if self.args.safety_checker:
            self.safety_checker = SafetyChecker(device=device.type)
        
        # Use the provided LoRACurationConfig
        self.lora_config = lora_config
        print(f"[main.py] Using provided LoRACurationConfig with curation index {getattr(self.args, 'default_curation_index', 0)}")
        
        # Initialize depth estimator if enabled
        self.use_depth_estimator = getattr(self.args, 'use_depth_estimator', False)
        if self.use_depth_estimator:
            print("[main.py] Depth estimator will be initialized on startup")
            # The actual initialization happens in the startup event
            
            # Get the engine path from config or use default
            self.depth_engine_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "modules", "depth_anything", "models", "depth_anything_v2_vits.trt"
            )
            
            # Check if the engine file exists
            if not os.path.exists(self.depth_engine_path):
                print(f"[main.py] Warning: Depth engine file not found: {self.depth_engine_path}")
                print("[main.py] Running without depth estimation")
                self.use_depth_estimator = False
            else:
                print(f"[main.py] Using depth engine: {self.depth_engine_path}")
        
        # Initialize prompt travel service
        self.use_prompt_travel = getattr(self.args, 'use_prompt_travel', False)
        self.use_latent_travel = getattr(self.args, 'use_latent_travel', False)
        if self.use_prompt_travel or self.use_latent_travel:
            print("[main.py] Travel service will be initialized on startup")
            print(f"[main.py] Use prompt travel: {self.use_prompt_travel}")
            print(f"[main.py] Use latent travel: {self.use_latent_travel}")
            # The actual initialization happens in the startup event
            
            # Get the prompts_file_name from the current curation config
            current_config = self.lora_config.get_config_for_curation(self.lora_config.default_curation_key)
            prompts_file_name = current_config.get('prompts_file_name') if current_config else getattr(self.args, 'prompts_file_name', 'glitch')
            print(f"[main.py] Using prompts_file_name from curation config: {prompts_file_name}")
            
            # Initialize prompt travel scheduler
            self.prompt_travel_scheduler = PromptTravelScheduler(
                min_factor=getattr(self.args, 'prompt_travel_min_factor', 0.0),
                max_factor=getattr(self.args, 'prompt_travel_max_factor', 1.0),
                factor_increment=getattr(self.args, 'prompt_travel_factor_increment', 0.025),
                stabilize_duration=getattr(self.args, 'prompt_travel_stabilize_duration', 3),
                oscillate=getattr(self.args, 'prompt_travel_oscillate', True),
                enabled=getattr(self.args, 'use_prompt_travel_scheduler', False),
                debug=getattr(self.args, 'debug', False),
                use_prompt_scheduler=getattr(self.args, 'use_prompt_scheduler', False),
                prompts_dir=getattr(self.args, 'prompts_dir', "prompts"),
                prompt_file_pattern=getattr(self.args, 'prompt_file_pattern', "*.txt"),
                loop_prompts=getattr(self.args, 'loop_prompts', True),
                prompts_file_name=prompts_file_name  # Use prompts_file_name from curation config
            )
        
        # Initialize acid processors
        self.use_acid_processor = getattr(self.args, 'use_acid_processor', False)
        self.use_lora_sound_control = getattr(self.args, 'use_lora_sound_control', False)
        if self.use_acid_processor:
            # print("[main.py] Initializing acid processor")
            self.input_processor = InputImageProcessor(device=device.type)
            # Configure input processor with default settings from config
            self.input_processor.set_human_seg(getattr(self.args, 'acid_human_seg', True))
            self.input_processor.set_blur(getattr(self.args, 'acid_blur', False))
            self.input_processor.set_brightness(getattr(self.args, 'acid_brightness', 1.0))
            self.input_processor.set_infrared_colorize(getattr(self.args, 'acid_infrared_colorize', False))
            
            # Get dimensions from pipeline info if available
            info = pipeline.Info()
            height = getattr(info, 'height', 512)  # Default height
            width = getattr(info, 'width', 512)    # Default width
            # print(f"[main.py] Pipeline dimensions: height={height}, width={width}")
            
            self.acid_processor = AcidProcessor(
                height_diffusion=height + 256,
                width_diffusion=width + 256,
                device=device.type,
            )
            
            # Configure acid processor with default settings from config
            self.acid_processor.set_acid_strength(getattr(self.args, 'acid_strength', 0.11))
            self.acid_processor.set_coef_noise(getattr(self.args, 'acid_coef_noise', 0.15))
            self.acid_processor.set_acid_tracers(getattr(self.args, 'acid_tracers', False))
            self.acid_processor.set_acid_strength_foreground(getattr(self.args, 'acid_strength_foreground', 0.11))
            self.acid_processor.set_zoom_factor(getattr(self.args, 'acid_zoom_factor', 1.0))
            self.acid_processor.set_x_shift(getattr(self.args, 'acid_x_shift', 0))
            self.acid_processor.set_y_shift(getattr(self.args, 'acid_y_shift', 0))
            self.acid_processor.set_do_acid_wobblers(getattr(self.args, 'acid_wobblers', False))
            self.acid_processor.set_color_matching(getattr(self.args, 'acid_color_matching', 0.5))

            # # Initialize the FFT analyzer
            # self.fft_analyzer = Stream_Analyzer(
            #     device = 0, # (self.args, 'mic_index', 0),        # Pyaudio (portaudio) device index, defaults to first mic input
            #     rate   = 44100,               # Audio samplerate, None uses the default source settings
            #     FFT_window_size_ms  = 60,    # Window size used for the FFT transform
            #     updates_per_second  = 500,   # How often to read the audio stream for new data
            #     smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
            #     n_frequency_bins = 3, # The FFT features are grouped in bins
            #     visualize = 0,               # Visualize the FFT features with PyGame
            #     verbose   = 0,    # Print running statistics (latency, fps, ...)
            #     height    = 480,     # Height, in pixels, of the visualizer window,
            #     window_ratio = 1  # Float ratio of the visualizer window. e.g. 24/9
            # )

            # print("[main.py] Using device index: ", self.args.mic_index)

            # # Initialize the frequency zoom controller
            # self.frequency_zoom_controller = FrequencyZoomController(
            #     baseline_window_size=10,  # match the client's window size
            #     low_bin_sensitivity=1, #getattr(self.args, 'acid_low_bin_sensitivity', 0.1),
            #     high_bin_sensitivity=1, #getattr(self.args, 'acid_high_bin_sensitivity', 0.1),
            #     min_zoom=0.5,
            #     max_zoom=2,
            #     rebalance_rate=0.1,
            #     activity_threshold=0.25,
            #     amplifying_factor=1000,
            #     enabled=getattr(self.args, 'use_frequency_zoom', False),
            #     debug=True #getattr(self.args, 'debug', False)
            # )

            self.frequency_zoom_controller = BeatZoomController(
                baseline_window_size=30,  # match the client's window size
                baseline_avg_pct=0.3,
                min_zoom=1,
                max_zoom=2,
                smoothing_factor=0.01,
                amplifying_factor=1000,
                energy_amplifier=0.40,
                use_baseline=False,
                max_bin_decay_rate=0.995,
                enabled=getattr(self.args, 'use_frequency_zoom', False),
                debug=True #getattr(self.args, 'debug', False)
            )
            # Enable debug output if in debug mode
            self.frequency_zoom_controller.enable_debug(getattr(self.args, 'debug', False))
            
            # Initialize the LoRA sound controller
            # Determine num_pipes based on whether prompt indexing is enabled
            
            # Parse treble boost factors from config string
            treble_boost_factors = None
            try:
                treble_boost_str = getattr(self.args, 'lora_treble_boost_factors', "1.0,1.0,1.5,2.0,2.5")
                treble_boost_factors = [float(x.strip()) for x in treble_boost_str.split(',')]
                print(f"[main.py] Using frequency bin boost factors: {treble_boost_factors}")
            except (ValueError, AttributeError) as e:
                print(f"[main.py] Error parsing frequency bin boost factors from config: {e}")
                print("[main.py] Using default frequency bin boost factors")
                treble_boost_factors = None

            self.lora_sound_controller = LoraSoundController(
                num_pipes=len(self.pipeline.pipes),
                num_prompts=len(self.prompt_travel_scheduler.prompt_scheduler.prompts),
                enabled=self.use_lora_sound_control,
                debug=getattr(self.args, 'debug', False),
                frequency_bin_boost_factors=treble_boost_factors
            )
            # Enable debug output if in debug mode
            self.lora_sound_controller.enable_debug(getattr(self.args, 'debug', False))
            
            # Initialize test oscillators with config parameters
            self.zoom_oscillator = ZoomOscillator(
                min_zoom=getattr(self.args, 'test_min_zoom', 0.5),
                max_zoom=getattr(self.args, 'test_max_zoom', 1.5),
                zoom_increment=getattr(self.args, 'test_zoom_increment', 0.03),
                stabilize_duration=getattr(self.args, 'test_zoom_stabilize_duration', 3),
                enabled=getattr(self.args, 'use_test_zoom', False),
                debug=getattr(self.args, 'debug', False)
            )
            
            self.shift_oscillator = ShiftOscillator(
                x_max=getattr(self.args, 'test_x_max', 50),
                y_max=getattr(self.args, 'test_y_max', 50),
                x_increment=getattr(self.args, 'test_x_shift_increment', 0),
                y_increment=getattr(self.args, 'test_y_shift_increment', 0),
                enabled=getattr(self.args, 'use_test_shift', False),
                debug=getattr(self.args, 'debug', False)
            )
        self.use_background_removal = getattr(self.args, 'use_background_removal', True)
        
        # Initialize background removal processor
        if self.use_background_removal:
            self.bg_removal_processor = get_bg_removal_processor(device=device.type)
            print("[main.py] Background removal processor initialized")
            
        # Initialize image saver
        self.use_image_saver = getattr(self.args, 'use_image_saver', False)
        if self.use_image_saver:
            self.image_saver = get_image_saver(
                base_dir=getattr(self.args, 'image_save_dir', 'output'),
                image_format=getattr(self.args, 'image_save_format', 'png'),
                quality=getattr(self.args, 'image_save_quality', 95),
                queue_size=getattr(self.args, 'image_save_queue_size', 100),
                enabled=True,
                debug=getattr(self.args, 'debug', False)
            )
            print(f"[main.py] Image saver initialized - session: {self.image_saver.session_dir}")
        
        # Initialize ZMQ context and socket
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind("tcp://*:5555")
        
        self.init_app()

    async def warmup_all_pipes(self):
        """Warmup all pipes by making dummy predictions to pre-trace computational graphs"""
        if not hasattr(self.pipeline, 'pipes'):
            return
            
        num_pipes = len(self.pipeline.pipes)
        print(f"[main.py] Warming up {num_pipes} pipes...")
        
        # Get default input params from curation config, similar to how predict method works
        default_input_params = self.lora_config.get_default_curation_input_params()
        
        if default_input_params:
            print(f"[main.py] Using curation config defaults for warmup: {default_input_params}")
            # Create InputParams instance with curation config defaults
            try:
                default_instance = self.pipeline.InputParams(**default_input_params)
            except Exception as e:
                print(f"[main.py] Error creating InputParams with curation config: {e}")
                print("[main.py] Falling back to pipeline defaults")
                default_instance = self.pipeline.InputParams()
        else:
            print("[main.py] No curation config defaults found, using pipeline defaults")
            default_instance = self.pipeline.InputParams()
        
        dummy_image = Image.new('RGB', (default_instance.width, default_instance.height), color='black')
        
        # Create base parameters using the configured defaults
        base_params = vars(default_instance)
        base_params['width'] = default_instance.width
        base_params['height'] = default_instance.height
        
        start_time = time.time()
        for pipe_idx in range(num_pipes):
            try:
                params = base_params.copy()
                params['pipe_index'] = pipe_idx
                params = SimpleNamespace(**self.pipeline.InputParams(**params).__dict__)
                
                if self.pipeline.Info().input_mode == "image":
                    params.image = dummy_image
                    if self.use_depth_estimator and hasattr(self, 'depth_estimator'):
                        try:
                            params.control_image = self.depth_estimator.get_depth(dummy_image)
                        except:
                            pass
                
                self.pipeline.predict(params)
                
            except Exception as e:
                print(f"[main.py] Error warming up pipe {pipe_idx}: {e}")
                continue
        
        print(f"[main.py] Warmup completed in {time.time() - start_time:.2f}s")

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Use on_event decorators for startup/shutdown
        @self.app.on_event("startup")
        async def startup_event():
            # Startup
            print("Application startup")
            
            # Initialize depth estimator first (before warmup) if enabled
            if self.use_depth_estimator:
                try:
                    print("[main.py] Initializing depth estimator")
                    self.depth_estimator = DepthAnything(
                        engine_path=self.depth_engine_path,
                        device=device.type,
                        grayscale=getattr(self.args, 'depth_grayscale', False),
                        normalized_distance_threshold=getattr(self.args, 'depth_normalized_distance_threshold', 0.225),
                        absolute_min=getattr(self.args, 'depth_absolute_min', 0.0),
                        absolute_max=getattr(self.args, 'depth_absolute_max', 18.0)
                    )
                    print("[main.py] Depth estimator initialized")
                except Exception as e:
                    print(f"[main.py] Error initializing depth estimator: {e}")
                    print("[main.py] Running without depth estimation")
                    self.use_depth_estimator = False
            
            # Warmup all pipes if enabled (after depth estimator is ready)
            if getattr(self.args, 'warmup', True):
                await self.warmup_all_pipes()
            else:
                print("[main.py] Pipe warmup disabled")
            
            # Start image saver if enabled
            if self.use_image_saver:
                await self.image_saver.start()
                print("[main.py] Image saver started")
            
            # Initialize embeddings service if prompt travel is enabled
            print(f"[main.py] Use prompt travel: {self.use_prompt_travel}")
            print(f"[main.py] Has pipeline pipe: {hasattr(self.pipeline, 'pipe')}")
            print(f"[main.py] Has pipeline pipes: {hasattr(self.pipeline, 'pipes')}")
            if self.use_prompt_travel and (hasattr(self.pipeline, 'pipe') or hasattr(self.pipeline, 'pipes')):
                try:
                    print("[main.py] Initializing prompt travel service")
                    # Get the models from the pipeline - handle both single pipe and multiple pipes
                    if hasattr(self.pipeline, 'pipe'):
                        text_encoder = self.pipeline.pipe.text_encoder
                        tokenizer = self.pipeline.pipe.tokenizer
                    else:
                        # Use the first pipe from the array
                        text_encoder = self.pipeline.pipes[0].text_encoder
                        tokenizer = self.pipeline.pipes[0].tokenizer
                    
                    # # Initialize the embeddings service
                    # await embeddings_service.initialize(
                    #     text_encoder=text_encoder,
                    #     tokenizer=tokenizer,
                    #     device=device.type
                    # )
                    
                    # # Start background tasks for the embeddings service
                    # await start_background_tasks()
                    print("[main.py] Prompt travel service initialized and background tasks started")
                except Exception as e:
                    print(f"[main.py] Error initializing embeddings service: {e}")
                    print("[main.py] Running without prompt travel")
                    self.use_prompt_travel = False
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            # Shutdown
            print("Application shutdown")
            
            # Stop image saver if enabled
            if self.use_image_saver:
                await self.image_saver.stop()
                print("[main.py] Image saver stopped")
            
            # No explicit cleanup needed for the async embeddings service
        
        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            
            try:
                while True:
                    loop_start_time = time.time()
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    
                    ######################################################33###########
                    ######## BACKEND BASED PREPROCESSING AND ACID PROCESSING ########33
                    ###############################################################

                    print(f"using prompt travel: {self.use_prompt_travel}")
                    # Process prompt travel requests if enabled
                    if self.use_prompt_travel:
                        print("[main.py] Processing prompt travel requests")
                        # We'll process prompt travel after params is defined in the next_frame section
                    
                    receive_json_start = time.time()
                    data = await self.conn_manager.receive_json(user_id)
                    if self.args.debug:
                        print(f"Time to receive JSON data: {time.time() - receive_json_start:.4f}s")

                    if data["status"] == "next_frame":
                        info = self.pipeline.Info()
                        receive_params_start = time.time()
                        params = await self.conn_manager.receive_json(user_id)

                        params_creation_start = time.time()
                        # Extract acid_settings before converting to SimpleNamespace
                        acid_settings = params.pop("acid_settings", {}) if isinstance(params, dict) else {}
                        params = self.pipeline.InputParams(**params)
                        params = SimpleNamespace(**vars(params))
                        # Add acid_settings back as an attribute
                        setattr(params, 'acid_settings', acid_settings)

                        print(f"[main.py] Received params: {params}")
                        print(f"[main.py] params type: {type(params)}")
                        if self.args.debug:
                            print(f"Time to receive params: {time.time() - receive_params_start:.4f}s")
                        # Update acid processor settings if included in params

                        ########### FREQ DATA FROM FRONTEND ######################33
                        # print(f"[main.py] Handle websocket data - params: {params}")
                        # print(f"[main.py] Use acid settings in params: {params}")
                        if self.use_acid_processor and hasattr(params, 'acid_settings'):
                            acid_settings_start = time.time()
                            acid_settings = getattr(params, 'acid_settings', {})
                            # print(f"[main.py] Handle websocket data - acid_settings: {acid_settings}")
                            self._update_acid_settings(acid_settings)
                            
                            # Process frequency bins if included in settings
                            if isinstance(acid_settings, dict) and "binned_fft" in acid_settings:
                                binned_fft = acid_settings.get("binned_fft")
                                normalized_energies = acid_settings.get("normalized_energies")
                                print(f"[main.py] Handle websocket data - binned_fft: {binned_fft}")
                                print(f"[main.py] Handle websocket data - normalized_energies: {normalized_energies}")
                                if binned_fft is not None:
                                    # Process frequency bins for zoom if acid processor is enabled and test oscillation is disabled
                                    if self.use_acid_processor and not self.zoom_oscillator.enabled:
                                        new_zoom = self.frequency_zoom_controller.process_frequency_bins(normalized_energies)
                                        if self.args.debug:
                                            print(f"[main.py] Updated zoom factor from frequency analysis: {new_zoom:.2f}")
                                        # Apply the updated zoom factor to the acid processor
                                        self.acid_processor.set_zoom_factor(new_zoom)

                                if normalized_energies is not None:
                                    # Process frequency bins for LoRA pipe selection if enabled
                                    if self.use_lora_sound_control:
                                        # Update treble boost factors with current parameters
                                        self.lora_sound_controller.update_frequency_bin_boost_factors(
                                            bass_boost=params.boost_factor_bass,
                                            low_mids_boost=params.boost_factor_low_mids,
                                            mids_boost=params.boost_factor_mids,
                                            high_mids_boost=params.boost_factor_high_mids,
                                            treble_boost=params.boost_factor_treble
                                        )

                                        print(f"[main.py] Updated frequency bin boost factors: {self.lora_sound_controller.frequency_bin_boost_factors}")
                                        
                                        new_pipe_index, new_prompt_index = self.lora_sound_controller.process_frequency_bins(
                                            normalized_energies,
                                            debug=self.args.debug,
                                        )

                                        if self.args.debug:
                                            print(f"[main.py] Updated pipe index from frequency analysis: {new_pipe_index}")
                                            print(f"[main.py] Updated prompt index from frequency analysis: {new_prompt_index}")
                                        # Update the pipe index in params
                                        setattr(params, 'pipe_index', new_pipe_index)
                                        setattr(params, 'prompt_index', new_prompt_index)
                                        

                            if self.args.debug:
                                print(f"Time to process acid settings: {time.time() - acid_settings_start:.4f}s")
                        
                        if self.args.debug:
                            print(f"Time to create params object: {time.time() - params_creation_start:.4f}s")
                        
                        # Process prompt travel requests if enabled
                        if self.use_prompt_travel and getattr(params, 'use_prompt_travel', False):
                            prompt_travel_start = time.time()
                            try:
                                # Set user_id on params for the pipeline to use
                                user_id_str = str(user_id)
                                
                                # Update prompt travel factor with scheduler if enabled
                                if hasattr(self, 'prompt_travel_scheduler') and self.prompt_travel_scheduler.enabled:
                                    # Get the next factor value and seed from the scheduler
                                    scheduler_factor, scheduler_seed = self.prompt_travel_scheduler.update()

                                    print(f"[main.py] Using scheduled prompt travel factor: {scheduler_factor:.2f}")
                                    print(f"[main.py] Using scheduled seed: {scheduler_seed}")
                                    # Use the scheduled factor for prompt travel
                                    setattr(params, 'prompt_travel_factor', scheduler_factor)
                                    setattr(params, 'latent_travel_factor', scheduler_factor)

                                    # Use the scheduled seeds if available
                                    if scheduler_seed is not None:
                                        # Get both current and next seeds for smooth transition
                                        current_seed, next_seed = self.prompt_travel_scheduler.get_seeds()
                                        setattr(params, 'seed', current_seed)
                                        setattr(params, 'target_seed', next_seed)
                                        if self.args.debug:
                                            print(f"[main.py] Using scheduled seeds: current={current_seed}, next={next_seed}")

                                    print(f"[main.py] Prompt scheduler enabled: {self.prompt_travel_scheduler.use_prompt_scheduler}")
                                    
                                    # Use scheduled prompts if prompt scheduler is enabled
                                    if self.prompt_travel_scheduler.use_prompt_scheduler:
                                        # Check if prompt indexing is enabled
                                        if getattr(params, 'use_prompt_indexing', False):
                                            # Use pipe index to select prompts
                                            prompt_index = getattr(params, 'prompt_index', 0)
                                            indexed_prompt = self.prompt_travel_scheduler.get_prompt_by_index(prompt_index)
                                            if indexed_prompt is not None:
                                                setattr(params, 'prompt', indexed_prompt)
                                                setattr(params, 'target_prompt', indexed_prompt)  # Same prompt for both
                                                if self.args.debug:
                                                    print(f"[main.py] Using indexed prompt for prompt index {prompt_index}: {indexed_prompt}")
                                        else:
                                            # Use sequential prompt scheduling
                                            current_prompt, next_prompt = self.prompt_travel_scheduler.get_prompts()
                                            if current_prompt is not None and next_prompt is not None:
                                                setattr(params, 'prompt', current_prompt)
                                                setattr(params, 'target_prompt', next_prompt)
                                                if self.args.debug:
                                                    print(f"[main.py] Using scheduled prompts:")
                                                    print(f"source: {current_prompt}")
                                                    print(f"target: {next_prompt}")
                                    
                                    if self.args.debug:
                                        print(f"[main.py] Using scheduled prompt travel factor: {scheduler_factor:.2f}")
                                
                                # # Queue the prompt travel request
                                # await embeddings_service.process_prompt_travel(
                                #     user_id=user_id_str,
                                #     prompt=getattr(params, 'prompt', ''),
                                #     target_prompt=getattr(params, 'target_prompt', ''),
                                #     factor=getattr(params, 'prompt_travel_factor', 0.0)
                                # )

                                print(f"[main.py] Prompt factor: {getattr(params, 'prompt_travel_factor')}")
                                
                                # # Get any available embeddings and attach them to the params
                                # embeddings = await embeddings_service.get_embeddings(user_id_str)
                                # if embeddings:
                                #     prompt_embeds, negative_prompt_embeds = embeddings
                                #     print(f"[main.py] Got embeddings - prompt shape: {prompt_embeds.shape}")
                                    
                                #     # Attach embeddings to params - using setattr for SimpleNamespace compatibility
                                #     setattr(params, 'prompt_embeds', prompt_embeds)
                                #     setattr(params, 'negative_prompt_embeds', negative_prompt_embeds)
                                #     # print(f"[main.py] Attached embeddings to params: {hasattr(params, 'prompt_embeds')}")
                                # else:
                                #     print(f"[main.py] No embeddings available for user {user_id_str}")
                            except Exception as e:
                                print(f"Error during prompt travel: {e}")
                                # Continue without prompt travel embeddings
                            if self.args.debug:
                                print(f"Time to process prompt travel: {time.time() - prompt_travel_start:.4f}s")
                        
                        if info.input_mode == "image":
                            receive_image_start = time.time()
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if self.args.debug:
                                print(f"Time to receive image data: {time.time() - receive_image_start:.4f}s")

                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            
                            image_processing_start = time.time()
                            params.image = bytes_to_pil(image_data)
                            
                            # Apply acid processing if enabled
                            if self.use_acid_processor and params.image:
                                # print(f"[main.py] Handle websocket data - image: {params.image}")
                                acid_start = time.time()
                                params.image = self._apply_acid_processing(params.image)
                                if self.args.debug:
                                    print(f"Time for acid processing: {time.time() - acid_start:.4f}s")
                                # print(f"[main.py] After acid processing, image type: {type(params.image)}")
                            
                            if self.use_background_removal and params.image:
                                bg_removal_start = time.time()
                                params.image = self._apply_background_removal(params.image)
                                if self.args.debug:
                                    print(f"Time for background removal: {time.time() - bg_removal_start:.4f}s")
                            
                            # Use camera image directly as control image if enabled
                            if getattr(self.args, 'use_camera_as_control', False):
                                camera_control_start = time.time()
                                print("[main.py] Using camera image directly as control image")
                                setattr(params, 'control_image', params.image)
                                if self.args.debug:
                                    print(f"Time for setting camera as control: {time.time() - camera_control_start:.4f}s")
                            # Apply depth estimation if enabled and not using camera as control
                            elif self.use_depth_estimator and params.image and getattr(params, 'use_depth_estimation', True):
                                depth_start = time.time()
                                try:
                                    print("[main.py] Applying depth estimation")
                                    # Get the depth map
                                    depth_map = self.depth_estimator.get_depth(params.image)

                                    # if self.use_background_removal:
                                    #     depth_map = self._apply_background_removal(depth_map)
                                    
                                    # Set the control image in the params
                                    # This is the key part that sets params.control_image for use in the pipeline
                                    setattr(params, 'control_image', depth_map)
                                    
                                    print("[main.py] Depth estimation applied")
                                except Exception as e:
                                    print(f"[main.py] Error during depth estimation: {e}")
                                    # Continue without depth estimation
                                if self.args.debug:
                                    print(f"Time for depth estimation: {time.time() - depth_start:.4f}s")
                            
                            if self.args.debug:
                                print(f"Total image processing time: {time.time() - image_processing_start:.4f}s")

                        update_data_start = time.time()
                        await self.conn_manager.update_data(user_id, params)
                        await self.conn_manager.send_json(user_id, {"status": "wait"})
                        if self.args.debug:
                            print(f"Time to update data and send wait status: {time.time() - update_data_start:.4f}s")
                            print(f"Total loop processing time: {time.time() - loop_start_time:.4f}s")

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            """Stream processed frames"""
            try:
                print(f"[Server] Starting stream for user {user_id}")
                # Start processing loop
                while True:
                    last_time = time.time()
                    await self.conn_manager.send_json(
                        user_id, {"status": "send_frame"}
                    )
                    params = await self.conn_manager.get_latest_data(user_id)
                    
                    if self.args.debug:
                        print(f"All the param stuff time taken: {time.time() - last_time}")

                    last_img_time = time.time()
                    image = self.pipeline.predict(params)
                    if self.args.debug:
                        print(f"Img gen time taken: {time.time() - last_img_time}")

                    # Save image if enabled
                    if self.use_image_saver and image is not None:
                        # Create metadata for the saved image
                        metadata = {
                            'user_id': str(user_id),
                            'timestamp': time.time(),
                            'prompt': getattr(params, 'prompt', ''),
                            'seed': getattr(params, 'seed', ''),
                            'guidance_scale': getattr(params, 'guidance_scale', ''),
                            'strength': getattr(params, 'strength', ''),
                            'pipe_index': getattr(params, 'pipe_index', ''),
                        }
                        # Queue image for saving (non-blocking)
                        await self.image_saver.save_image(image, metadata)

                        # Save depth image if depth estimation was used
                        if self.use_depth_estimator and hasattr(params, 'control_image') and params.control_image is not None:
                            depth_metadata = metadata.copy()
                            depth_metadata['image_type'] = 'depth'
                            # Queue depth image for saving (non-blocking)
                            await self.image_saver.save_image(params.control_image, depth_metadata, filename_suffix='_depth')
                        else:
                            print(f"[main.py] No depth image saved as {params.control_image} is None or depth estimation is not enabled")
                    if self.args.safety_checker:
                        image, has_nsfw_concept = self.safety_checker(image)
                        if has_nsfw_concept:
                            image = None

                    if image is None:
                        continue

                    if self.use_background_removal and getattr(params, 'use_output_bg_removal', False):
                        last_remove_time = time.time()
                        image = self._apply_background_removal(image)
                        after_remove_time = time.time() - last_remove_time
                        if self.args.debug:
                            print(f"Output background removal time taken: {after_remove_time}")
                            
                    # Update acid processor with the diffused image for next processing cycle
                    if self.use_acid_processor:
                        last_acid_time = time.time()
                        # Convert PIL image to numpy array if needed
                        img_diffusion = np.array(image)
                        self.acid_processor.update(img_diffusion)
                        after_acid_time = time.time() - last_acid_time
                        if self.args.debug:
                            print(f"Acid time taken: {after_acid_time}")
                    
                    # Convert PIL to numpy array and send over ZMQ
                    print("[Server] Converting image to numpy array...")
                    image_np = np.array(image)
                    print(f"[Server] Sending image of shape {image_np.shape} over ZMQ...")
                    self.zmq_socket.send(image_np.tobytes())
                    print("[Server] Image sent successfully")
                    
                    if self.args.debug:
                        print(f"Total processing time: {time.time() - last_time}")

                # This will never be reached, but needed for type checking
                return JSONResponse({"status": "connected", "message": "ZMQ stream started"})
            except Exception as e:
                print(f"[Server] Streaming Error: {e}")
                import traceback
                traceback.print_exc()
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                    "current_curation_index": getattr(self.args, 'default_curation_index', 0),
                }
            )

        @self.app.post("/api/update_curation_index")
        async def update_curation_index(request: Request):
            """Update the curation index and reinitialize the pipeline with proper cleanup"""
            try:
                data = await request.json()
                new_curation_index = data.get("curation_index", 0)
                
                # Update the config
                old_index = getattr(self.args, 'default_curation_index', 0)
                # Create new args with updated curation index
                args_dict = self.args._asdict()
                args_dict['default_curation_index'] = new_curation_index
                self.args = Args(**args_dict)
                
                print(f"[main.py] Updating curation index from {old_index} to {new_curation_index}")
                
                # Update the app-level lora_config with new curation index
                print("[main.py] Updating LoRACurationConfig with new curation index...")
                self.lora_config = LoRACurationConfig(
                    self.args.lora_config_dir, 
                    default_curation_index=new_curation_index
                )
                print(f"[main.py] LoRACurationConfig updated with curation index {new_curation_index}")
                
                # Update the PromptTravelScheduler's prompts_file_name if it exists
                if hasattr(self, 'prompt_travel_scheduler') and self.prompt_travel_scheduler is not None:
                    current_config = self.lora_config.get_config_for_curation(self.lora_config.default_curation_key)
                    new_prompts_file_name = current_config.get('prompts_file_name') if current_config else getattr(self.args, 'prompts_file_name', 'glitch')
                    print(f"[main.py] Updating PromptTravelScheduler with new prompts_file_name: {new_prompts_file_name}")
                    self.prompt_travel_scheduler.update_prompts_file_name(new_prompts_file_name)
                
                # Properly cleanup the old pipeline before creating new one
                if hasattr(self, 'pipeline') and self.pipeline is not None:
                    print("[main.py] Cleaning up old pipeline...")
                    
                    # Clear pipeline from GPU memory
                    if hasattr(self.pipeline, 'pipes'):
                        # Multiple pipes case
                        for pipe in self.pipeline.pipes:
                            if hasattr(pipe, 'to'):
                                pipe.to('cpu')
                        del self.pipeline.pipes
                    elif hasattr(self.pipeline, 'pipe'):
                        # Single pipe case
                        if hasattr(self.pipeline.pipe, 'to'):
                            self.pipeline.pipe.to('cpu')
                        del self.pipeline.pipe
                    
                    # Delete the pipeline object
                    del self.pipeline
                    self.pipeline = None
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear GPU cache if using CUDA
                    if device.type == 'cuda':
                        import torch
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print("[main.py] GPU memory cleared")
                
                # Reinitialize the pipeline with new curation index
                print(f"[main.py] Reinitializing pipeline with curation index {new_curation_index}")
                global pipeline

            
                pipeline_class = get_pipeline_class(self.args.pipeline)
                pipeline = pipeline_class(self.args, device, torch_dtype, lora_config=self.lora_config)
                self.pipeline = pipeline
                
                # Get default input params from curation config, similar to how predict method works
                default_input_params = self.lora_config.get_default_curation_input_params()
                
                if default_input_params:
                    print(f"[main.py] Pipeline reinitialized with curation config defaults: {default_input_params}")
                else:
                    print("[main.py] Pipeline reinitialized with no curation config defaults found")
                
                # Apply the curation config defaults to the pipeline's InputParams class
                if default_input_params:
                    try:
                        # Dynamically update the default values in the InputParams model
                        for param_name, param_value in default_input_params.items():
                            if hasattr(self.pipeline.InputParams, '__fields__') and param_name in self.pipeline.InputParams.__fields__:
                                # Update the default value in the field
                                field = self.pipeline.InputParams.__fields__[param_name]
                                field.default = param_value
                                print(f"[main.py] Updated {param_name} default to: {param_value}")
                        print(f"[main.py] Successfully applied {len(default_input_params)} curation config defaults to pipeline")
                    except Exception as e:
                        print(f"[main.py] Error applying curation config defaults: {e}")
                        print("[main.py] Pipeline will use original defaults")
                
                return JSONResponse({
                    "status": "success", 
                    "message": f"Curation index updated to {new_curation_index} and pipeline reinitialized",
                    "new_curation_index": new_curation_index
                })
                
            except Exception as e:
                print(f"[main.py] Error updating curation index: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({
                    "status": "error", 
                    "message": f"Failed to update curation index: {str(e)}"
                }, status_code=500)

        if not os.path.exists("public"):
            os.makedirs("public")

        # self.app.mount(
        #     "/", StaticFiles(directory="frontend/public", html=True), name="public"
        # )
        
    def _update_acid_settings(self, settings):
        """Update acid processor settings from parameters"""
        if not self.use_acid_processor:
            return
            
        # Input processor settings
        if hasattr(settings, 'do_human_seg'):
            self.input_processor.set_human_seg(getattr(settings, 'do_human_seg'))
        if hasattr(settings, 'resizing_factor'):
            self.input_processor.set_resizing_factor_humanseg(getattr(settings, 'resizing_factor'))
        if hasattr(settings, 'do_blur'):
            self.input_processor.set_blur(getattr(settings, 'do_blur'))
        if hasattr(settings, 'brightness'):
            self.input_processor.set_brightness(getattr(settings, 'brightness'))
        if hasattr(settings, 'do_infrared_colorize'):
            self.input_processor.set_infrared_colorize(getattr(settings, 'do_infrared_colorize'))
        
        # Acid processor settings
        if hasattr(settings, 'acid_strength'):
            self.acid_processor.set_acid_strength(getattr(settings, 'acid_strength'))
        if hasattr(settings, 'coef_noise'):
            self.acid_processor.set_coef_noise(getattr(settings, 'coef_noise'))
        if hasattr(settings, 'do_acid_tracers'):
            self.acid_processor.set_acid_tracers(getattr(settings, 'do_acid_tracers'))
        if hasattr(settings, 'acid_strength_foreground'):
            self.acid_processor.set_acid_strength_foreground(getattr(settings, 'acid_strength_foreground'))
        if hasattr(settings, 'zoom_factor') and not hasattr(settings, 'binned_fft'):
            # Only set zoom directly if we're not getting it from frequency analysis
            self.acid_processor.set_zoom_factor(getattr(settings, 'zoom_factor'))
        if hasattr(settings, 'x_shift'):
            self.acid_processor.set_x_shift(getattr(settings, 'x_shift'))
        if hasattr(settings, 'y_shift'):
            self.acid_processor.set_y_shift(getattr(settings, 'y_shift'))
        if hasattr(settings, 'do_acid_wobblers'):
            self.acid_processor.set_do_acid_wobblers(getattr(settings, 'do_acid_wobblers'))
        if hasattr(settings, 'color_matching'):
            self.acid_processor.set_color_matching(getattr(settings, 'color_matching'))
            
        # Update frequency zoom controller settings if present
        if "low_bin_sensitivity" in settings or "high_bin_sensitivity" in settings:
            low_sens = settings.get("low_bin_sensitivity")
            high_sens = settings.get("high_bin_sensitivity")
            if hasattr(self, 'frequency_zoom_controller'):
                self.frequency_zoom_controller.set_sensitivity(
                    low_sensitivity=low_sens, 
                    high_sensitivity=high_sens
                )
                
        # Update test oscillators if needed
        if "use_test_zoom" in settings:
            self.zoom_oscillator.set_enabled(settings["use_test_zoom"])
        if "use_test_shift" in settings:
            self.shift_oscillator.set_enabled(settings["use_test_shift"])
        if "test_x_shift_increment" in settings:
            self.shift_oscillator.set_increments(x_increment=settings["test_x_shift_increment"])
        if "test_y_shift_increment" in settings:
            self.shift_oscillator.set_increments(y_increment=settings["test_y_shift_increment"])
            
        # Update prompt travel scheduler if enabled
        if self.use_prompt_travel and hasattr(self, 'prompt_travel_scheduler'):
            # Enable/disable the scheduler
            if "use_prompt_travel_scheduler" in settings:
                self.prompt_travel_scheduler.set_enabled(settings["use_prompt_travel_scheduler"])
            # Set increment value
            if "prompt_travel_factor_increment" in settings:
                self.prompt_travel_scheduler.set_factor_increment(settings["prompt_travel_factor_increment"])
            # Set oscillation mode
            if "prompt_travel_oscillate" in settings:
                self.prompt_travel_scheduler.set_oscillation(settings["prompt_travel_oscillate"])
            # Set boundaries
            min_factor = settings.get("prompt_travel_min_factor")
            max_factor = settings.get("prompt_travel_max_factor")
            if min_factor is not None or max_factor is not None:
                self.prompt_travel_scheduler.set_boundaries(min_factor, max_factor)
            # Enable/disable prompt scheduler
            if "use_prompt_scheduler" in settings:
                self.prompt_travel_scheduler.set_prompt_scheduler_enabled(settings["use_prompt_scheduler"])
            # Set loop prompts
            if "loop_prompts" in settings:
                if hasattr(self.prompt_travel_scheduler, 'prompt_scheduler') and self.prompt_travel_scheduler.prompt_scheduler is not None:
                    self.prompt_travel_scheduler.prompt_scheduler.set_loop_prompts(settings["loop_prompts"])
            # Reload prompts
            if "reload_prompts" in settings and settings["reload_prompts"]:
                self.prompt_travel_scheduler.reload_prompts()

    def _apply_acid_processing(self, pil_image):
        """Process image with acid processor and return processed PIL image"""

        print("\n[main.py] Applying ACID processing...")
        # Convert PIL to numpy array
        np_image = np.array(pil_image)
        # print(f"[main.py] Input PIL image shape: {np_image.shape}")
        
        # Process with input processor first
        processed_img, mask = self.input_processor.process(np_image)
        # print(f"[main.py] After input processor, image shape: {processed_img.shape}")
        # if mask is not None:
        #     print(f"[main.py] Mask shape: {mask.shape}")
        # else:
        #     print(f"[main.py] No mask generated")

        # acid_img = self.acid_processor.process(processed_img, mask)
        acid_img = self.acid_processor.process_input(processed_img, mask)

        # print(f"[main.py] After acid processor, image shape: {acid_img.shape}")
        
        # Convert back to PIL
        return Image.fromarray(acid_img) #pil_image #Image.fromarray(acid_img)

    def _apply_background_removal(self, pil_image):
        """
        Apply background removal to a PIL image using MODNet.
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with background removed
        """
        if not self.use_background_removal:
            return pil_image
            
        return self.bg_removal_processor.process_image(pil_image)

print(f"Device: {device}")
print(f"torch_dtype: {torch_dtype}")

# Create LoRACurationConfig first at module level
lora_config = LoRACurationConfig(
    config.lora_config_dir, 
    default_curation_index=getattr(config, 'default_curation_index', 0)
)
print(f"[main.py] LoRACurationConfig created at module level with curation index {getattr(config, 'default_curation_index', 0)}")

pipeline_class = get_pipeline_class(config.pipeline)

# Create pipeline with lora_config
pipeline = pipeline_class(config, device, torch_dtype, lora_config=lora_config)

# Create app_instance with both pipeline and lora_config
app_instance = App(config, pipeline, lora_config)
app = app_instance.app

if __name__ == "__main__":
    import uvicorn
    
    try:
        print(f"Starting server on {config.host}:{config.port}")
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=config.reload,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        # Ensure we clean up any global resources
        print("Cleaning up resources...")
        # No additional cleanup needed for the async service
