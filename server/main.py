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
import torch
# Import the acid processor
from modules.acid_processor import AcidProcessor, InputImageProcessor
# Import the frequency zoom controller
from modules.audio_controller import BeatZoomController
# Import fft analyzer
from modules.fft.stream_analyzer import Stream_Analyzer
# Import test oscillators
from utils.test_oscillators import ZoomOscillator, ShiftOscillator
# Import the embeddings service
from modules.prompt_travel.embeddings_service import router as embeddings_router, embeddings_service, start_background_tasks
# Import the prompt travel scheduler
from modules.prompt_travel_scheduler import PromptTravelScheduler

# # Import background removal
from rembg import remove

import numpy as np


THROTTLE = 1.0 / 120


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        if self.args.safety_checker:
            self.safety_checker = SafetyChecker(device=device.type)
        
        # Initialize prompt travel service
        self.use_prompt_travel = getattr(self.args, 'use_prompt_travel', False)
        if self.use_prompt_travel:
            print("[main.py] Prompt travel service will be initialized on startup")
            # The actual initialization happens in the startup event
            
            # Initialize prompt travel scheduler
            self.prompt_travel_scheduler = PromptTravelScheduler(
                min_factor=getattr(self.args, 'prompt_travel_min_factor', 0.0),
                max_factor=getattr(self.args, 'prompt_travel_max_factor', 1.0),
                factor_increment=getattr(self.args, 'prompt_travel_factor_increment', 0.025),
                stabilize_duration=getattr(self.args, 'prompt_travel_stabilize_duration', 3),
                oscillate=getattr(self.args, 'prompt_travel_oscillate', True),
                enabled=getattr(self.args, 'use_prompt_travel_scheduler', False),
                debug=getattr(self.args, 'debug', False)
            )
        
        # Initialize acid processors
        self.use_acid_processor = getattr(self.args, 'use_acid_processor', False)
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
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include the embeddings router if prompt travel is enabled
        if self.use_prompt_travel:
            self.app.include_router(embeddings_router)
        
        # Use on_event decorators for startup/shutdown
        @self.app.on_event("startup")
        async def startup_event():
            # Startup
            print("Application startup")
            
            # Initialize embeddings service if prompt travel is enabled
            if self.use_prompt_travel and hasattr(self.pipeline, 'pipe'):
                try:
                    print("[main.py] Initializing prompt travel service")
                    # Get the models from the pipeline
                    text_encoder = self.pipeline.pipe.text_encoder
                    tokenizer = self.pipeline.pipe.tokenizer
                    
                    # Initialize the embeddings service
                    await embeddings_service.initialize(
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        device=device.type
                    )
                    
                    # Start background tasks for the embeddings service
                    await start_background_tasks()
                    print("[main.py] Prompt travel service initialized and background tasks started")
                except Exception as e:
                    print(f"[main.py] Error initializing embeddings service: {e}")
                    print("[main.py] Running without prompt travel")
                    self.use_prompt_travel = False
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            # Shutdown
            print("Application shutdown")
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
                    
                    data = await self.conn_manager.receive_json(user_id)

                    if data["status"] == "next_frame":
                        info = pipeline.Info()
                        params = await self.conn_manager.receive_json(user_id)
                        # Update acid processor settings if included in params

                        ########### FREQ DATA FROM FRONTEND ######################33
                        # print(f"[main.py] Handle websocket data - params: {params}")
                        # print(f"[main.py] Use acid settings in params: {params}")
                        if self.use_acid_processor and "acid_settings" in params:
                            acid_settings = params.pop("acid_settings", {})
                            # print(f"[main.py] Handle websocket data - acid_settings: {acid_settings}")
                            self._update_acid_settings(acid_settings)
                            
                            # Process frequency bins if included in settings
                            if "binned_fft" in acid_settings and self.use_acid_processor and not self.zoom_oscillator.enabled:
                                # Only process FFT data if test oscillation is disabled
                                binned_fft = acid_settings.get("binned_fft")
                                print(f"[main.py] Handle websocket data - binned_fft: {binned_fft}")
                                if binned_fft is not None:
                                    # Process the frequency bins and get updated zoom factor
                                    new_zoom = self.frequency_zoom_controller.process_frequency_bins(binned_fft)
                                    if self.args.debug:
                                        print(f"[main.py] Updated zoom factor from frequency analysis: {new_zoom:.2f}")
                                    # Apply the updated zoom factor to the acid processor
                                    self.acid_processor.set_zoom_factor(new_zoom)
                        
                        params = pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.dict())
                        
                        # Process prompt travel requests if enabled
                        if self.use_prompt_travel and getattr(params, 'use_prompt_travel', False):
                            try:
                                # Set user_id on params for the pipeline to use
                                user_id_str = str(user_id)
                                
                                # Update prompt travel factor with scheduler if enabled
                                if hasattr(self, 'prompt_travel_scheduler') and self.prompt_travel_scheduler.enabled:
                                    # Get the next factor value and seed from the scheduler
                                    scheduler_factor, scheduler_seed = self.prompt_travel_scheduler.update()
                                    # Use the scheduled factor for prompt travel
                                    setattr(params, 'prompt_travel_factor', scheduler_factor)
                                    
                                    # Use the scheduled seed if available
                                    if scheduler_seed is not None:
                                        setattr(params, 'seed', scheduler_seed)
                                        if self.args.debug:
                                            print(f"[main.py] Using scheduled seed: {scheduler_seed}")
                                    
                                    if self.args.debug:
                                        print(f"[main.py] Using scheduled prompt travel factor: {scheduler_factor:.2f}")
                                
                                # Queue the prompt travel request
                                await embeddings_service.process_prompt_travel(
                                    user_id=user_id_str,
                                    prompt=getattr(params, 'prompt', ''),
                                    target_prompt=getattr(params, 'target_prompt', ''),
                                    factor=getattr(params, 'prompt_travel_factor', 0.0)
                                )

                                print(f"[main.py] Prompt factor: {getattr(params, 'prompt_travel_factor')}")
                                
                                # Get any available embeddings and attach them to the params
                                embeddings = await embeddings_service.get_embeddings(user_id_str)
                                if embeddings:
                                    prompt_embeds, negative_prompt_embeds = embeddings
                                    print(f"[main.py] Got embeddings - prompt shape: {prompt_embeds.shape}")
                                    
                                    # Attach embeddings to params - using setattr for SimpleNamespace compatibility
                                    setattr(params, 'prompt_embeds', prompt_embeds)
                                    setattr(params, 'negative_prompt_embeds', negative_prompt_embeds)
                                    # print(f"[main.py] Attached embeddings to params: {hasattr(params, 'prompt_embeds')}")
                                else:
                                    print(f"[main.py] No embeddings available for user {user_id_str}")
                            except Exception as e:
                                print(f"Error during prompt travel: {e}")
                                # Continue without prompt travel embeddings
                        
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)

                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            params.image = bytes_to_pil(image_data)
                            
                            # Apply acid processing if enabled
                            if self.use_acid_processor and params.image:
                                # print(f"[main.py] Handle websocket data - image: {params.image}")
                                params.image = self._apply_acid_processing(params.image)
                                # print(f"[main.py] After acid processing, image type: {type(params.image)}")
                            if self.use_background_removal and params.image:
                                params.image = self._apply_background_removal(params.image)

                        await self.conn_manager.update_data(user_id, params)
                        await self.conn_manager.send_json(user_id, {"status": "wait"})

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:

                async def generate():
                    last_params = SimpleNamespace()
                    while True:
                        last_time = time.time()
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        
                        # Compare params while excluding tensor attributes to avoid comparison error
                        params_equal = False
                        if params is not None and hasattr(params, '__dict__') and hasattr(last_params, '__dict__'):
                            # Create filtered dictionaries excluding tensor attributes
                            params_dict = {k: v for k, v in params.__dict__.items() 
                                          if not isinstance(v, torch.Tensor)}
                            last_params_dict = {k: v for k, v in last_params.__dict__.items() 
                                               if not isinstance(v, torch.Tensor)}
                            params_equal = params_dict == last_params_dict
                            
                        if params_equal or params is None:
                            await asyncio.sleep(THROTTLE)
                            continue
                            
                        last_params: SimpleNamespace = params
                        last_img_time = time.time()
                        image = pipeline.predict(params)
                        if self.args.debug:
                            print(f"Img gen time taken: {time.time() - last_img_time}")

                        if self.args.safety_checker:
                            image, has_nsfw_concept = self.safety_checker(image)
                            if has_nsfw_concept:
                                image = None

                        if image is None:
                            continue
                            
                        # Update acid processor with the diffused image for next processing cycle
                        if self.use_acid_processor:
                            # Convert PIL image to numpy array if needed
                            img_diffusion = np.array(image)
                            self.acid_processor.update(img_diffusion)
                            
                        frame = pil_to_frame(image)
                        # frame = pil_to_frame(params.acid_image)

                        yield frame
                        # https://bugs.chromium.org/p/chromium/issues/detail?id=1250396
                        if not is_firefox(request.headers["user-agent"]):
                            yield frame
                        if self.args.debug:
                            print(f"Websocket Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
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
                }
            )

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount(
            "/", StaticFiles(directory="frontend/public", html=True), name="public"
        )
        
    def _update_acid_settings(self, settings):
        """Update acid processor settings from parameters"""
        if not self.use_acid_processor:
            return
            
        # Input processor settings
        if "do_human_seg" in settings:
            self.input_processor.set_human_seg(settings["do_human_seg"])
        if "resizing_factor" in settings:
            self.input_processor.set_resizing_factor_humanseg(settings["resizing_factor"])
        if "do_blur" in settings:
            self.input_processor.set_blur(settings["do_blur"])
        if "brightness" in settings:
            self.input_processor.set_brightness(settings["brightness"])
        if "do_infrared_colorize" in settings:
            self.input_processor.set_infrared_colorize(settings["do_infrared_colorize"])
        
        # Acid processor settings
        if "acid_strength" in settings:
            self.acid_processor.set_acid_strength(settings["acid_strength"])
        if "coef_noise" in settings:
            self.acid_processor.set_coef_noise(settings["coef_noise"])
        if "do_acid_tracers" in settings:
            self.acid_processor.set_acid_tracers(settings["do_acid_tracers"])
        if "acid_strength_foreground" in settings:
            self.acid_processor.set_acid_strength_foreground(settings["acid_strength_foreground"])
        if "zoom_factor" in settings and "binned_fft" not in settings:
            # Only set zoom directly if we're not getting it from frequency analysis
            self.acid_processor.set_zoom_factor(settings["zoom_factor"])
        if "x_shift" in settings:
            self.acid_processor.set_x_shift(settings["x_shift"])
        if "y_shift" in settings:
            self.acid_processor.set_y_shift(settings["y_shift"])
        if "do_acid_wobblers" in settings:
            self.acid_processor.set_do_acid_wobblers(settings["do_acid_wobblers"])
        if "color_matching" in settings:
            self.acid_processor.set_color_matching(settings["color_matching"])
            
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
        #return pil_image
        return remove(pil_image)

print(f"Device: {device}")
print(f"torch_dtype: {torch_dtype}")
pipeline_class = get_pipeline_class(config.pipeline)
pipeline = pipeline_class(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
    import uvicorn
    
    try:
        print(f"Starting server on {config.host}:{config.port}")
        uvicorn.run(
            "main:app",
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
