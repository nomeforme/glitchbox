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
from modules.audio_controller import BeatZoomController, LoraSoundController
# Import fft analyzer
from modules.fft.stream_analyzer import Stream_Analyzer
# Import test oscillators
from utils.test_oscillators import ZoomOscillator, ShiftOscillator
# Import the embeddings service
from modules.prompt_travel.embeddings_service import router as embeddings_router, embeddings_service, start_background_tasks
# Import the prompt travel scheduler
from modules.prompt_scheduler import PromptTravelScheduler
# Import background removal processor
from modules.bg_removal import get_processor as get_bg_removal_processor
# Import the depth estimator
from modules.depth_anything.depth_anything_trt import DepthAnythingTRT

import numpy as np
#import zmq

# Add aiortc imports
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaPlayer # We might need MediaRelay or a custom track for streaming generated frames
from aiortc import VideoStreamTrack # Import VideoStreamTrack
from av import VideoFrame # For converting PIL images to VideoFrames
import fractions # ADD THIS IMPORT

THROTTLE = 1.0 / 120

# A dictionary to store active peer connections
pcs = set()

# Custom VideoStreamTrack to send frames from our pipeline
class LiveVideoStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue()
        self._last_frame_time = time.time()

    async def push_frame(self, pil_image):
        # Convert PIL image to VideoFrame
        # Common formats are 'rgb24' or 'bgr24'. Let's assume 'rgb24'
        # Ensure image is in RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        frame = VideoFrame.from_ndarray(np.array(pil_image), format="rgb24")
        
        # Timestamping is important for WebRTC
        now = time.time()
        if hasattr(self, "_timestamp"):
            self._timestamp += int((now - self._last_frame_time) * 1000 * 1000 * 1000) # nanoseconds
        else:
            self._timestamp = 0 # Initial timestamp, can be relative
        self._last_frame_time = now
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, 1000 * 1000 * 1000) # nanosecond resolution for time_base

        await self._queue.put(frame)

    async def recv(self):
        # This method is called by aiortc to get the next frame
        frame = await self._queue.get()
        return frame

class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        if self.args.safety_checker:
            self.safety_checker = SafetyChecker(device=device.type)
        
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
                loop_prompts=getattr(self.args, 'loop_prompts', True)
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
            self.lora_sound_controller = LoraSoundController(
                num_pipes=len(self.pipeline.pipes),  # Get number of pipes from pipeline
                enabled=self.use_lora_sound_control,
                debug=getattr(self.args, 'debug', False)
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
            
        # Dictionary to store RTCPeerConnection objects, keyed by user_id
        self.peer_connections = {}
        self.video_tracks = {} # Store video tracks for each user
        self.streaming_tasks = {} # Store asyncio tasks for streaming loops
        
        self.init_app()

    async def _cleanup_user_resources(self, user_id: uuid.UUID):
        """Close WebRTC connection, cancel streaming task, and remove user resources."""
        logging.info(f"Cleaning up resources for user {user_id}")
        # Cancel and remove streaming task
        if user_id in self.streaming_tasks:
            task = self.streaming_tasks.pop(user_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logging.info(f"Streaming task for {user_id} cancelled.")
            except Exception as e:
                logging.error(f"Error during streaming task cleanup for {user_id}: {e}")

        # Close and remove peer connection
        if user_id in self.peer_connections:
            pc = self.peer_connections.pop(user_id)
            if pc.signalingState != "closed":
                try:
                    await pc.close()
                    logging.info(f"Peer connection for {user_id} closed.")
                    if pc in pcs:
                        pcs.remove(pc)
                except Exception as e:
                    logging.error(f"Error closing peer connection for {user_id}: {e}")

        # Remove video track reference
        if user_id in self.video_tracks:
            del self.video_tracks[user_id]
            
    async def _video_stream_loop(self, user_id: uuid.UUID):
        """Generates frames and pushes them to the user's WebRTC video track."""
        logging.info(f"Starting video stream loop for user {user_id}")
        try:
            while True:
                # Ensure user is still connected before proceeding
                if not self.conn_manager.check_user(user_id) or user_id not in self.video_tracks:
                    logging.warning(f"User {user_id} disconnected or track missing, stopping stream loop.")
                    break

                # Request the client to send the next frame parameters
                await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                
                # Wait for and get the latest parameters
                # This assumes params are updated via the websocket handler
                params = await self.conn_manager.get_latest_data(user_id)
                if params is None: # Might happen if user disconnects while waiting
                    logging.warning(f"No params received for user {user_id}, stopping stream loop.")
                    break

                if self.args.debug:
                    start_time = time.time()
                    # print(f"[{user_id}] Received params: {params}")
                
                # Generate image using the pipeline
                image = self.pipeline.predict(params)

                if self.args.debug:
                    gen_time = time.time() - start_time
                    # print(f"[{user_id}] Image generation time: {gen_time:.4f}s")

                # Apply safety checker if enabled
                if self.args.safety_checker:
                    image, has_nsfw_concept = self.safety_checker(image)
                    if has_nsfw_concept:
                        logging.warning(f"[{user_id}] NSFW content detected, skipping frame.")
                        image = None # Skip this frame

                if image is None:
                    continue # Skip if image is None (e.g., NSFW)

                # Apply output background removal if enabled in params
                if self.use_background_removal and getattr(params, 'use_output_bg_removal', False):
                    bg_start_time = time.time()
                    image = self._apply_background_removal(image)
                    if self.args.debug:
                        bg_time = time.time() - bg_start_time
                        # print(f"[{user_id}] Output background removal time: {bg_time:.4f}s")
                        
                # Update acid processor state if enabled
                if self.use_acid_processor:
                    acid_start_time = time.time()
                    img_diffusion = np.array(image)
                    self.acid_processor.update(img_diffusion)
                    if self.args.debug:
                        acid_time = time.time() - acid_start_time
                        # print(f"[{user_id}] Acid update time: {acid_time:.4f}s")
                
                # Push the frame to the WebRTC video track
                video_track = self.video_tracks.get(user_id)
                if video_track:
                    try:
                        await video_track.push_frame(image)
                        if self.args.debug:
                            # print(f"[{user_id}] Frame pushed to WebRTC track.")
                            pass
                    except Exception as e:
                        logging.error(f"[{user_id}] Error pushing frame to WebRTC track: {e}")
                        # Consider breaking the loop or handling specific errors
                else:
                    logging.warning(f"[{user_id}] Video track not found, stopping stream loop.")
                    break # Exit loop if track is missing
                
                # Optional: Add a small sleep to prevent tight looping if needed
                # await asyncio.sleep(0.001) 

        except asyncio.CancelledError:
            logging.info(f"Video stream loop for {user_id} cancelled.")
        except WebSocketDisconnect:
            logging.info(f"WebSocket disconnected for {user_id} during streaming.")
        except Exception as e:
            logging.error(f"Error in video stream loop for {user_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logging.info(f"Exiting video stream loop for user {user_id}")
            # Ensure cleanup happens even if loop exits unexpectedly
            await self._cleanup_user_resources(user_id)

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
            
            # Initialize depth estimator if enabled
            if self.use_depth_estimator:
                try:
                    print("[main.py] Initializing depth estimator")
                    self.depth_estimator = DepthAnythingTRT(
                        engine_path=self.depth_engine_path,
                        device=device.type
                    )
                    print("[main.py] Depth estimator initialized")
                except Exception as e:
                    print(f"[main.py] Error initializing depth estimator: {e}")
                    print("[main.py] Running without depth estimation")
                    self.use_depth_estimator = False
            
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
            
            # Close all WebRTC peer connections
            # Coroutine for closing a single PC
            async def close_pc(pc):
                if pc.signalingState != "closed":
                    await pc.close()

            # Gather all close coroutines
            close_tasks = [close_pc(pc) for pc in pcs]
            await asyncio.gather(*close_tasks)
            pcs.clear()
            self.peer_connections.clear()
            self.video_tracks.clear()
            self.streaming_tasks.clear() # Clear streaming tasks dict on shutdown
        
        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            # No finally block needed here, cleanup is handled in handle_websocket_data's finally

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                # This check should ideally be in the caller (websocket_endpoint)
                # but keeping it here for robustness if called directly.
                logging.error(f"User {user_id} not found in ConnectionManager at start of handle_websocket_data")
                # Return or raise appropriate exception if user not found
                # For now, let it proceed, connect might re-add, but this is a symptom of a state issue.
                pass # Or raise HTTPException(status_code=404, detail="User not found") if appropriate here

            last_time = time.time()
            webrtc_connected = False
            
            try:
                while True:
                    # Timeout check
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        logging.warning(f"User {user_id} timed out.")
                        await self.conn_manager.send_json(user_id, {"status": "timeout", "message": "Session timed out"})
                        break 
                    
                    data = await self.conn_manager.receive_json(user_id)
                    if data is None: # Connection might have closed while waiting
                        logging.warning(f"Received None from receive_json for user {user_id}, likely disconnect.")
                        break
                    last_time = time.time() # Reset timeout counter

                    # --- WebRTC Signaling Phase --- 
                    if "sdp" in data and isinstance(data["sdp"], dict) and "type" in data["sdp"]:
                        sdp_data = data["sdp"]
                        if sdp_data["type"] == "offer":
                            if user_id in self.peer_connections:
                                logging.warning(f"[{user_id}] Received offer, but connection already exists. Cleaning up old one.")
                                await self._cleanup_user_resources(user_id)
                                
                            logging.info(f"[{user_id}] Received WebRTC offer.")
                            offer = RTCSessionDescription(sdp=sdp_data["sdp"], type=sdp_data["type"])
                            pc = RTCPeerConnection()
                            self.peer_connections[user_id] = pc
                            pcs.add(pc)

                            @pc.on("icecandidate")
                            async def on_icecandidate(candidate):
                                if candidate:
                                    logging.debug(f"[{user_id}] Sending ICE candidate: {candidate.sdpMid} {candidate.sdpMLineIndex}")
                                    await self.conn_manager.send_json(user_id, {"type": "icecandidate", "candidate": {"candidate": candidate.candidate, "sdpMid": candidate.sdpMid, "sdpMLineIndex": candidate.sdpMLineIndex}})
                            
                            @pc.on("connectionstatechange")
                            async def on_connectionstatechange():
                                logging.info(f"[{user_id}] WebRTC Connection state is {pc.connectionState}")
                                nonlocal webrtc_connected # Ensure this is the intended webrtc_connected
                                if pc.connectionState == "connected":
                                    webrtc_connected = True
                                    logging.info(f"[{user_id}] WebRTC connection established.")
                                elif pc.connectionState in ["failed", "closed", "disconnected"]:
                                    webrtc_connected = False
                                    # Important: Schedule cleanup but don't await here if it can lead to deadlocks
                                    # or re-entrant issues within event handlers.
                                    # asyncio.create_task(self._cleanup_user_resources(user_id))
                                    # For simplicity now, direct call, but be mindful of async event handler contexts.
                                    await self._cleanup_user_resources(user_id) # Potential re-entry if cleanup also sends ws messages
                                    
                            video_track = LiveVideoStreamTrack()
                            pc.addTrack(video_track)
                            self.video_tracks[user_id] = video_track
                            logging.debug(f"[{user_id}] Added LiveVideoStreamTrack.")

                            await pc.setRemoteDescription(offer)
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)

                            logging.debug(f"[{user_id}] Sending WebRTC answer.")
                            await self.conn_manager.send_json(user_id, {"type": "answer", "sdp": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}})
                        # Server typically does not receive sdp type "answer" first, but good to have structure
                        # elif sdp_data["type"] == "answer":
                        #    pass 

                    elif "candidate" in data and user_id in self.peer_connections:
                        candidate_data = data["candidate"]
                        # Ensure candidate_data is not None and has the 'candidate' key itself
                        if candidate_data and isinstance(candidate_data, dict) and candidate_data.get("candidate") is not None:
                            try:
                                candidate = RTCIceCandidate(
                                    sdpMid=candidate_data.get("sdpMid"),
                                    sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                                    candidate=candidate_data["candidate"]
                                )
                                logging.debug(f"[{user_id}] Received ICE candidate: {candidate.sdpMid} {candidate.sdpMLineIndex}")
                                await self.peer_connections[user_id].addIceCandidate(candidate)
                            except Exception as e:
                                logging.warning(f"[{user_id}] Error adding ICE candidate: {e} - Candidate: {candidate_data}")
                        else:
                             logging.warning(f"[{user_id}] Received invalid or null ICE candidate message: {data}")

                    # --- Application Logic Phase --- (Check for "status" key now)
                    elif "status" in data:
                        if data["status"] == "next_frame":
                            if not webrtc_connected or user_id not in self.peer_connections:
                                logging.warning(f"[{user_id}] Received next_frame but WebRTC not connected/ready. Ignoring.")
                                continue 
                                
                            if user_id not in self.streaming_tasks or self.streaming_tasks[user_id].done():
                                 logging.info(f"[{user_id}] Received next_frame. Starting/Restarting stream loop.")
                                 # Cancel previous task if it exists but is done with an error
                                 if user_id in self.streaming_tasks:
                                     del self.streaming_tasks[user_id]
                                 task = asyncio.create_task(self._video_stream_loop(user_id))
                                 self.streaming_tasks[user_id] = task
                            else:
                                logging.debug(f"[{user_id}] Stream loop already running.")
                            
                            # Params are received *after* "next_frame"
                            params_dict = await self.conn_manager.receive_json(user_id)
                            if params_dict is None: 
                                logging.warning(f"[{user_id}] Did not receive params after next_frame. Disconnecting.")
                                break
                            
                            # ... (rest of your existing params processing logic) ...
                            acid_settings = params_dict.pop("acid_settings", {}) if isinstance(params_dict, dict) else {}
                            try:
                                params = self.pipeline.InputParams(**params_dict)
                                params = SimpleNamespace(**vars(params))
                                setattr(params, 'acid_settings', acid_settings)
                            except Exception as e:
                                logging.error(f"[{user_id}] Error parsing parameters: {e} - Params Data: {params_dict}")
                                continue 

                            # ... (Acid processing, Prompt Travel, Input Image processing) ...
                            if self.use_acid_processor and hasattr(params, 'acid_settings'):
                                current_acid_settings = getattr(params, 'acid_settings', {})
                                self._update_acid_settings(current_acid_settings)
                                if isinstance(current_acid_settings, dict):
                                    if "binned_fft" in current_acid_settings:
                                        binned_fft = current_acid_settings.get("binned_fft")
                                        if binned_fft is not None and self.use_acid_processor and not self.zoom_oscillator.enabled:
                                            new_zoom = self.frequency_zoom_controller.process_frequency_bins(binned_fft)
                                            self.acid_processor.set_zoom_factor(new_zoom)
                                    if "normalized_energies" in current_acid_settings:
                                        normalized_energies = current_acid_settings.get("normalized_energies")
                                        if normalized_energies is not None and self.use_lora_sound_control:
                                            new_pipe_index = self.lora_sound_controller.process_frequency_bins(normalized_energies)
                                            setattr(params, 'pipe_index', new_pipe_index)
                            
                            if self.use_prompt_travel and getattr(params, 'use_prompt_travel', False):
                                try:
                                    user_id_str = str(user_id)
                                    if hasattr(self, 'prompt_travel_scheduler') and self.prompt_travel_scheduler.enabled:
                                        scheduler_factor, scheduler_seed = self.prompt_travel_scheduler.update()
                                        setattr(params, 'prompt_travel_factor', scheduler_factor)
                                        setattr(params, 'latent_travel_factor', scheduler_factor)
                                        if scheduler_seed is not None:
                                            current_seed, next_seed = self.prompt_travel_scheduler.get_seeds()
                                            setattr(params, 'seed', current_seed)
                                            setattr(params, 'target_seed', next_seed)
                                        if self.prompt_travel_scheduler.use_prompt_scheduler:
                                            current_prompt, next_prompt = self.prompt_travel_scheduler.get_prompts()
                                            if current_prompt is not None and next_prompt is not None:
                                                setattr(params, 'prompt', current_prompt)
                                                setattr(params, 'target_prompt', next_prompt)
                                        
                                    await embeddings_service.process_prompt_travel(
                                        user_id=user_id_str,
                                        prompt=getattr(params, 'prompt', ''),
                                        target_prompt=getattr(params, 'target_prompt', ''),
                                        factor=getattr(params, 'prompt_travel_factor', 0.0)
                                    )
                                    embeddings = await embeddings_service.get_embeddings(user_id_str)
                                    if embeddings:
                                        prompt_embeds, negative_prompt_embeds = embeddings
                                        setattr(params, 'prompt_embeds', prompt_embeds)
                                        setattr(params, 'negative_prompt_embeds', negative_prompt_embeds)
                                except Exception as e:
                                    logging.error(f"[{user_id}] Error during prompt travel processing: {e}")

                            info = self.pipeline.Info()
                            if info.input_mode == "image":
                                image_data = await self.conn_manager.receive_bytes(user_id)
                                if len(image_data) == 0:
                                    logging.warning(f"[{user_id}] Received empty image data for 'image' mode frame.")
                                    # Decide if we should skip or use a placeholder if pipeline allows
                                    # For now, pipeline might fail if it strictly expects an image here
                                    params.image = None # Or some default PIL.Image.new('RGB', (W,H), color='grey')
                                else:
                                    pil_image = bytes_to_pil(image_data)
                                    processed_image = pil_image
                                    if self.use_acid_processor and processed_image:
                                        processed_image = self._apply_acid_processing(processed_image) 
                                    if self.use_background_removal and processed_image:
                                        processed_image = self._apply_background_removal(processed_image) 
                                    if self.use_depth_estimator and processed_image and getattr(params, 'use_depth_estimation', True):
                                        try:
                                            depth_map = self.depth_estimator.get_depth(processed_image)
                                            setattr(params, 'control_image', depth_map)
                                        except Exception as e:
                                            logging.error(f"[{user_id}] Error during depth estimation: {e}")
                                    params.image = processed_image
                            
                            await self.conn_manager.update_data(user_id, params)
                            
                        elif data["status"] == "disconnect":
                            logging.info(f"[{user_id}] Received disconnect signal from client.")
                            break
                        # elif data["status"] == "connected" or data["status"] == "wait":
                        #    logging.debug(f"[{user_id}] Received server status message: {data['status']}. Ignoring.")
                        #    pass # Client handles these, server can ignore if they come through WS for some reason
                        else:
                            logging.warning(f"[{user_id}] Received WebSocket message with unhandled 'status': {data}")
                    
                    else: # Message does not contain 'sdp', 'candidate', or 'status' keys at the top level
                        logging.warning(f"[{user_id}] Received unknown WebSocket message format: {data}")

                # End of while True loop
            except WebSocketDisconnect:
                logging.info(f"WebSocket disconnected for user {user_id}.")
            except Exception as e:
                logging.error(f"WebSocket Error for user {user_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await self._cleanup_user_resources(user_id)
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User {user_id} fully disconnected and cleaned up.")

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

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
pipeline_class = get_pipeline_class(config.pipeline)
pipeline = pipeline_class(config, device, torch_dtype)
app_instance = App(config, pipeline)
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
