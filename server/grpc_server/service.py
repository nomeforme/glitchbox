"""
gRPC service implementation for generation control.

This module implements the GenerationControlServicer which handles
all incoming gRPC requests and queues parameter updates to be applied
on the next frame cycle.
"""

import threading
from typing import Dict, Any, Optional, Callable
import logging

# Import generated protobuf classes (will be generated from proto file)
try:
    from . import generation_control_pb2 as pb2
    from . import generation_control_pb2_grpc as pb2_grpc
except ImportError:
    # Fallback for when running before proto compilation
    pb2 = None
    pb2_grpc = None


class GenerationControlServicer:
    """
    gRPC servicer that handles control commands for real-time generation.

    Uses a pending params pattern where updates are queued and then
    applied atomically on the next frame processing cycle.
    """

    def __init__(self,
                 on_curation_switch: Optional[Callable[[int], None]] = None,
                 debug: bool = False,
                 sticky_frames: int = 5):
        """
        Initialize the servicer.

        Args:
            on_curation_switch: Callback function for curation switches
            debug: Enable debug logging
            sticky_frames: Number of frames gRPC params should override frontend (default 5)
        """
        self._lock = threading.RLock()
        self._pending_params: Dict[str, Any] = {}
        self._current_state: Dict[str, Any] = {}
        self._on_curation_switch = on_curation_switch
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        # Direct flag for headless loop — set by gRPC thread, read by async loop
        self._headless_generate = False
        self._headless_start_time = 0
        self._headless_duration = 0  # 0 = unlimited
        self._headless_frame_count = 0
        self._headless_target_frames = 0
        self._headless_done = False

        # Sticky params: params that override frontend for N frames
        # Format: {param_name: (value, frames_remaining)}
        self._sticky_params: Dict[str, tuple] = {}
        self._sticky_frames = sticky_frames

        # Prompt transition state for smooth blending
        self._prompt_transition = {
            'active': False,
            'source_prompt': '',
            'target_prompt': '',
            'current_frame': 0,
            'total_frames': 0,
            'default_frames': 30,  # Default transition frames (0-100)
        }

        # Prompt journey state (for headless autonomous generation)
        self._journey = {
            'active': False,
            'completed': False,
            'prompts': [],
            'transition_frames': 120,
            'hold_frames': 30,
            'loop': False,
            'current_frame': 0,
            'total_frames': 0,
            'current_segment': 0,
            'total_segments': 0,
            'current_prompt': '',
            'target_prompt': '',
        }

        # Initialize default state
        self._init_default_state()

    def _init_default_state(self):
        """Initialize the current state with default values."""
        self._current_state = {
            # Prompt state
            'prompt': '',
            'target_prompt': '',
            'prompt_travel_factor': 0.0,

            # Generation params
            'seed': -1,
            'num_inference_steps': 4,
            'guidance_scale': 1.0,
            'strength': 0.5,
            'width': 512,
            'height': 512,
            'temporal_coherence': 0.03,
            'temporal_coherence_latent': 0.03,

            # ControlNet params
            'controlnet_scale': 1.0,
            'controlnet_start': 0.0,
            'controlnet_end': 1.0,

            # LoRA params
            'lora_scale': 1.0,
            'pipe_index': 0,

            # Prompt travel params
            'prompt_travel_enabled': False,
            'prompt_travel_min_factor': 0.0,
            'prompt_travel_max_factor': 1.0,
            'prompt_travel_factor_increment': 0.025,

            # Acid params
            'acid_strength': 0.4,
            'acid_zoom_factor': 1.0,
            'acid_x_shift': 0,
            'acid_y_shift': 0,

            # Curation
            'curation_index': 0,
        }

    def _log(self, message: str):
        """Log a message if debug is enabled."""
        if self._debug:
            self._logger.info(f"[gRPC Service] {message}")
            print(f"[gRPC Service] {message}")

    def has_pending_params(self) -> bool:
        """Check if there are any pending parameters without consuming them."""
        with self._lock:
            return bool(self._pending_params)

    def get_and_clear_pending_params(self) -> Dict[str, Any]:
        """
        Get all pending parameters and clear the pending queue.

        This should be called at the start of each frame cycle to get
        any gRPC-submitted parameter updates.

        Returns:
            Dictionary of parameter updates to apply
        """
        with self._lock:
            params = self._pending_params.copy()
            self._pending_params.clear()
            if params:
                self._log(f"Returning pending params: {params}")
            return params

    def update_current_state(self, params: Dict[str, Any]):
        """
        Update the current state with new values.

        This should be called after parameters are applied to keep
        the state in sync.

        Args:
            params: Dictionary of parameter values to update
        """
        with self._lock:
            for key, value in params.items():
                if key in self._current_state:
                    self._current_state[key] = value

    def _queue_params(self, params: Dict[str, Any]):
        """Queue parameter updates to be applied on next frame."""
        with self._lock:
            self._pending_params.update(params)
            # Also make these params "sticky" to override frontend for several frames
            for key, value in params.items():
                self._sticky_params[key] = (value, self._sticky_frames)
            self._log(f"Queued params: {params}")

    def get_sticky_overrides(self) -> Dict[str, Any]:
        """
        Get current sticky param values that should override frontend values.

        These are params set by gRPC that should persist for several frames
        to prevent the frontend's old values from overwriting them before
        the frontend receives the sync update.

        Returns:
            Dictionary of param names to values that should override frontend
        """
        with self._lock:
            overrides = {}
            for key, (value, frames) in self._sticky_params.items():
                if frames > 0:
                    overrides[key] = value
            return overrides

    def decrement_sticky_frames(self):
        """
        Decrement the frame counter for all sticky params.

        Should be called once per frame cycle. When a param's counter
        reaches 0, it will no longer override frontend values.
        """
        with self._lock:
            expired = []
            for key, (value, frames) in self._sticky_params.items():
                if frames > 1:
                    self._sticky_params[key] = (value, frames - 1)
                else:
                    expired.append(key)

            for key in expired:
                del self._sticky_params[key]
                self._log(f"Sticky param expired: {key}")

    def clear_sticky_param(self, param_name: str):
        """
        Clear a sticky param immediately (e.g., when frontend confirms sync).

        Args:
            param_name: The name of the param to clear from sticky list
        """
        with self._lock:
            if param_name in self._sticky_params:
                del self._sticky_params[param_name]
                self._log(f"Sticky param cleared: {param_name}")

    # Prompt Transition Methods

    def start_prompt_transition(self, new_prompt: str, transition_frames: int = None):
        """
        Start a smooth transition to a new prompt.

        Args:
            new_prompt: The target prompt to transition to
            transition_frames: Number of frames for the transition (0-100)
        """
        with self._lock:
            if transition_frames is None:
                transition_frames = self._prompt_transition['default_frames']

            # Clamp to 0-100
            transition_frames = max(0, min(100, transition_frames))

            if transition_frames == 0:
                # Instant change, no transition
                self._prompt_transition['active'] = False
                self._prompt_transition['source_prompt'] = new_prompt
                self._prompt_transition['target_prompt'] = new_prompt
                self._prompt_transition['current_frame'] = 0
                self._prompt_transition['total_frames'] = 0
                self._log(f"Instant prompt change to: {new_prompt[:50]}...")
            else:
                # Get current effective prompt as source
                if self._prompt_transition['active']:
                    # Mid-transition: use interpolated value as new source
                    source = self.get_interpolated_prompt()[0]
                else:
                    # Use current prompt as source
                    source = self._current_state.get('prompt', '')

                self._prompt_transition['active'] = True
                self._prompt_transition['source_prompt'] = source
                self._prompt_transition['target_prompt'] = new_prompt
                self._prompt_transition['current_frame'] = 0
                self._prompt_transition['total_frames'] = transition_frames
                self._log(f"Starting {transition_frames}-frame transition: '{source[:30]}...' -> '{new_prompt[:30]}...'")

    def get_interpolated_prompt(self) -> tuple:
        """
        Get the current interpolated prompt state.

        Returns:
            Tuple of (source_prompt, target_prompt, interpolation_factor)
            - If no transition active: (current_prompt, current_prompt, 1.0)
            - If transition active: (source, target, progress 0.0-1.0)
        """
        with self._lock:
            if not self._prompt_transition['active']:
                current = self._current_state.get('prompt', '')
                return (current, current, 1.0)

            source = self._prompt_transition['source_prompt']
            target = self._prompt_transition['target_prompt']
            current = self._prompt_transition['current_frame']
            total = self._prompt_transition['total_frames']

            if total <= 0:
                return (target, target, 1.0)

            progress = min(1.0, current / total)
            return (source, target, progress)

    def advance_prompt_transition(self) -> bool:
        """
        Advance the prompt transition by one frame.

        Should be called once per frame cycle.

        Returns:
            True if transition is still active, False if completed
        """
        with self._lock:
            if not self._prompt_transition['active']:
                return False

            self._prompt_transition['current_frame'] += 1

            if self._prompt_transition['current_frame'] >= self._prompt_transition['total_frames']:
                # Transition complete
                final_prompt = self._prompt_transition['target_prompt']
                self._prompt_transition['active'] = False
                self._prompt_transition['source_prompt'] = final_prompt
                self._current_state['prompt'] = final_prompt
                self._log(f"Transition complete: {final_prompt[:50]}...")
                return False

            return True

    def get_transition_state(self) -> Dict[str, Any]:
        """
        Get the current transition state for status queries.

        Returns:
            Dictionary with transition state info
        """
        with self._lock:
            total = self._prompt_transition['total_frames']
            current = self._prompt_transition['current_frame']
            progress = (current / total) if total > 0 else 1.0

            return {
                'active': self._prompt_transition['active'],
                'progress': min(1.0, progress),
                'frames_remaining': max(0, total - current),
                'total_frames': total,
                'default_frames': self._prompt_transition['default_frames'],
            }

    def set_default_transition_frames(self, frames: int):
        """
        Set the default number of transition frames.

        Args:
            frames: Default frames (0-100) for transitions without explicit frame count
        """
        with self._lock:
            self._prompt_transition['default_frames'] = max(0, min(100, frames))
            self._log(f"Default transition frames set to: {frames}")

    # gRPC Service Methods

    def SetPrompt(self, request, context):
        """Handle SetPrompt RPC."""
        self._log(f"SetPrompt called: prompt='{request.prompt[:50] if request.prompt else ''}...'")

        # Check if this is a transition request
        if request.HasField('transition_frames'):
            transition_frames = request.transition_frames
            self._log(f"Transition requested: {transition_frames} frames")

            # Start the smooth transition
            self.start_prompt_transition(request.prompt, transition_frames)

            # Queue the transition state to be applied
            source, target, factor = self.get_interpolated_prompt()
            params = {
                'prompt': source,
                'target_prompt': target,
                'prompt_travel_factor': factor,
                'use_prompt_travel': True,
                '_grpc_transition_active': True,  # Internal flag for main.py
            }
            self._queue_params(params)

            return pb2.ControlResponse(
                success=True,
                message=f"Prompt transition started: {transition_frames} frames to '{request.prompt[:30]}...'"
            )

        # Standard prompt update (instant or with explicit target/factor)
        params = {'prompt': request.prompt}

        if request.HasField('target_prompt'):
            params['target_prompt'] = request.target_prompt

        if request.HasField('prompt_travel_factor'):
            params['prompt_travel_factor'] = request.prompt_travel_factor

        self._queue_params(params)

        return pb2.ControlResponse(
            success=True,
            message=f"Prompt queued: '{request.prompt[:30]}...'"
        )

    def SetGenerationParams(self, request, context):
        """Handle SetGenerationParams RPC."""
        self._log("SetGenerationParams called")

        params = {}

        if request.HasField('seed'):
            params['seed'] = request.seed
        if request.HasField('num_inference_steps'):
            params['num_inference_steps'] = request.num_inference_steps
        if request.HasField('guidance_scale'):
            params['guidance_scale'] = request.guidance_scale
        if request.HasField('strength'):
            params['strength'] = request.strength
        if request.HasField('width'):
            params['width'] = request.width
        if request.HasField('height'):
            params['height'] = request.height
        if request.HasField('temporal_coherence'):
            params['temporal_coherence'] = request.temporal_coherence
        if request.HasField('temporal_coherence_latent'):
            params['temporal_coherence_latent'] = request.temporal_coherence_latent

        if params:
            self._queue_params(params)
            return pb2.ControlResponse(
                success=True,
                message=f"Generation params queued: {list(params.keys())}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters to update"
        )

    def SetControlNetParams(self, request, context):
        """Handle SetControlNetParams RPC."""
        self._log("SetControlNetParams called")

        params = {}

        if request.HasField('controlnet_scale'):
            params['controlnet_scale'] = request.controlnet_scale
        if request.HasField('controlnet_start'):
            params['controlnet_start'] = request.controlnet_start
        if request.HasField('controlnet_end'):
            params['controlnet_end'] = request.controlnet_end

        if params:
            self._queue_params(params)
            return pb2.ControlResponse(
                success=True,
                message=f"ControlNet params queued: {list(params.keys())}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters to update"
        )

    def SetLoRAParams(self, request, context):
        """Handle SetLoRAParams RPC."""
        self._log("SetLoRAParams called")

        params = {}

        if request.HasField('lora_scale'):
            params['lora_scale'] = request.lora_scale
        if request.HasField('pipe_index'):
            params['pipe_index'] = request.pipe_index

        if params:
            self._queue_params(params)
            return pb2.ControlResponse(
                success=True,
                message=f"LoRA params queued: {list(params.keys())}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters to update"
        )

    def SetPromptTravelParams(self, request, context):
        """Handle SetPromptTravelParams RPC."""
        self._log("SetPromptTravelParams called")

        params = {}

        if request.HasField('enabled'):
            params['use_prompt_travel_scheduler'] = request.enabled
            # Set a direct flag for the headless loop (thread-safe boolean)
            self._headless_generate = request.enabled
            if request.enabled:
                import time as _time
                self._headless_start_time = _time.time()
                self._headless_frame_count = 0
                self._headless_done = False
                # Compute target frames from journey's duration and fps
                j = self._journey
                fps = j.get('output_fps', 20)
                dur = self._headless_duration
                self._headless_target_frames = int(dur * fps) if dur > 0 else 0
        if request.HasField('min_factor'):
            params['prompt_travel_min_factor'] = request.min_factor
        if request.HasField('max_factor'):
            params['prompt_travel_max_factor'] = request.max_factor
        if request.HasField('factor_increment'):
            params['prompt_travel_factor_increment'] = request.factor_increment
        if request.HasField('stabilize_duration'):
            params['prompt_travel_stabilize_duration'] = request.stabilize_duration
        if request.HasField('oscillate'):
            params['prompt_travel_oscillate'] = request.oscillate
        if request.HasField('use_prompt_scheduler'):
            params['use_prompt_scheduler'] = request.use_prompt_scheduler
        if request.HasField('loop_prompts'):
            params['loop_prompts'] = request.loop_prompts

        if params:
            self._queue_params(params)
            return pb2.ControlResponse(
                success=True,
                message=f"Prompt travel params queued: {list(params.keys())}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters to update"
        )

    def SetAcidParams(self, request, context):
        """Handle SetAcidParams RPC."""
        self._log("SetAcidParams called")

        acid_settings = {}

        if request.HasField('acid_strength'):
            acid_settings['acid_strength'] = request.acid_strength
        if request.HasField('zoom_factor'):
            acid_settings['zoom_factor'] = request.zoom_factor
        if request.HasField('x_shift'):
            acid_settings['x_shift'] = request.x_shift
        if request.HasField('y_shift'):
            acid_settings['y_shift'] = request.y_shift
        if request.HasField('coef_noise'):
            acid_settings['coef_noise'] = request.coef_noise
        if request.HasField('do_acid_tracers'):
            acid_settings['do_acid_tracers'] = request.do_acid_tracers
        if request.HasField('acid_strength_foreground'):
            acid_settings['acid_strength_foreground'] = request.acid_strength_foreground
        if request.HasField('do_acid_wobblers'):
            acid_settings['do_acid_wobblers'] = request.do_acid_wobblers
        if request.HasField('color_matching'):
            acid_settings['color_matching'] = request.color_matching
        if request.HasField('do_human_seg'):
            acid_settings['do_human_seg'] = request.do_human_seg
        if request.HasField('do_blur'):
            acid_settings['do_blur'] = request.do_blur
        if request.HasField('brightness'):
            acid_settings['brightness'] = request.brightness

        if acid_settings:
            # Queue acid settings as a nested dict to match existing pattern
            self._queue_params({'acid_settings': acid_settings})
            return pb2.ControlResponse(
                success=True,
                message=f"Acid params queued: {list(acid_settings.keys())}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters to update"
        )

    def SwitchCuration(self, request, context):
        """Handle SwitchCuration RPC."""
        curation_index = request.curation_index
        self._log(f"SwitchCuration called: index={curation_index}")

        # Queue the curation index change
        self._queue_params({'curation_index': curation_index})

        # Also trigger the callback if provided (for immediate pipeline reload)
        if self._on_curation_switch:
            try:
                self._on_curation_switch(curation_index)
                return pb2.ControlResponse(
                    success=True,
                    message=f"Curation switched to index {curation_index}"
                )
            except Exception as e:
                self._log(f"Error switching curation: {e}")
                return pb2.ControlResponse(
                    success=False,
                    message=f"Error switching curation: {str(e)}"
                )

        return pb2.ControlResponse(
            success=True,
            message=f"Curation index {curation_index} queued (no immediate callback)"
        )

    def GetCurrentState(self, request, context):
        """Handle GetCurrentState RPC."""
        self._log("GetCurrentState called")

        with self._lock:
            state = self._current_state.copy()
            transition = self.get_transition_state()

        return pb2.CurrentStateResponse(
            success=True,
            prompt=state.get('prompt', ''),
            target_prompt=state.get('target_prompt', ''),
            prompt_travel_factor=state.get('prompt_travel_factor', 0.0),
            seed=state.get('seed', -1),
            num_inference_steps=state.get('num_inference_steps', 4),
            guidance_scale=state.get('guidance_scale', 1.0),
            strength=state.get('strength', 0.5),
            width=state.get('width', 512),
            height=state.get('height', 512),
            controlnet_scale=state.get('controlnet_scale', 1.0),
            controlnet_start=state.get('controlnet_start', 0.0),
            controlnet_end=state.get('controlnet_end', 1.0),
            lora_scale=state.get('lora_scale', 1.0),
            pipe_index=state.get('pipe_index', 0),
            prompt_travel_enabled=state.get('prompt_travel_enabled', False),
            prompt_travel_min_factor=state.get('prompt_travel_min_factor', 0.0),
            prompt_travel_max_factor=state.get('prompt_travel_max_factor', 1.0),
            prompt_travel_factor_increment=state.get('prompt_travel_factor_increment', 0.025),
            acid_strength=state.get('acid_strength', 0.4),
            acid_zoom_factor=state.get('acid_zoom_factor', 1.0),
            acid_x_shift=state.get('acid_x_shift', 0),
            acid_y_shift=state.get('acid_y_shift', 0),
            curation_index=state.get('curation_index', 0),
            temporal_coherence=state.get('temporal_coherence', 0.03),
            temporal_coherence_latent=state.get('temporal_coherence_latent', 0.03),
            prompt_transition_frames=transition['default_frames'],
            prompt_transition_progress=transition['progress'],
            prompt_transition_active=transition['active'],
        )

    def BatchUpdate(self, request, context):
        """Handle BatchUpdate RPC for updating multiple parameter groups at once."""
        self._log("BatchUpdate called")

        updated_groups = []

        # Process each parameter group if present
        if request.HasField('prompt_params'):
            self.SetPrompt(request.prompt_params, context)
            updated_groups.append('prompt')

        if request.HasField('generation_params'):
            self.SetGenerationParams(request.generation_params, context)
            updated_groups.append('generation')

        if request.HasField('controlnet_params'):
            self.SetControlNetParams(request.controlnet_params, context)
            updated_groups.append('controlnet')

        if request.HasField('lora_params'):
            self.SetLoRAParams(request.lora_params, context)
            updated_groups.append('lora')

        if request.HasField('prompt_travel_params'):
            self.SetPromptTravelParams(request.prompt_travel_params, context)
            updated_groups.append('prompt_travel')

        if request.HasField('acid_params'):
            self.SetAcidParams(request.acid_params, context)
            updated_groups.append('acid')

        if updated_groups:
            return pb2.ControlResponse(
                success=True,
                message=f"Batch update queued: {', '.join(updated_groups)}"
            )

        return pb2.ControlResponse(
            success=True,
            message="No parameters in batch update"
        )

    # --- Prompt Journey RPCs ---

    def StartPromptJourney(self, request, context):
        """Start an autonomous prompt journey."""
        self._log("StartPromptJourney called")
        # Reset headless state for fresh run
        self._headless_done = False
        self._headless_frame_count = 0

        prompts = list(request.prompts)
        if len(prompts) < 2:
            return pb2.PromptJourneyResponse(
                success=False,
                message="Need at least 2 prompts",
                total_frames=0,
            )

        transition_frames = request.transition_frames or 120
        hold_frames = request.hold_frames or 30
        loop = request.loop
        input_video = request.input_video or ""
        audio_file = request.audio_file or ""
        duration = request.duration or 0
        output_path = request.output_path or ""
        output_fps = request.fps or 20

        # Store duration for headless timer
        self._headless_duration = duration

        # If looping, add first prompt at the end
        if loop:
            prompts.append(prompts[0])

        num_segments = len(prompts) - 1
        total_frames = (num_segments * (hold_frames + transition_frames)) + hold_frames

        with self._lock:
            self._journey = {
                'active': True,
                'completed': False,
                'prompts': prompts,
                'transition_frames': transition_frames,
                'hold_frames': hold_frames,
                'loop': loop,
                'current_frame': 0,
                'total_frames': total_frames,
                'current_segment': 0,
                'total_segments': num_segments,
                'current_prompt': prompts[0],
                'target_prompt': prompts[1] if num_segments > 0 else prompts[0],
                'input_video': input_video,
                'audio_file': audio_file,
                'output_path': output_path,
                'output_fps': output_fps,
            }

        print(f"[gRPC Service] Journey started: {len(prompts)} prompts, "
              f"{num_segments} segments, {total_frames} total frames")

        return pb2.PromptJourneyResponse(
            success=True,
            message=f"Journey started: {num_segments} transitions, {total_frames} frames",
            total_frames=total_frames,
        )

    def GetJourneyStatus(self, request, context):
        """Get status of the running prompt journey."""
        with self._lock:
            j = self._journey
            total = j['total_frames'] if j['total_frames'] > 0 else 1
            # Use headless state only if scheduler was explicitly activated
            using_headless = self._headless_generate or self._headless_done
            cur_frame = self._headless_frame_count if using_headless else j['current_frame']
            done = self._headless_done if using_headless else j['completed']
            total = self._headless_target_frames if using_headless else j['total_frames']
            return pb2.PromptJourneyStatus(
                active=self._headless_generate or j['active'],
                current_frame=cur_frame,
                total_frames=total,
                progress=j['current_frame'] / total if total > 0 else 0.0,
                current_segment=j['current_segment'],
                total_segments=j['total_segments'],
                current_prompt=j['current_prompt'],
                target_prompt=j['target_prompt'],
                completed=done,
                output_file=j.get('output_path', ''),
            )

    def StopPromptJourney(self, request, context):
        """Stop a running prompt journey."""
        with self._lock:
            was_active = self._journey['active']
            self._journey['active'] = False
            self._journey['completed'] = True

        return pb2.ControlResponse(
            success=True,
            message="Journey stopped" if was_active else "No journey was running",
        )

    def get_journey_state(self) -> dict:
        """Get the current journey state (called by the headless loop)."""
        with self._lock:
            return self._journey.copy()

    def advance_journey_frame(self):
        """
        Advance the journey by one frame.
        Returns (prompt, target_prompt, factor, active) for the current frame,
        or None if no journey is active.
        """
        with self._lock:
            j = self._journey
            if not j['active'] or j['completed']:
                return None

            frame = j['current_frame']
            hold = j['hold_frames']
            trans = j['transition_frames']
            segment_length = hold + trans
            prompts = j['prompts']
            num_segments = j['total_segments']

            # Which segment are we in?
            if frame < num_segments * segment_length:
                segment = frame // segment_length
                frame_in_segment = frame % segment_length

                src = prompts[segment]
                dst = prompts[segment + 1]

                if frame_in_segment < hold:
                    # Hold phase
                    factor = 0.0
                else:
                    # Transition phase
                    t = frame_in_segment - hold
                    factor = t / max(trans - 1, 1)
            else:
                # Final hold phase
                segment = num_segments - 1
                src = prompts[-1]
                dst = prompts[-1]
                factor = 0.0

            j['current_segment'] = segment
            j['current_prompt'] = src
            j['target_prompt'] = dst
            j['current_frame'] = frame + 1

            # Check if journey is complete
            if frame + 1 >= j['total_frames']:
                j['active'] = False
                j['completed'] = True
                print(f"[gRPC Service] Journey completed: {frame + 1} frames")

            return (src, dst, factor, True)
