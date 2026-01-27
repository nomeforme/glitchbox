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
                 debug: bool = False):
        """
        Initialize the servicer.

        Args:
            on_curation_switch: Callback function for curation switches
            debug: Enable debug logging
        """
        self._lock = threading.RLock()
        self._pending_params: Dict[str, Any] = {}
        self._current_state: Dict[str, Any] = {}
        self._on_curation_switch = on_curation_switch
        self._debug = debug
        self._logger = logging.getLogger(__name__)

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
            self._log(f"Queued params: {params}")

    # gRPC Service Methods

    def SetPrompt(self, request, context):
        """Handle SetPrompt RPC."""
        self._log(f"SetPrompt called: prompt='{request.prompt[:50] if request.prompt else ''}...'")

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
