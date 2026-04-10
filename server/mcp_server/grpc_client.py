"""Async gRPC client wrapper for the Glitchbox GenerationControl service."""

import os

import grpc
import grpc.aio

from grpc_server import generation_control_pb2 as pb2
from grpc_server import generation_control_pb2_grpc as pb2_grpc


def _set_optional_fields(msg, **kwargs):
    """Set fields on a protobuf message, skipping None values.

    Raises AttributeError if a field name does not exist on the message.
    """
    descriptor_fields = {f.name for f in msg.DESCRIPTOR.fields}
    for field, value in kwargs.items():
        if value is None:
            continue
        if field not in descriptor_fields:
            raise AttributeError(
                f"{msg.DESCRIPTOR.name} has no field '{field}'"
            )
        setattr(msg, field, value)


def _response_to_dict(response) -> dict:
    """Convert a ControlResponse to a dict."""
    return {"success": response.success, "message": response.message}


def _state_to_dict(state) -> dict:
    """Convert a CurrentStateResponse to a dict."""
    return {
        "success": state.success,
        "prompt": state.prompt,
        "target_prompt": state.target_prompt,
        "prompt_travel_factor": state.prompt_travel_factor,
        "seed": state.seed,
        "num_inference_steps": state.num_inference_steps,
        "guidance_scale": state.guidance_scale,
        "strength": state.strength,
        "width": state.width,
        "height": state.height,
        "controlnet_scale": state.controlnet_scale,
        "controlnet_start": state.controlnet_start,
        "controlnet_end": state.controlnet_end,
        "lora_scale": state.lora_scale,
        "pipe_index": state.pipe_index,
        "prompt_travel_enabled": state.prompt_travel_enabled,
        "prompt_travel_min_factor": state.prompt_travel_min_factor,
        "prompt_travel_max_factor": state.prompt_travel_max_factor,
        "prompt_travel_factor_increment": state.prompt_travel_factor_increment,
        "acid_strength": state.acid_strength,
        "acid_zoom_factor": state.acid_zoom_factor,
        "acid_x_shift": state.acid_x_shift,
        "acid_y_shift": state.acid_y_shift,
        "curation_index": state.curation_index,
        "temporal_coherence": state.temporal_coherence,
        "temporal_coherence_latent": state.temporal_coherence_latent,
        "prompt_transition_frames": state.prompt_transition_frames,
        "prompt_transition_progress": state.prompt_transition_progress,
        "prompt_transition_active": state.prompt_transition_active,
    }


def _grpc_error_dict(error: grpc.aio.AioRpcError) -> dict:
    """Convert a gRPC error into a standard error response dict."""
    return {
        "success": False,
        "message": f"gRPC error ({error.code().name}): {error.details()}",
    }


class GlitchboxGRPCClient:
    """Async gRPC client for the Glitchbox GenerationControl service."""

    def __init__(self):
        host = os.environ.get("GLITCHBOX_GRPC_HOST", "localhost")
        port = os.environ.get("GLITCHBOX_GRPC_PORT", "50053")
        self._target = f"{host}:{port}"
        self._channel = None
        self._stub = None

    async def connect(self):
        self._channel = grpc.aio.insecure_channel(self._target)
        self._stub = pb2_grpc.GenerationControlStub(self._channel)

    async def close(self):
        if self._channel:
            await self._channel.close()

    async def set_prompt(
        self,
        prompt: str,
        target_prompt: str | None = None,
        prompt_travel_factor: float | None = None,
        transition_frames: int | None = None,
    ) -> dict:
        try:
            req = pb2.PromptRequest(prompt=prompt)
            _set_optional_fields(
                req,
                target_prompt=target_prompt,
                prompt_travel_factor=prompt_travel_factor,
                transition_frames=transition_frames,
            )
            return _response_to_dict(await self._stub.SetPrompt(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def set_generation_params(self, **kwargs) -> dict:
        try:
            req = pb2.GenerationParamsRequest()
            _set_optional_fields(req, **kwargs)
            return _response_to_dict(await self._stub.SetGenerationParams(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def set_controlnet_params(self, **kwargs) -> dict:
        try:
            req = pb2.ControlNetParamsRequest()
            _set_optional_fields(req, **kwargs)
            return _response_to_dict(await self._stub.SetControlNetParams(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def set_lora_params(self, **kwargs) -> dict:
        try:
            req = pb2.LoRAParamsRequest()
            _set_optional_fields(req, **kwargs)
            return _response_to_dict(await self._stub.SetLoRAParams(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def set_prompt_travel_params(self, **kwargs) -> dict:
        try:
            req = pb2.PromptTravelParamsRequest()
            _set_optional_fields(req, **kwargs)
            return _response_to_dict(await self._stub.SetPromptTravelParams(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def set_acid_params(self, **kwargs) -> dict:
        try:
            req = pb2.AcidParamsRequest()
            _set_optional_fields(req, **kwargs)
            return _response_to_dict(await self._stub.SetAcidParams(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def switch_curation(self, curation_index: int) -> dict:
        try:
            req = pb2.SwitchCurationRequest(curation_index=curation_index)
            return _response_to_dict(await self._stub.SwitchCuration(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def get_current_state(self) -> dict:
        try:
            req = pb2.GetStateRequest()
            return _state_to_dict(await self._stub.GetCurrentState(req))
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def batch_update(
        self,
        prompt_params: dict | None = None,
        generation_params: dict | None = None,
        controlnet_params: dict | None = None,
        lora_params: dict | None = None,
        prompt_travel_params: dict | None = None,
        acid_params: dict | None = None,
    ) -> dict:
        try:
            return await self._batch_update_inner(
                prompt_params=prompt_params,
                generation_params=generation_params,
                controlnet_params=controlnet_params,
                lora_params=lora_params,
                prompt_travel_params=prompt_travel_params,
                acid_params=acid_params,
            )
        except grpc.aio.AioRpcError as e:
            return _grpc_error_dict(e)

    async def _batch_update_inner(self, **param_groups) -> dict:
        """Build the BatchUpdateRequest from provided parameter group dicts."""
        message_types = {
            "prompt_params": pb2.PromptRequest,
            "generation_params": pb2.GenerationParamsRequest,
            "controlnet_params": pb2.ControlNetParamsRequest,
            "lora_params": pb2.LoRAParamsRequest,
            "prompt_travel_params": pb2.PromptTravelParamsRequest,
            "acid_params": pb2.AcidParamsRequest,
        }
        kwargs = {}
        for group_name, fields in param_groups.items():
            if fields is None:
                continue
            msg = message_types[group_name]()
            _set_optional_fields(msg, **fields)
            kwargs[group_name] = msg
        req = pb2.BatchUpdateRequest(**kwargs)
        return _response_to_dict(await self._stub.BatchUpdate(req))
