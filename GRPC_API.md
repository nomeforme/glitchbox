# gRPC API Documentation

## Connection Info

| Property | Value |
|----------|-------|
| **Port** | `50051` |
| **Package** | `generation_control` |
| **Service** | `GenerationControl` |

**Connection string:** `<IP>:50051`

---

## RPC Methods

### 1. SetPrompt
Update the generation prompt.

```protobuf
rpc SetPrompt(PromptRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Main generation prompt |
| `target_prompt` | string | No | Target prompt for prompt travel interpolation |
| `prompt_travel_factor` | float | No | Interpolation factor (0.0-1.0) |
| `transition_frames` | int32 | No | Smooth transition frames (0-100). When set, smoothly blends from current prompt to new prompt over N frames. |

**Smooth Transitions:** When `transition_frames` is provided, the prompt change will be gradual:
- `0` = instant change (default)
- `30` = ~1 second transition at 30fps
- `100` = maximum smooth transition

---

### 2. SetGenerationParams
Update core generation parameters.

```protobuf
rpc SetGenerationParams(GenerationParamsRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `seed` | int32 | No | Random seed (-1 for random) |
| `num_inference_steps` | int32 | No | Number of denoising steps |
| `guidance_scale` | float | No | CFG scale |
| `strength` | float | No | Denoising strength (0.0-1.0) |
| `width` | int32 | No | Output width |
| `height` | int32 | No | Output height |
| `temporal_coherence` | float | No | Noise blending (0.0-1.0) |
| `temporal_coherence_latent` | float | No | Latent feedback blending (0.0-1.0) |

---

### 3. SetControlNetParams
Update ControlNet parameters.

```protobuf
rpc SetControlNetParams(ControlNetParamsRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `controlnet_scale` | float | No | ControlNet conditioning scale |
| `controlnet_start` | float | No | Start point (0.0-1.0) |
| `controlnet_end` | float | No | End point (0.0-1.0) |

---

### 4. SetLoRAParams
Update LoRA parameters.

```protobuf
rpc SetLoRAParams(LoRAParamsRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lora_scale` | float | No | LoRA weight scale |
| `pipe_index` | int32 | No | Which pipe/LoRA combination to use |

---

### 5. SetPromptTravelParams
Update prompt travel scheduler parameters.

```protobuf
rpc SetPromptTravelParams(PromptTravelParamsRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `enabled` | bool | No | Enable prompt travel scheduler |
| `min_factor` | float | No | Minimum interpolation factor |
| `max_factor` | float | No | Maximum interpolation factor |
| `factor_increment` | float | No | Step size per frame |
| `stabilize_duration` | int32 | No | Frames to hold at endpoints |
| `oscillate` | bool | No | Ping-pong between prompts |
| `use_prompt_scheduler` | bool | No | Use scheduled prompt switching |
| `loop_prompts` | bool | No | Loop through prompt list |

---

### 6. SetAcidParams
Update acid processor (visual effects) parameters.

```protobuf
rpc SetAcidParams(AcidParamsRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `acid_strength` | float | No | Overall acid effect strength |
| `zoom_factor` | float | No | Zoom amount (1.0 = no zoom) |
| `x_shift` | int32 | No | Horizontal shift in pixels |
| `y_shift` | int32 | No | Vertical shift in pixels |
| `coef_noise` | float | No | Noise coefficient |
| `do_acid_tracers` | bool | No | Enable tracer effects |
| `acid_strength_foreground` | float | No | Foreground effect strength |
| `do_acid_wobblers` | bool | No | Enable wobble effects |
| `color_matching` | float | No | Color matching strength |
| `do_human_seg` | bool | No | Enable human segmentation |
| `do_blur` | bool | No | Enable blur effect |
| `brightness` | float | No | Brightness adjustment |

---

### 7. SwitchCuration
Switch to a different curation/preset index.

```protobuf
rpc SwitchCuration(SwitchCurationRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `curation_index` | int32 | Yes | Curation preset index to switch to |

---

### 8. GetCurrentState
Get the current state of all parameters.

```protobuf
rpc GetCurrentState(GetStateRequest) returns (CurrentStateResponse)
```

**Request:** Empty (no parameters)

**Response fields:**
- `success` (bool)
- `prompt`, `target_prompt`, `prompt_travel_factor`
- `seed`, `num_inference_steps`, `guidance_scale`, `strength`, `width`, `height`
- `controlnet_scale`, `controlnet_start`, `controlnet_end`
- `lora_scale`, `pipe_index`
- `prompt_travel_enabled`, `prompt_travel_min_factor`, `prompt_travel_max_factor`, `prompt_travel_factor_increment`
- `acid_strength`, `acid_zoom_factor`, `acid_x_shift`, `acid_y_shift`
- `curation_index`
- `temporal_coherence`, `temporal_coherence_latent`
- `prompt_transition_frames` (int32) - Default transition frames setting
- `prompt_transition_progress` (float) - Current transition progress (0.0-1.0)
- `prompt_transition_active` (bool) - Whether a transition is in progress

---

### 9. BatchUpdate
Update multiple parameter groups in a single call.

```protobuf
rpc BatchUpdate(BatchUpdateRequest) returns (ControlResponse)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt_params` | PromptRequest | No | Prompt parameters |
| `generation_params` | GenerationParamsRequest | No | Generation parameters |
| `controlnet_params` | ControlNetParamsRequest | No | ControlNet parameters |
| `lora_params` | LoRAParamsRequest | No | LoRA parameters |
| `prompt_travel_params` | PromptTravelParamsRequest | No | Prompt travel parameters |
| `acid_params` | AcidParamsRequest | No | Acid processor parameters |

---

## Response Format

All methods (except `GetCurrentState`) return:

```protobuf
message ControlResponse {
    bool success = 1;
    string message = 2;
}
```

---

## Example Usage (Python)

```python
import grpc
from generation_control_pb2 import PromptRequest, GenerationParamsRequest, GetStateRequest
from generation_control_pb2_grpc import GenerationControlStub

# Connect
channel = grpc.insecure_channel('<IP>:50051')
stub = GenerationControlStub(channel)

# Set prompt (instant change)
response = stub.SetPrompt(PromptRequest(
    prompt="a beautiful sunset over mountains"
))

# Set prompt with smooth transition (30 frames)
response = stub.SetPrompt(PromptRequest(
    prompt="a starry night sky",
    transition_frames=30
))

# Manual interpolation between two prompts
response = stub.SetPrompt(PromptRequest(
    prompt="a beautiful sunset over mountains",
    target_prompt="a starry night sky",
    prompt_travel_factor=0.5
))

# Set generation params
response = stub.SetGenerationParams(GenerationParamsRequest(
    seed=42,
    temporal_coherence=0.1,
    temporal_coherence_latent=0.05
))

# Get current state
state = stub.GetCurrentState(GetStateRequest())
print(f"Current prompt: {state.prompt}")
print(f"Transition active: {state.prompt_transition_active}")
print(f"Transition progress: {state.prompt_transition_progress}")
```

## Command Line Usage

```bash
# Instant prompt change
python grpc_remote_client.py "a beautiful sunset"

# Smooth transition (30 frames)
python grpc_remote_client.py "a starry night" --transition 30

# Check status
python grpc_remote_client.py --status
```

---

## Proto File Location

```
server/grpc_server/protos/generation_control.proto
```

To generate client stubs:
```bash
python -m grpc_tools.protoc -I server/grpc_server/protos \
    --python_out=. --grpc_python_out=. \
    server/grpc_server/protos/generation_control.proto
```
