
# Glitchbox
WIP. This is pre-publication. An official release will be made including full documentation and comprehensive installation instructions. Currently undocumented.
## Hardware Requirements
A Linux computer with an NVIDIA RTX 4090 GPU.
Support for Windows and the NVIDIA RTX 5090 GPU are underway.

## MCP Integration

Glitchbox exposes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that lets AI agents control the generation pipeline. It wraps the gRPC `GenerationControl` service (see [GRPC_API.md](GRPC_API.md)) and exposes all RPCs as MCP tools.

### Available Tools

| Tool | Description |
|------|-------------|
| `set_prompt` | Set the generation prompt, with optional smooth transitions and prompt travel |
| `set_generation_params` | Adjust seed, inference steps, guidance scale, strength, resolution, temporal coherence |
| `set_controlnet_params` | Adjust ControlNet conditioning scale, start, and end |
| `set_lora_params` | Adjust LoRA weight scale and active pipe index |
| `set_prompt_travel_params` | Configure prompt interpolation (oscillation, looping, scheduling) |
| `set_acid_params` | Configure acid visual effects (zoom, shift, tracers, wobble, blur, etc.) |
| `switch_curation` | Switch the active LoRA curation preset |
| `get_current_state` | Get the full current state of all generation parameters |
| `batch_update` | Update multiple parameter groups atomically in a single call |

### Resource

| URI | Description |
|-----|-------------|
| `glitchbox://state/current` | Read-only access to the current generation state |

### Transports

**SSE** -- Available at `http://<host>:7860/mcp/sse` when the server is running. No extra setup needed.

**stdio** -- For use with Claude Code and other MCP clients that spawn a subprocess.

### Claude Code Configuration

#### stdio (local -- MCP client spawns the server)

Add to your project's `.mcp.json` (or `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "glitchbox": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/glitchbox/server", "python", "-m", "mcp_server.stdio_main"],
      "env": {
        "GLITCHBOX_GRPC_HOST": "localhost",
        "GLITCHBOX_GRPC_PORT": "50053"
      }
    }
  }
}
```

#### Streamable HTTP (remote -- connects to a running server)

When the glitchbox server is already running, the MCP endpoint is available automatically. Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "glitchbox": {
      "type": "streamable-http",
      "url": "http://<host>:7860/mcp/"
    }
  }
}
```

Replace `<host>` with the IP or hostname of the machine running glitchbox (e.g. `192.168.1.247`).

### Prerequisites

The glitchbox server must be running with gRPC enabled (default). The MCP server connects to the gRPC service on port 50053.

