from pydantic import BaseModel, Field
from PIL import Image
import os
import glob
from typing import List, Dict, Optional, Union
import time
import uuid

# Mock version - minimal imports needed
from config import Args
from .config import LoRACurationConfig

# Function to read prompt prefix from .txt files (same as original)
def get_prompt_prefix():
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    prompt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))
    
    if not prompt_files:
        # Default prompt prefix if no files are found
        return ""
    
    # Read the first prompt file
    with open(prompt_files[0], 'r') as f:
        return f.read().strip()

prompt_prefix = get_prompt_prefix()
default_prompt = prompt_prefix + "two figures in a dynamic, abstract dance, forming a surreal, interconnected sculpture against a black background."
default_target_prompt = prompt_prefix + "a blue dog"

page_content = """
<h1 class="text-3xl font-bold">Mock Real-Time SDv2.1 Turbo</h1>
<h3 class="text-xl font-bold">Mock Image-to-Image ControlNet</h3>
<p class="text-sm">
    This is a mock demo that returns the input control image for testing without GPU.
    All parameters are preserved but no actual diffusion is performed.
</p>
<p class="text-sm text-gray-500">
    Mock pipeline for development and testing purposes.
</p>
"""

class Pipeline:
    class Info(BaseModel):
        name: str = "controlnet+sd21Turbo+mock"
        title: str = "Mock SDv2.1 Turbo + Controlnet"
        description: str = "Mock pipeline that returns control image for testing without GPU"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        target_prompt: str = Field(
            default_target_prompt,
            title="Target Prompt",
            field="textarea",
            id="target_prompt",
            hide=True,
        )
        client_prompt_prefix: str = Field(
            prompt_prefix,
            title="Client Prompt Prefix",
            field="textarea",
            id="client_prompt_prefix",
            description="Prefix to prepend to client-provided prompts from STT",
        )
        pipe_index: int = Field(
            0,
            min=0,
            max=4,
            step=1,
            title="Pipe Index",
            field="range",
            id="pipe_index",
            description="Select which weight combination to use for the selected LoRA pair"
        )
        use_prompt_travel: bool = Field(
            True,
            title="Use Prompt Travel",
            field="checkbox",
            id="use_prompt_travel",
        )
        prompt_travel_factor: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Prompt Travel Factor",
            field="range",
            id="prompt_travel_factor",
            hide=True,
        )
        use_latent_travel: bool = Field(
            True,
            title="Use Latent Travel",
            field="checkbox",
            id="use_latent_travel",
            hide=True,
        )
        latent_travel_method: str = Field(
            "slerp",
            title="Latent Travel Method",
            field="select",
            id="latent_travel_method",
            options=["slerp", "linear"],
            hide=True,
        )
        latent_travel_factor: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            title="Latent Travel Factor",
            field="range",
            id="latent_travel_factor",
            hide=True,
        )
        lora_scale: float = Field(
            1.0,
            min=0.0,
            max=2.0,
            step=0.01,
            title="LoRA Scale",
            field="range",
            id="lora_scale",
        )
        seed: int = Field(
            4402026899276587, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        target_seed: Optional[int] = Field(
            None, min=0, title="Target Seed", field="seed", hide=True, id="target_seed"
        )
        steps: int = Field(
            4, min=1, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            640, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            360, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            1.00,
            min=0,
            max=10,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.8,
            min=0.10,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )
        controlnet_scale: float = Field(
            0.75,
            min=0,
            max=2.0,
            step=0.001,
            title="Controlnet Scale",
            field="range",
            hide=True,
            id="controlnet_scale",
        )
        # Add boost factor parameters
        boost_factor_bass: float = Field(
            1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Bass",
            field="range",
            id="boost_factor_bass",
        )
        boost_factor_low_mids: float = Field(
            1.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Low Mids",
            field="range",
            id="boost_factor_low_mids",
        )
        boost_factor_mids: float = Field(
            1.5,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Mids",
            field="range",
            id="boost_factor_mids",
        )
        boost_factor_high_mids: float = Field(
            2.0,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - High Mids",
            field="range",
            id="boost_factor_high_mids",
        )
        boost_factor_treble: float = Field(
            2.5,
            min=0.0,
            max=3.0,
            step=0.1,
            title="Boost - Treble",
            field="range",
            id="boost_factor_treble",
        )
        controlnet_start: float = Field(
            0.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet Start",
            field="range",
            hide=True,
            id="controlnet_start",
        )
        controlnet_end: float = Field(
            1.0,
            min=0,
            max=1.0,
            step=0.001,
            title="Controlnet End",
            field="range",
            hide=True,
            id="controlnet_end",
        )
        debug_controlnet: bool = Field(
            False,
            title="Debug ControlNet",
            field="checkbox",
            hide=True,
            id="debug_controlnet",
        )
        use_output_bg_removal: bool = Field(
            False,
            title="Use Output Background Removal",
            field="checkbox",
            id="use_output_bg_removal",
        )
        use_prompt_indexing: bool = Field(
            False,
            title="Use Prompt Indexing",
            field="checkbox",
            id="use_prompt_indexing",
            description="Use pipe index to select prompts from file instead of sequential scheduling",
        )
        use_client_prompts: bool = Field(
            True,
            title="Use Client Prompts",
            field="checkbox",
            id="use_client_prompts",
            description="Use client-provided prompts instead of scheduled prompts for prompt travel",
        )

    def __init__(self, args: Args, device=None, torch_dtype=None, lora_config=None):
        """Mock initialization - no GPU models loaded"""
        print(f"[controlnetSDTurbot2i_mock.py] Initializing mock pipeline (no GPU required)")
        
        # Store args for reference
        self.args = args
        
        # Mock the same structure as original
        self.current_curation_index = None
        self.pipes = []
        self.pipe_states = []
        
        # Mock upscaler and pixelate processor flags
        self.use_upscaler = getattr(args, 'use_upscaler', False)
        self.use_pixelate_processor = getattr(args, 'use_pixelate_processor', False)
        
        # Mock LoRA configuration
        self.curation_index = getattr(args, 'default_curation_index', 0)
        
        if lora_config is not None:
            self.lora_config = lora_config
            print(f"[controlnetSDTurbot2i_mock.py] Using provided LoRACurationConfig from app level")
        else:
            # Create mock config or use None if config dir doesn't exist
            try:
                self.lora_config = LoRACurationConfig(args.lora_config_dir, default_curation_index=self.curation_index)
                print(f"[controlnetSDTurbot2i_mock.py] Created new LoRACurationConfig")
            except:
                self.lora_config = None
                print(f"[controlnetSDTurbot2i_mock.py] No LoRA config available (mock mode)")

        # Mock LoRA data structures
        if self.lora_config:
            self.lora_models = self.lora_config.get_lora_models()
            self.default_lora_scale = self.lora_config.DEFAULT_LORA_SCALE
            self.lora_curation = self.lora_config.get_lora_curation()
            self.curation_keys = self.lora_config.get_curation_keys()
            self.adapter_weights_set_curation = self.lora_config.get_adapter_weights_set_curation()
            self.adapter_weights_sets = self.lora_config.get_default_adapter_weights()
        else:
            # Mock defaults
            self.lora_models = {}
            self.default_lora_scale = 1.0
            self.lora_curation = {}
            self.curation_keys = []
            self.adapter_weights_set_curation = {}
            self.adapter_weights_sets = [[1.0]] * 5  # 5 default weight sets

        # Create mock pipes (just empty dictionaries)
        for i in range(len(self.adapter_weights_sets)):
            mock_pipe = {
                'mock': True,
                'index': i,
                'name': f'mock_pipe_{i}'
            }
            
            mock_pipe_state = {
                'current_lora_models': [],
                'fuse_loras': False,
                'lora_scale': 1.0,
                'lora_weights_loaded': False,
                'current_adapter_weights': [],
                'stored_result_latents': None
            }
            
            self.pipes.append(mock_pipe)
            self.pipe_states.append(mock_pipe_state)

        # Store the current pipe index
        self.current_pipe_idx = 0
        
        print(f"[controlnetSDTurbot2i_mock.py] Mock pipeline initialized with {len(self.pipes)} mock pipes")

    def load_loras_for_pipe(self, pipe, pipe_state, lora_models_list: List[str], fuse_loras: bool = False, lora_scale: float = 1.0, adapter_weights: Optional[List[float]] = None) -> None:
        """Mock LoRA loading - just prints what would be loaded"""
        print(f"[MOCK] Would load LoRAs: {lora_models_list} with fuse={fuse_loras}, scale={lora_scale}, weights={adapter_weights}")
        
        # Update mock state
        pipe_state['current_lora_models'] = lora_models_list
        pipe_state['fuse_loras'] = fuse_loras
        pipe_state['lora_scale'] = lora_scale
        pipe_state['current_adapter_weights'] = adapter_weights or [1.0] * len(lora_models_list)
        pipe_state['lora_weights_loaded'] = True

    def update_lora_set(self, pipe, pipe_state, curation_index: int) -> None:
        """Mock LoRA set update"""
        if self.current_curation_index != curation_index:
            print(f"[MOCK] Would update LoRAs for curation index {curation_index}")
            self.current_curation_index = curation_index

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        """Mock prediction - returns the control_image instead of generating"""
        print(f"[controlnetSDTurbot2i_mock.py] Mock predict called with pipe_index={params.pipe_index}")
        
        # Use the pipe index from params
        self.current_pipe_idx = int(params.pipe_index)
        pipe = self.pipes[self.current_pipe_idx]
        pipe_state = self.pipe_states[self.current_pipe_idx]
        
        print(f"[MOCK] Using pipe {self.current_pipe_idx}: {pipe}")
        
        # Mock all the parameter processing
        print(f"[MOCK] Prompt: {params.prompt}")
        print(f"[MOCK] Target prompt: {params.target_prompt}")
        print(f"[MOCK] Seed: {params.seed}")
        print(f"[MOCK] Steps: {params.steps}")
        print(f"[MOCK] Strength: {params.strength}")
        print(f"[MOCK] Guidance scale: {params.guidance_scale}")
        print(f"[MOCK] ControlNet scale: {params.controlnet_scale}")
        print(f"[MOCK] Use prompt travel: {params.use_prompt_travel}")
        print(f"[MOCK] Use latent travel: {params.use_latent_travel}")
        
        # Get the control image
        control_image = getattr(params, 'control_image', None)
        
        if control_image is None:
            print("[MOCK] No control_image provided, creating a placeholder")
            # Create a placeholder image if no control image is provided
            placeholder = Image.new('RGB', (params.width, params.height), color='gray')
            return placeholder
        
        print(f"[MOCK] Returning control_image with size: {control_image.size}")
        
        # Optional: resize to match expected output dimensions
        if control_image.size != (params.width, params.height):
            print(f"[MOCK] Resizing control_image from {control_image.size} to ({params.width}, {params.height})")
            control_image = control_image.resize((params.width, params.height))

        time.sleep(0.01)
        
        return control_image 