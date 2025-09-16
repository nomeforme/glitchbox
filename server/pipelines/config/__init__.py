import os
import json

class LoRACurationConfig:
    """
    Configuration class for LoRA curation.
    Loads all curation configurations from JSON files in a specified directory.
    Exposes data for a single, default curation (determined by default_curation_index)
    through an API compatible with the previous version for controlnetSDTurbot2i.py.
    """

    def __init__(self, lora_config_dir: str, default_curation_index=0):
        print(f"[LoRACurationConfig] Initializing. Loading all configs from: {lora_config_dir}. Default index: {default_curation_index}")
        
        # This is a general mapping of LoRA names to their potential file paths or HF IDs.
        self.lora_models = {
            "None": None,
            "radames/sd-21-DPO-LoRA": "radames/sd-21-DPO-LoRA",
            "latent-consistency/lcm-lora-sdv2-1": "latent-consistency/lcm-lora-sdv2-1",
            "latent-consistency/lcm-lora-sdv2-1-turbo": "latent-consistency/lcm-lora-sdv2-1-turbo",
            "garance": "loras/garance-000038.safetensors",
            "dark": "loras/flowers-000022.safetensors",
            "abstract-monochrome": "loras/abstract-monochrome-000140.safetensors",
            "abstract-brokenglass-red": "loras/abstract_brokenglass_red-000140.safetensors",
            "full-body-glitch-reddish": "loras/full_body_glitch_reddish.safetensors",
            "melies-bw": "loras/melies_bw-000012.safetensors",
            "melies-col": "loras/melies_col-000013.safetensors",
            "nature-water": "loras/nature_water-000023.safetensors",
            "nature-fire": "loras/nature_fire-000017.safetensors",
            "nature-smoke": "loras/nature_smoke-000016.safetensors",
            "nature-sand": "loras/nature_sand2-000009.safetensors",
            "robwood": "loras/robwood.safetensors",
            "sweet-vicious": "loras/sweet_vicious-000072.safetensors",
            "liquid-love": "loras/liquid_love.safetensors",
            "glitch": "loras/glitch-step00000400.safetensors",
            "pixels-bodies": "loras/pixels_bodies.safetensors",
            "pixels-face": "loras/pixels_faces-000023.safetensors",
            "origami": "loras/origami-000023.safetensors",
            "twisted-bodies": "loras/twistedbodies-000014.safetensors",
            "HAHACards_A2": "loras/HAHACards_A2-000015.safetensors",
            "goldworld": "loras/goldworld-000006.safetensors",
            "monoblue": "loras/jas_monoblue.safetensors",
            "psychaos": "loras/jas_psychaos2-000019.safetensors",
            "angels": "loras/angelsai-step00001500.safetensors",
            "mannequin": "loras/mannequin-step00003100.safetensors",
            "children": "loras/sota2-step00000900.safetensors",
            "building": "loras/sota21-step00000700.safetensors",
        }

        self._all_curations = {} # Stores all loaded JSON data {key: data}
        self._all_curation_keys = [] # Stores all keys from JSON filenames
        self._load_all_curation_configs(lora_config_dir)

        self.DEFAULT_CURATION_INDEX = default_curation_index
        self.default_curation_key = None
        
        if self._all_curation_keys:
            if 0 <= self.DEFAULT_CURATION_INDEX < len(self._all_curation_keys):
                self.default_curation_key = self._all_curation_keys[self.DEFAULT_CURATION_INDEX]
                print(f"[LoRACurationConfig] Default curation key set to: '{self.default_curation_key}' (index {self.DEFAULT_CURATION_INDEX})")
            else:
                print(f"[LoRACurationConfig] Warning: default_curation_index {self.DEFAULT_CURATION_INDEX} is out of bounds for {len(self._all_curation_keys)} loaded curations. Falling back to index 0.")
                self.DEFAULT_CURATION_INDEX = 0 # Adjust index to be valid
                if self._all_curation_keys: # Ensure list is not empty after adjustment
                    self.default_curation_key = self._all_curation_keys[0]
                    print(f"[LoRACurationConfig] Default curation key set to: '{self.default_curation_key}' (fallback index 0)")
                else:
                    print(f"[LoRACurationConfig] Critical: No curations loaded, cannot set a default key.")
        else:
            print("[LoRACurationConfig] Warning: No curation JSON files found. LoRA curation will be empty.")

        self.lora_curation = {} 
        self.curation_keys = []   
        self.adapter_weights_set_curation = {} 
        self.DEFAULT_LORA_SCALE = 1.0 
        self.default_curation_input_params = {} # For other params from the default JSON

        if self.default_curation_key and self.default_curation_key in self._all_curations:
            default_config_data = self._all_curations[self.default_curation_key]
            
            self.lora_curation = {self.default_curation_key: default_config_data.get("loras", [])}
            self.curation_keys = [self.default_curation_key]
            self.adapter_weights_set_curation = {self.default_curation_key: default_config_data.get("adapter_weights_sets", [[]])}
            # Ensure adapter_weights_set_curation[key] is a list of lists, even if empty an outer list is needed by callers.
            if not self.adapter_weights_set_curation[self.default_curation_key] or not isinstance(self.adapter_weights_set_curation[self.default_curation_key][0], list):
                 self.adapter_weights_set_curation[self.default_curation_key] = [[]] # Default to list containing one empty list of weights

            self.default_curation_input_params = default_config_data.get("input_params", {})
            self.DEFAULT_LORA_SCALE = float(self.default_curation_input_params.get("lora_scale", 1.0))
            print(f"[LoRACurationConfig] DEFAULT_LORA_SCALE set to: {self.DEFAULT_LORA_SCALE} from '{self.default_curation_key}.json'")
        else:
            print(f"[LoRACurationConfig] Warning: Default curation key '{self.default_curation_key}' not found or no curations loaded. Curation API will use empty/default values.")
            # Ensure structure compatibility for old API consumers if default key isn't usable
            if self.default_curation_key: # If key was determined but not found in _all_curations (should not happen if logic is correct)
                self.curation_keys = [self.default_curation_key]
                self.lora_curation = {self.default_curation_key: []}
                self.adapter_weights_set_curation = {self.default_curation_key: [[]]}
            else: # No keys loaded at all
                self.curation_keys = []
                self.lora_curation = {}
                self.adapter_weights_set_curation = {}

    def _load_all_curation_configs(self, lora_config_dir: str):
        if not os.path.isdir(lora_config_dir):
            print(f"[LoRACurationConfig] Error: LoRA config directory does not exist: {lora_config_dir}")
            return

        for filename in os.listdir(lora_config_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(lora_config_dir, filename)
                curation_key = filename[:-5] 
                try:
                    with open(filepath, 'r') as f:
                        config_data = json.load(f)
                    
                    if not all(k in config_data for k in ["prompts_file_name", "loras", "adapter_weights_sets", "input_params"]):
                        print(f"[LoRACurationConfig] Warning: Skipping {filename}. Missing one or more required keys (prompts_file_name, loras, adapter_weights_sets, input_params).")
                        continue
                    if not isinstance(config_data.get("adapter_weights_sets"), list) or \
                       (config_data.get("adapter_weights_sets") and not all(isinstance(i, list) for i in config_data.get("adapter_weights_sets"))):
                        print(f"[LoRACurationConfig] Warning: Skipping {filename}. 'adapter_weights_sets' must be a list of lists.")
                        continue

                    self._all_curations[curation_key] = config_data
                    self._all_curation_keys.append(curation_key)
                    print(f"[LoRACurationConfig] Successfully loaded and validated: {curation_key}.json")
                except json.JSONDecodeError:
                    print(f"[LoRACurationConfig] Warning: Error decoding JSON from {filename}. Skipping.")
                except Exception as e:
                    print(f"[LoRACurationConfig] Warning: Error loading {filename}: {e}. Skipping.")
        
        self._all_curation_keys.sort() 

    def get_lora_models(self):
        return self.lora_models

    def get_curation_keys(self):
        return self.curation_keys 

    def get_lora_curation(self):
        return self.lora_curation 

    def get_adapter_weights_set_curation(self):
        return self.adapter_weights_set_curation 

    def get_default_adapter_weights(self) -> list[list[float]]:
        if self.default_curation_key and self.default_curation_key in self.adapter_weights_set_curation:
            weights = self.adapter_weights_set_curation[self.default_curation_key]
            # Ensure it's a non-empty list of lists, as expected by controlnetSDTurbot2i.py
            if not weights or not isinstance(weights[0], list):
                # This case should be handled by the constructor ensuring adapter_weights_set_curation[key] is correct
                print(f"[LoRACurationConfig] Warning: Default adapter weights for '{self.default_curation_key}' are malformed. Returning [[]].")
                return [[]]
            return weights
        
        print(f"[LoRACurationConfig] Warning: Could not get default adapter weights for key '{self.default_curation_key}'. Returning [[]] as fallback.")
        return [[]] 

    def get_all_curation_keys(self) -> list[str]:
        return self._all_curation_keys

    def get_config_for_curation(self, curation_key: str) -> dict | None:
        return self._all_curations.get(curation_key)
        
    def get_default_curation_input_params(self) -> dict:
        return self.default_curation_input_params 