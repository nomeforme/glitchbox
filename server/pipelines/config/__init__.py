class LoRACurationConfig:
    """
    Configuration class for LoRA curation in the controlnetSDTurbot2i pipeline.

    This class is responsible for:
    - Setting the default curation index and LoRA scale.
    - Defining a dictionary of available LoRA models and their corresponding file paths.
    - Establishing curated combinations of LoRA models for specific artistic effects.
    - Providing adapter weight sets for each curated combination to control the blending of models.
    - Offering methods to retrieve default adapter weights, LoRA models, curated combinations, and adapter weight sets.

    Attributes:
        DEFAULT_CURATION_INDEX (int): The default index for curation selection.
        DEFAULT_LORA_SCALE (float): The default scale for LoRA application.
        lora_models (dict): A dictionary mapping model names to their file paths.
        lora_curation (dict): A dictionary mapping curation names to lists of model names.
        curation_keys (list): A list of keys representing available curations.
        adapter_weights_set_curation (dict): A dictionary mapping curation names to lists of adapter weight sets.

    Methods:
        get_default_adapter_weights(): Returns the default adapter weights for the current curation.
        get_lora_models(): Returns the dictionary of LoRA models.
        get_lora_curation(): Returns the dictionary of curated LoRA combinations.
        get_curation_keys(): Returns the list of curation keys.
        get_adapter_weights_set_curation(): Returns the dictionary of adapter weight sets for curations.
    """

    def __init__(self, default_curation_index=0):
        self.DEFAULT_CURATION_INDEX = default_curation_index
        self.DEFAULT_LORA_SCALE = 1.0
    
        self.lora_models = {
            "None": None,
            "radames/sd-21-DPO-LoRA": "radames/sd-21-DPO-LoRA",
            "latent-consistency/lcm-lora-sdv2-1": "latent-consistency/lcm-lora-sdv2-1",
            "latent-consistency/lcm-lora-sdv2-1-turbo": "latent-consistency/lcm-lora-sdv2-1-turbo",
            "garance": "server/loras/garance-000038.safetensors",
            "dark": "server/loras/flowers-000022.safetensors",
            "abstract-monochrome": "server/loras/abstract-monochrome-000140.safetensors",
            "abstract-brokenglass-red": "server/loras/abstract_brokenglass_red-000140.safetensors",
            "full-body-glitch-reddish": "server/loras/full_body_glitch_reddish.safetensors",
            "melier-bw": "server/loras/melier_bw-000052.safetensors",
            "melier-col": "server/loras/melier_col-000032.safetensors",
            "nature-water": "server/loras/nature_water-000023.safetensors",
            "nature-fire": "server/loras/nature_fire-000019.safetensors",
            "nature-smoke": "server/loras/nature_smoke-000019.safetensors",
            "nature-sand": "server/loras/nature_sand-000015.safetensors",
            "robwood": "server/loras/robwood.safetensors",
            "sweet-vicious": "server/loras/sweet_vicious-000072.safetensors",
            "liquid-love": "server/loras/liquid_love-000032.safetensors",
            "glitch": "server/loras/glitch.safetensors",
            "pixels": "server/loras/pixels-000011.safetensors",
            "origami": "server/loras/origami-000100.safetensors",
            "twisted-bodies": "server/loras/twistedbodies-000014.safetensors",
            "HAHACards_A2": "server/loras/HAHACards_A2-000015.safetensors",
            "goldworld": "server/loras/goldworld-000006.safetensors",
        }

        self.lora_curation = {
            "twister_water": ["twisted-bodies", "nature-water"],
            "glitch_abstract": ["full-body-glitch-reddish", "abstract-monochrome"],
            "melier": ["melier-bw", "melier-col"],
            "liquid_nature": ["nature-bw", "nature-water"],
            "sweet_robwood": ["sweet-vicious", "robwood"],
            "glitch_pixels": ["pixels", "nature-water"],
            "marina_abstract": ["marina-red", "abstract-brokenglass-red"],
            "mid_body_glitch": ["mid-body-torso-glitch-reddish", "mid-body-shoulders-glitch-monochrome"],
            "garance": ["garance", "garance"],
            "hahacards_goldworld": ["goldworld", "HAHACards_A2"]
        }

        self.curation_keys = list(self.lora_curation.keys())

        self.adapter_weights_set_curation = {
            "glitch_abstract": [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4]
            ],
            "melier": [
                [1.0, 0.0],
                [0.85, 0.15],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.1, 0.9]
            ],
            "liquid_nature": [
                [0.6, 0.4],
                [0.6, 0.7],
                [0.4, 0.9],
                [0.3, 0.9],
                [0.2, 0.9]
            ],
            "sweet_robwood": [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.4, 0.6],
                [0.3, 0.7]
            ],
            "glitch_pixels": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ],
            "twister_water": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ],
            "marina_abstract": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ],
            "mid_body_glitch": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ],
            "garance": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ],
            "hahacards_goldworld": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0]
            ]
        }

    def get_default_adapter_weights(self):
        return self.adapter_weights_set_curation[self.curation_keys[self.DEFAULT_CURATION_INDEX]]

    def get_lora_models(self):
        return self.lora_models

    def get_lora_curation(self):
        return self.lora_curation

    def get_curation_keys(self):
        return self.curation_keys

    def get_adapter_weights_set_curation(self):
        return self.adapter_weights_set_curation 