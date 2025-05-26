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

    def __init__(self):

        self.DEFAULT_CURATION_INDEX = 9
        self.DEFAULT_LORA_SCALE = 1.0
    
        self.lora_models = {
            "None": None,
            "radames/sd-21-DPO-LoRA": "radames/sd-21-DPO-LoRA",
            "latent-consistency/lcm-lora-sdv2-1": "latent-consistency/lcm-lora-sdv2-1",
            "latent-consistency/lcm-lora-sdv2-1-turbo": "latent-consistency/lcm-lora-sdv2-1-turbo",
            "hakurei/waifu-diffusion": "hakurei/waifu-diffusion",
            "ostris/ikea-instructions-lora": "ostris/ikea-instructions-lora",
            "ostris/super-cereal-sdxl-lora": "ostris/super-cereal-sdxl-lora",
            "pbarbarant/sd-sonio": "pbarbarant/sd-sonio",
            "artificialguybr/studioghibli-redmond-2-1v-studio-ghibli-lora-for-freedom-redmond-sd-2-1": "artificialguybr/studioghibli-redmond-2-1v-studio-ghibli-lora-for-freedom-redmond-sd-2-1",
            "style_pi_2": "server/loras/style_pi_2.safetensors",
            "pytorch_lora_weights": "server/loras/pytorch_lora_weights.safetensors",
            "garance": "server/loras/garance-000038.safetensors",
            "dark": "server/loras/flowers-000022.safetensors",
            "marina1": "server/loras/marina-glitch-000140.safetensors",
            "marina-red": "server/loras/marina-red-000140.safetensors",
            "abstract-monochrome": "server/loras/abstract-monochrome-000140.safetensors",
            "abstract-brokenglass-red": "server/loras/abstract_brokenglass_red-000140.safetensors",
            "full-body-glitch-monochrome": "server/loras/full_body_glitch-monochrome-000140.safetensors",
            "full-body-glitch-reddish": "server/loras/full_body_glitch-reddish-000140.safetensors",
            "mid-body-shoulders-glitch-monochrome": "server/loras/mid_body_shoulders_glitch-monochrome-000140.safetensors",
            "mid-body-shoulders-glitch-reddish": "server/loras/mid_body_shoulders_glitch-reddish-000140.safetensors",
            "mid-body-torso-glitch-monochrome": "server/loras/mid_body_torso_glitch-monochrome-000140.safetensors",
            "mid-body-torso-glitch-reddish": "server/loras/mid_body_torso_glitch-reddish-000140.safetensors",
            "melier-bw": "server/loras/melier_bw-000052.safetensors",
            "melier-col": "server/loras/melier_col-000032.safetensors",
            "nature-bw": "server/loras/nature_bw-000052.safetensors",
            "nature-water": "server/loras/nature_water-000072.safetensors",
            "robwood": "server/loras/robwood-000060.safetensors",
            "sweet-vicious": "server/loras/sweet_vicious-000072.safetensors",
            "liquid-love": "server/loras/liquid_love-000032.safetensors",
            "glitch": "server/loras/glitch-000060.safetensors",
            "pixels": "server/loras/pixels-000092.safetensors",
            "origami": "server/loras/origami-000100.safetensors",
            "twisted-bodies": "server/loras/twisted bodies-000100.safetensors",
            "HAHACards_A2": "server/loras/HAHACards_A2-000015.safetensors",
            "goldworld": "server/loras/goldworld-000006.safetensors",
        }

        self.lora_curation = {
            "glitch_abstract": ["full-body-glitch-reddish", "abstract-monochrome"],
            "melier": ["melier-bw", "melier-col"],
            "liquid_nature": ["nature-bw", "nature-water"],
            "sweet_robwood": ["sweet-vicious", "robwood"],
            "glitch_pixels": ["nature-water", "pixels"],
            "origami_twisted": ["twisted-bodies", "origami"],
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
            "origami_twisted": [
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