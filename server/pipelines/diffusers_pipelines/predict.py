import argparse
import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers import OnnxRuntimeModel
from diffusers import AutoencoderKL
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

# Import the PromptTravel module
from modules.prompt_travel.prompt_travel import PromptTravel

from pipeline_controlnet_tensorrt import TensorRTStableDiffusionControlNetImg2ImgPipeline, TensorRTModel 