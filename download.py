# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os, torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import (
    # DDPMScheduler,
    # DDIMScheduler,
    # PNDMScheduler,
    # LMSDiscreteScheduler,
    # EulerDiscreteScheduler,
    # EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_name = os.getenv("MODEL_NAME")
    vae_name = os.getenv("VAE_NAME")

    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16)

    model = StableDiffusionPipeline.from_pretrained(model_name, vae=vae, scheduler=scheduler, torch_dtype=torch.float16)
    

if __name__ == "__main__":
    download_model()