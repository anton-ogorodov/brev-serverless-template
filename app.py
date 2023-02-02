import os
import torch
import random
import base64
from io import BytesIO
from transformers import pipeline
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


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, vae
    
    model_name = os.getenv("MODEL_NAME")
    vae_name = os.getenv("VAE_NAME")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16)
    model = StableDiffusionPipeline.from_pretrained(model_name, vae=vae, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    
    # model = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    negative_prompt = model_inputs.get('negative_prompt', 'child, childish')
    num_inference_steps = model_inputs.get('num_inference_steps', 30)
    guidance_scale = model_inputs.get('guidance_scale', 7)
    
    width = model_inputs.get('width', 512)
    height = model_inputs.get('height', 696)

    random_seed = random.randint(0, 4294967295)
    seed = model_inputs.get('seed', random_seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Run the model
    result = model(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, width=width, height=height)

    # Check if result is an image or text
    image = result.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
