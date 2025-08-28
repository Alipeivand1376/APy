# Upgraded Stable Diffusion Code
# This enhanced Python script builds on your original code by adding several new 
# features that give you more control and flexibility when generating images with 
# the Stable Diffusion model.

import torch
from diffusers import StableDiffusionPipeline
import random
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("Warning: CUDA is not available")

# Initializing by a pretrained model
model_id = "runwayml/stable-diffusion-v1-5"
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32 # ---> appropriate GPU data type is float16
    ).to(device)
    print(f"Model '{model_id}' loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()


prompt = input("Enter your prompt (e.g., 'A cinematic shot of a red car'): ")
negative_prompt = input("Enter a negative prompt (e.g., 'blurry, bad quality'): ")
try:
    num_inference_steps = int(input("Enter number of inference steps (e.g., 30): "))
except ValueError:
    print("Invalid input. Using default value of 30 steps.")
    num_inference_steps = 30

seed = random.randint(0, 100000)
generator = torch.Generator(device=device).manual_seed(seed)
print(f"Using seed: {seed}")

print("Generating image...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=7.5, # between 5 to 9 is reasonable
    generator=generator
).images[0]

output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_path = os.path.join(output_dir, f"generated_image_seed_{seed}.png")
image.save(image_path)
print(f"Image saved to: {image_path}")


image.show()