from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import DiffusionPipeline
from io import BytesIO
import base64

app=FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
access_token="Your Huggingface Token"
model_id="CompVis/stable-diffusion-v1-4"
device ="cuda"
pipe = DiffusionPipeline.from_pretrained(model_id, token=access_token, torch_dtype=torch.float32, use_safetensors=True)
pipe.to(device)
pipe.enable_attention_slicing()


@app.get("/")
async def check():
    return {"Message":"Working"}

@app.post("/im_generate/")
async def generate(prompt: str):
 
    image = pipe(prompt, guidance_scale=8.5).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")