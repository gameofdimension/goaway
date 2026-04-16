import sys
import time

import torch
from diffusers import FluxPipeline


def main():
    offload = sys.argv[1] == "offload"
    ckpt = "/warehouse/FLUX.1-dev/"
    pipe = FluxPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    prompt = "A cat holding a sign that says hello world"
    prompt = "A motorcycle parked in an ornate bank lobby"
    # warmup
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=5,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

    step = 30
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=step,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save(f"flux-dev-{int(time.time())}.png")


if __name__ == "__main__":
    main()
