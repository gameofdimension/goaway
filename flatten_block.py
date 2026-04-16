import time

import torch
import torch.nn as nn
from diffusers import FluxPipeline


def apply_flatten(block: nn.Module) -> None:
    """Flatten all parameters of a block into a single 1D nn.Parameter (shared-storage).

    After calling this, all individual parameters share storage with
    ``block._flat_param`` — modifying one updates the others automatically.

    **Must be called after** ``.to(device)``, otherwise the views become stale
    after a subsequent device transfer.
    """
    params = list(block.parameters())
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    flat.requires_grad_(False)

    offset = 0
    for p in params:
        numel = p.numel()
        p.data = flat[offset : offset + numel].view(p.shape)
        offset += numel

    block._flat_param = flat


def prepare_pipeline():
    ckpt = "/warehouse/FLUX.1-dev/"
    pipe = FluxPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16)

    device = "cuda"
    pipe.vae.to(device)
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    # pipe.transformer.to(device)
    if pipe.image_encoder is not None:
        pipe.image_encoder.to(device)
    if pipe.feature_extractor is not None:
        pipe.feature_extractor.to(device)
    return pipe


def main():
    pipe = prepare_pipeline()

    # Apply flatten to every transformer block (must be after .to / offload)
    for block in pipe.transformer.transformer_blocks:
        apply_flatten(block)
    for block in pipe.transformer.single_transformer_blocks:
        apply_flatten(block)

    prompt = "A cat holding a sign that says hello world"
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

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save(f"flux-dev-flatten-{int(time.time())}.png")


if __name__ == "__main__":
    main()
