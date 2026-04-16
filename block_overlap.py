import contextlib
import logging
import time

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

logger = logging.getLogger()


@contextlib.contextmanager
def use_stream(stream: torch.cuda.Stream):
    """Context manager for using a specific CUDA stream"""
    current = torch.cuda.current_stream()
    torch.cuda.set_stream(stream)
    yield
    torch.cuda.set_stream(current)


def offload_prepare(
    transformer: FluxTransformer2DModel,
    device,
    non_blocking: bool,
    overlap: bool,
    cleanup: bool,
):
    compute_stream = torch.cuda.current_stream()
    collect_stream = torch.cuda.Stream() if overlap else compute_stream

    old_forward = transformer.forward
    transformer.last_event = None
    transformer.last_layer = None

    def transformer_forward(self, *args, **kwargs):
        def root_modules_load():
            for module in self.named_children():
                if not isinstance(module[1], torch.nn.ModuleList):
                    module[1].to(device, non_blocking=non_blocking)
                else:
                    assert module[0] in ["transformer_blocks", "single_transformer_blocks"], module[0]

        def root_modules_offload():
            for module in self.named_children():
                if not isinstance(module[1], torch.nn.ModuleList):
                    module[1].to("cpu", non_blocking=False)
                else:
                    assert module[0] in ["transformer_blocks", "single_transformer_blocks"], module[0]
                    # assert module[0] == 'blocks', module[0]
            if cleanup:
                torch.cuda.empty_cache()

        with use_stream(collect_stream):
            root_modules_load()

        hidden_states = old_forward(*args, **kwargs)
        with use_stream(collect_stream):
            collect_stream.wait_event(self.last_event)
            self.last_layer.to(device="cpu", non_blocking=False)
            if cleanup:
                torch.cuda.empty_cache()

        with use_stream(collect_stream):
            root_modules_offload()

        return hidden_states

    def block_forward(self, *args, **kwargs):
        with use_stream(collect_stream):
            self.to(device=device, non_blocking=non_blocking)
            gather_event = torch.cuda.Event()
            gather_event.record(collect_stream)
        with use_stream(compute_stream):
            compute_stream.wait_event(gather_event)
            hidden_states = self.old_forward(*args, **kwargs)
            compute_event = torch.cuda.Event()
            compute_event.record(compute_stream)
        with use_stream(collect_stream):
            if transformer.last_event is not None:
                assert transformer.last_layer is not None, "last_layer should not be None"
                collect_stream.wait_event(transformer.last_event)
                transformer.last_layer.to(device="cpu", non_blocking=False)
                if cleanup:
                    torch.cuda.empty_cache()
        transformer.last_event = compute_event
        transformer.last_layer = self

        return hidden_states

    for block in transformer.transformer_blocks:
        block.old_forward = block.forward
        block.forward = block_forward.__get__(block)

    for block in transformer.single_transformer_blocks:
        block.old_forward = block.forward
        block.forward = block_forward.__get__(block)

    transformer.forward = transformer_forward.__get__(transformer)
    return transformer


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

    pipe.transformer = offload_prepare(
        pipe.transformer,
        device=device,
        non_blocking=True,
        overlap=True,
        cleanup=False,
    )
    return pipe


def main():
    pipe = prepare_pipeline()

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
    image.save(f"flux-dev-overlap-{int(time.time())}.png")


if __name__ == "__main__":
    main()
