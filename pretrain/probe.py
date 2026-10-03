from __future__ import annotations

import argparse
import json

import torch

from .common import build_model, choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one full-model training step without saving weights")
    parser.add_argument("--model-config", default="configs/formal/model.json")
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()

    device = choose_device(0)
    if device.type != "cuda":
        raise RuntimeError("the full-model probe requires a CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.model_config, device, "bfloat16")
    model.set_gradient_checkpointing(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, args.sequence_length + 1),
        device=device,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, labels=tokens, return_logits=False, loss_chunk_size=16).loss
    loss.backward()
    optimizer.step()
    result = {
        "parameters": model.num_parameters(),
        "loss": float(loss.detach()),
        "sequence_length": args.sequence_length,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
