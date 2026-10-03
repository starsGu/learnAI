from __future__ import annotations

"""按真实训练路径（bf16 + 梯度检查点 + fused AdamW + 分块 loss）测量
给定 micro batch 下完成一个完整训练 step 的峰值显存。

用法（在仓库根目录）：
    python scripts/probe_batch.py --micro-batch 1 --sequence-length 2048
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain.common import build_model, choose_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one full training step at a given micro batch size")
    parser.add_argument("--model-config", default="configs/formal/model.json")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--loss-chunk-size", type=int, default=1024, help="与 pretrain.train 的默认值一致")
    args = parser.parse_args()

    device = choose_device(0)
    if device.type != "cuda":
        raise RuntimeError("this probe requires a CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args.model_config, device, "bfloat16")
    use_checkpointing = not args.no_gradient_checkpointing
    model.set_gradient_checkpointing(use_checkpointing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (args.micro_batch, args.sequence_length + 1),
        device=device,
    )
    result = {
        "parameters": model.num_parameters(),
        "micro_batch": args.micro_batch,
        "sequence_length": args.sequence_length,
        "gradient_checkpointing": use_checkpointing,
        "loss_chunk_size": args.loss_chunk_size,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
    }
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                tokens,
                labels=tokens,
                return_logits=False,
                loss_chunk_size=args.loss_chunk_size,
            ).loss
        loss.backward()
        optimizer.step()
        result.update(
            status="ok",
            loss=float(loss.detach()),
            peak_memory_gib=round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        )
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=repr(error))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
