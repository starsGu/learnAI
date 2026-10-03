from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets import PackedTokenCorpus
from datasets.layout import DEFAULT_DATA_ROOT, configure_train_paths
from datasets.tokenizer import ensure_tokenizer
from .checkpoint import load_training_checkpoint, save_checkpoint
from .common import (
    build_model,
    choose_device,
    distributed_environment,
    read_json,
    resolve_device,
    seed_everything,
    unwrap_model,
    write_json,
)


def cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


# 验证集固定使用同一随机排列种子：每次验证评估的都是同一个固定子集，
# 不同 step / 不同 checkpoint 的 validation loss 才可直接比较。
VALIDATION_SEED = 12345


@torch.no_grad()
def evaluate_loss(model, corpus, config, device, rank, world_size) -> float:
    model.eval()
    losses = torch.zeros(2, device=device, dtype=torch.float64)
    batches = int(config["validation_batches"])
    batch_size = int(config["micro_batch_size"])
    sequence_length = int(config["sequence_length"])
    for index in range(batches):
        offset = index * batch_size * world_size
        tokens = corpus.batch(offset, batch_size, rank, world_size, VALIDATION_SEED, device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with autocast:
            output = model(tokens, labels=tokens, return_logits=False)
        losses[0] += output.loss.double() * batch_size * sequence_length
        losses[1] += batch_size * sequence_length
    if world_size > 1:
        dist.all_reduce(losses)
    model.train()
    return float((losses[0] / losses[1]).item())


def train(config: dict) -> dict:
    # 先确认设备：配置要求 CUDA 而不可用时立刻失败，避免静默降级到 CPU 烧时间
    device = resolve_device(config, int(os.environ.get("LOCAL_RANK", "0")))
    rank, local_rank, world_size = distributed_environment()
    configured_gpus = int(config.get("num_gpus", world_size))
    if configured_gpus != world_size:
        raise ValueError(
            f"num_gpus={configured_gpus}, but torchrun started WORLD_SIZE={world_size}; "
            "set --nproc-per-node and num_gpus to the same value"
        )
    device = choose_device(local_rank)
    seed = int(config.get("seed", 42))
    seed_everything(seed + rank)

    model = build_model(config["model_config"], device, config.get("dtype", "bfloat16"))
    if config.get("gradient_checkpointing", True):
        model.set_gradient_checkpointing(True)
    raw_model = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer_kwargs = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(config.get("betas", [0.9, 0.95])),
        "eps": float(config.get("adam_epsilon", 1e-8)),
        "weight_decay": float(config.get("weight_decay", 0.1)),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(raw_model.parameters(), **optimizer_kwargs)

    sequence_length = int(config["sequence_length"])
    batch_size = int(config["micro_batch_size"])
    requested_global_tokens = int(config["global_tokens_per_step"])
    micro_global_tokens = sequence_length * batch_size * world_size
    if requested_global_tokens % micro_global_tokens:
        raise ValueError("global_tokens_per_step must be divisible by sequence_length * micro_batch_size * world_size")
    accumulation_steps = requested_global_tokens // micro_global_tokens
    target_tokens = int(config["target_tokens"])
    total_steps = math.ceil(target_tokens / requested_global_tokens)
    warmup_steps = max(1, math.ceil(total_steps * float(config.get("warmup_ratio", 0.01))))
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        float(config.get("minimum_learning_rate_ratio", 0.1)),
    )

    train_corpus = PackedTokenCorpus(config["data_manifest"], "train", sequence_length)
    validation_corpus = PackedTokenCorpus(config["data_manifest"], "validation", sequence_length)
    tokenizer = ensure_tokenizer(config["tokenizer_repo"], config["tokenizer_dir"])
    state = {"step": 0, "tokens_seen": 0, "global_sequence_offset": 0, "best_validation_loss": None}
    if config.get("resume_from"):
        state = load_training_checkpoint(config["resume_from"], raw_model, optimizer, scheduler, rank)
        # 全局位置只在分片清单不变时才指向同一批数据；清单变了就必须从头训练，
        # 否则会出现"部分数据重复、部分数据从未见过"。
        fingerprint = train_corpus.fingerprint()
        previous = state.get("train_corpus_fingerprint")
        if previous is not None and previous != fingerprint:
            raise ValueError(
                "data manifest changed since this checkpoint (train shard fingerprint mismatch); "
                "resume would skip or repeat data, so restart training from step 0"
            )
        state["train_corpus_fingerprint"] = fingerprint
    else:
        state["train_corpus_fingerprint"] = train_corpus.fingerprint()

    output_root = Path(config["output_dir"])
    if (
        not config.get("resume_from")
        and (output_root / "latest.txt").exists()
    ):
        raise FileExistsError(
            f"{output_root} already contains checkpoints; set resume_from or choose another output_dir"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.jsonl"
    initial_validation_loss = evaluate_loss(model, validation_corpus, config, device, rank, world_size)
    state.setdefault("initial_validation_loss", initial_validation_loss)
    if rank == 0 and state["step"] == 0:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 0, "validation_loss": initial_validation_loss}) + "\n")

    run_start_tokens = int(state["tokens_seen"])
    start_time = time.perf_counter()
    try:
        model.train()
        while state["step"] < total_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for micro_step in range(accumulation_steps):
                offset = int(state["global_sequence_offset"])
                tokens = train_corpus.batch(offset, batch_size, rank, world_size, seed, device)
                state["global_sequence_offset"] = offset + batch_size * world_size
                sync_context = (
                    model.no_sync()
                    if world_size > 1 and micro_step < accumulation_steps - 1
                    else nullcontext()
                )
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with sync_context, autocast:
                    output = model(tokens, labels=tokens, return_logits=False)
                    loss = output.loss / accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {state['step']}")
                loss.backward()
                accumulated_loss += float(loss.detach())

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            state["tokens_seen"] += requested_global_tokens
            elapsed = max(time.perf_counter() - start_time, 1e-6)

            # 多卡时 rank0 只看到自己那部分 micro-batch，日志里的 train_loss 必须取全局平均，
            # 这样 1 卡与多卡的 train_loss 才可比（梯度本身已是全局等价，不受影响）。
            logged_loss = accumulated_loss
            if world_size > 1:
                loss_tensor = torch.tensor(accumulated_loss, device=device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                logged_loss = float(loss_tensor.item() / world_size)

            if state["step"] % int(config["log_every_steps"]) == 0 and rank == 0:
                metric = {
                    "step": state["step"],
                    "tokens_seen": state["tokens_seen"],
                    "train_loss": logged_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": float(gradient_norm),
                    "tokens_per_second": (state["tokens_seen"] - run_start_tokens) / elapsed,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric) + "\n")

            if state["step"] % int(config["eval_every_steps"]) == 0 or state["step"] == total_steps:
                validation_loss = evaluate_loss(model, validation_corpus, config, device, rank, world_size)
                best = state["best_validation_loss"]
                state["best_validation_loss"] = validation_loss if best is None else min(best, validation_loss)
                if rank == 0:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": state["step"], "validation_loss": validation_loss, "perplexity": math.exp(min(validation_loss, 20.0))}) + "\n")

            if state["step"] % int(config["save_every_steps"]) == 0 or state["step"] == total_steps:
                save_checkpoint(
                    output_root,
                    model,
                    optimizer,
                    scheduler,
                    tokenizer,
                    config,
                    state,
                    rank,
                    world_size,
                    keep_recent=int(config.get("keep_recent_checkpoints", 0)),
                    keep_total=int(config.get("keep_total_checkpoints", 0)),
                )
    except Exception as error:
        if rank == 0:
            write_json(
                output_root / "diagnostic.json",
                {"error": repr(error), "traceback": traceback.format_exc(), "state": state},
            )
        raise
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 weights from random initialization")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_ROOT),
        help="Root containing processed training data and tokenizer files (default: D:/datasets/llm)",
    )
    parser.add_argument("--resume-from", help="Checkpoint directory or a latest pointer")
    args = parser.parse_args()
    config = configure_train_paths(read_json(args.config), args.data)
    if args.resume_from:
        config["resume_from"] = args.resume_from
    state = train(config)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
