from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_model, save_model

from models import Qwen3ForCausalLM
from .common import resolve_checkpoint, unwrap_model, write_json


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _align_cuda_rng_states(states, device_count: int, *, warn: bool = False) -> list:
    """把保存的 CUDA 随机状态对齐到当前可见的卡数。

    在 N 卡上存的 checkpoint 会包含 N 份 CUDA 状态；换到 M 卡续训时：
    - M < N：只恢复前 M 份，忽略多出来的（保存时的设备已不存在）；
    - M > N：多出来的卡在保存时不存在，**复制保存的第一份状态**（不用当前设备的
      默认状态，因为续训时那份默认状态已经被加载过程扰动过）。
    不这样对齐就会在 torch.cuda.set_rng_state_all 里 IndexError。
    """
    states = list(states)
    if not states:
        return [torch.cuda.get_rng_state(index) for index in range(device_count)]
    if len(states) == device_count:
        return states
    if warn:
        print(
            f"[警告] checkpoint 保存时可见 {len(states)} 张卡，当前 {device_count} 张；"
            "随机状态已按当前卡数对齐（数据顺序与卡数无关，不影响训练等价性）",
            flush=True,
        )
    if len(states) > device_count:
        return states[:device_count]
    return states + [states[0]] * (device_count - len(states))


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        states = _align_cuda_rng_states(
            state["cuda"], torch.cuda.device_count(), warn=True
        )
        torch.cuda.set_rng_state_all(states)


def prune_checkpoints(root: str | Path, keep_recent: int) -> list[str]:
    """滚动保留：只让最近 keep_recent 个 checkpoint 保留优化器/调度器/RNG。

    单卡 0.6B 的一个完整 checkpoint 约 5.5GB（模型 1.11GB + 优化器 4.44GB），
    10B 训练若全量保留会占 200GB 以上。更早的 checkpoint 只留模型与元数据，
    既能继续训练（optimizer 可从最近一个恢复）、又能评测与导出。
    """
    root = Path(root)
    steps = sorted(
        (path for path in root.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    heavy = {"optimizer.pt", "scheduler.pt"}
    trimmed: list[str] = []
    for directory in steps[:-keep_recent] if keep_recent > 0 else steps:
        for name in heavy:
            candidate = directory / name
            if candidate.exists():
                candidate.unlink()
        for stale in directory.glob("rng-rank-*.pt"):
            stale.unlink()
        trimmed.append(directory.name)
    return trimmed


def enforce_checkpoint_window(root: str | Path, keep_total: int) -> list[str]:
    """目录数上限：删除最旧的 checkpoint 目录，避免每 N 步存一个把磁盘填满。

    returns: 被整体删除的 checkpoint 目录名（最新一个永不删除）。
    """
    root = Path(root)
    if keep_total <= 0:
        return []
    steps = sorted(
        (path for path in root.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    removed: list[str] = []
    for directory in steps[:-keep_total]:
        shutil.rmtree(directory)
        removed.append(directory.name)
    return removed


def save_checkpoint(
    root: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer,
    train_config: dict,
    trainer_state: dict,
    rank: int,
    world_size: int,
    keep_recent: int = 0,
    keep_total: int = 0,
) -> Path:
    root = Path(root)
    name = f"step-{trainer_state['step']:08d}"
    temporary = root / f".{name}.incomplete"
    final = root / name
    if rank == 0:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        raw_model = unwrap_model(model)
        save_model(raw_model, temporary / "model.safetensors")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        write_json(temporary / "config.json", raw_model.config.to_dict())
        write_json(temporary / "train_config.json", train_config)
        write_json(temporary / "trainer_state.json", trainer_state)
        tokenizer.save_pretrained(temporary)
    if world_size > 1:
        dist.barrier()
    torch.save(_rng_state(), temporary / f"rng-rank-{rank:05d}.pt")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        # os.replace 无法覆盖已存在的非空目录：同一 step 重复保存（中断后重跑）时先清掉旧的
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        root.mkdir(parents=True, exist_ok=True)
        (root / "latest.txt").write_text(name, encoding="utf-8")
        if keep_recent > 0:
            prune_checkpoints(root, keep_recent)
        if keep_total > 0:
            enforce_checkpoint_window(root, keep_total)
    if world_size > 1:
        dist.barrier()
    return final


def load_training_checkpoint(
    path: str | Path,
    model: Qwen3ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
) -> dict:
    checkpoint_dir = resolve_checkpoint(path)
    if not (checkpoint_dir / "optimizer.pt").exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} 已被滚动保留裁剪（只剩模型权重），无法精确续训；"
            "请改用最新的 checkpoint 恢复"
        )
    load_model(model, checkpoint_dir / "model.safetensors", strict=True)
    model.tie_weights()
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=False))
    # 单卡 checkpoint 只有 rng-rank-00000.pt；1 卡暂停改多卡续训时，
    # 新增 rank 回退到 rank0 的随机状态（数据顺序与 world size 无关，故不影响等价性）。
    rng_path = checkpoint_dir / f"rng-rank-{rank:05d}.pt"
    if not rng_path.exists():
        rng_path = checkpoint_dir / "rng-rank-00000.pt"
    _restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))
    from .common import read_json

    return read_json(checkpoint_dir / "trainer_state.json")
