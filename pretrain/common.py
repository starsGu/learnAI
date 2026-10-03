from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from models import Qwen3Config, Qwen3ForCausalLM


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_environment() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def choose_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cuda_probe() -> tuple[bool, str]:
    """在子进程里执行一次真实 CUDA kernel，返回 (是否可用, 说明)。

    torch.cuda.is_available() 只返回布尔值；驱动版本不匹配、内核模块缺失等问题
    必须真正初始化一次 CUDA 才会暴露。CUDA_LAUNCH_BLOCKING 让错误同步抛出。
    """
    import subprocess
    import sys

    probe = (
        "import torch\n"
        "torch.cuda.init()\n"
        "torch.zeros(1, device='cuda')\n"
        "print('ok')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"探测 CUDA 失败: {error!r}"
    if result.returncode == 0 and "ok" in result.stdout:
        return True, "CUDA 可用"
    output = (result.stderr or result.stdout or "").strip().splitlines()
    return False, " / ".join(line.strip() for line in output[-4:]) or "未知原因"


def resolve_device(config: dict, local_rank: int) -> torch.device:
    """按配置决定设备；配置要求 CUDA 而 CUDA 不可用时直接报错。

    ``device`` 可选 "auto"（默认）、"cuda"、"cpu"：
    - auto：有 CUDA 用 CUDA，否则降级到 CPU 并打印醒目警告；
    - cuda：不可用即抛错，避免像以前那样静默跑 CPU（GPU 训练会慢 50-100 倍）；
    - cpu：显式要求 CPU，用于冒烟测试。
    """
    preference = str(config.get("device", "auto")).lower()
    if preference not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"device must be auto/cuda/cpu, got {preference!r}")
    available = torch.cuda.is_available()
    if preference == "cpu":
        return torch.device("cpu")
    if available:
        return choose_device(local_rank)
    if preference == "cuda":
        raise RuntimeError(
            "配置要求 CUDA（device=cuda）但 torch.cuda.is_available() 为 False。\n"
            f"诊断: {cuda_probe()[1]}\n"
            "请检查显卡驱动、实例是否挂载 GPU、CUDA_VISIBLE_DEVICES；"
            "如确实要在 CPU 上跑冒烟测试，请在配置里显式写 \"device\": \"cpu\"。"
        )
    print(
        "[警告] CUDA 不可用，已降级到 CPU。GPU 训练通常快 50-100 倍，正式训练请勿这样跑。\n"
        f"[警告] 诊断: {cuda_probe()[1]}",
        flush=True,
    )
    return torch.device("cpu")


def unwrap_model(model: torch.nn.Module) -> Qwen3ForCausalLM:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def build_model(model_config_path: str | Path, device: torch.device, dtype_name: str) -> Qwen3ForCausalLM:
    config = Qwen3Config.from_json(model_config_path)
    model = Qwen3ForCausalLM(config)
    dtype = torch.bfloat16 if dtype_name == "bfloat16" and device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype)


def resolve_checkpoint(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        return target
    if target.name == "latest":
        pointer = target.parent / "latest.txt"
        if pointer.exists():
            resolved = target.parent / pointer.read_text(encoding="utf-8").strip()
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"checkpoint not found: {target}")


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[Qwen3ForCausalLM, dict, Path]:
    from safetensors.torch import load_model

    checkpoint_dir = resolve_checkpoint(path)
    model_config_path = checkpoint_dir / "config.json"
    config = Qwen3Config.from_json(model_config_path)
    model = Qwen3ForCausalLM(config).to(device)
    load_model(model, checkpoint_dir / "model.safetensors", strict=True)
    model.tie_weights()
    train_config = read_json(checkpoint_dir / "train_config.json")
    return model, train_config, checkpoint_dir
