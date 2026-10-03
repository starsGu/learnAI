from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
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
    """构建模型：默认参数保持 fp32（master weights），bf16 只用于 autocast 前向。

    为什么不能用 model.to(bfloat16) 直接转参数：那样 AdamW 的参数更新在 bf16 里进行，
    而 bf16 在 1.0 附近分辨率约 3e-5，学习率量级的更新（~3e-4）会被舍入掉，
    RMSNorm 这类初始值为 1.0 的参数会永久冻结（实测 65536/65536 个元素零更新）。

    若配置里设置 `master_weights: false`，模型转 bf16 省显存，改用 fp32 主权重副本
    （见 MasterWeightOptimizer）来保证更新精度——显存与"全 fp32 参数"方案相当，
    但参数与梯度仍是 bf16。
    """
    config = Qwen3Config.from_json(model_config_path)
    model = Qwen3ForCausalLM(config)
    if dtype_name == "bfloat16":
        return model.to(device=device, dtype=torch.float32)
    return model.to(device=device)


def amp_context(device: torch.device):
    """bf16 前向计算上下文；CUDA 与 CPU 都支持，其他设备退化为空上下文。"""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def parameter_dtypes(model: torch.nn.Module) -> dict[str, int]:
    """按 dtype 统计参数量，用于核对"参数确实是 fp32 主权重"。"""
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


class MasterWeightOptimizer:
    """低精度参数 + fp32 主权重副本：省显存，同时保证小更新不被舍入掉。

    适用场景：参数存 bf16（省 2.22 GiB），但直接让 AdamW 更新 bf16 参数会把
    1.0 附近的微小更新舍入掉（RMSNorm 永久冻结）。这里为每个参数维护一份 fp32
    主权重：梯度从 bf16 参数搬过来、AdamW 在 fp32 上更新、再写回 bf16 参数。

    用法：
        optimizer = MasterWeightOptimizer(model, AdamW(model.parameters(), **kwargs))
        loss.backward()
        optimizer.step()      # 内部自动同步梯度 -> 更新主权重 -> 回写参数
        optimizer.zero_grad() # 内部清空主权重梯度
    """

    def __init__(self, model: torch.nn.Module, inner: torch.optim.Optimizer | None = None) -> None:
        self.inner = inner
        self.masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen: set[int] = set()
        for parameter in model.parameters():
            if id(parameter) in seen:  # tied weights 会出现两次
                continue
            seen.add(id(parameter))
            master = parameter.detach().to(torch.float32).clone().requires_grad_(True)
            self.masters.append((parameter, master))

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, master in self.masters:
            master.grad = None
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for parameter, master in self.masters:
            if parameter.grad is not None:
                if master.grad is None:
                    master.grad = master.detach().clone().zero_()
                master.grad.add_(parameter.grad)  # bf16 梯度 -> fp32
        result = self.inner.step(closure)
        with torch.no_grad():
            for parameter, master in self.masters:
                parameter.copy_(master)  # fp32 主权重 -> bf16 参数
                parameter.grad = None
                master.grad = None
        return result

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.inner.load_state_dict(state["inner"])


def build_optimizer(model: torch.nn.Module, device: torch.device, config: dict):
    """按配置构建优化器；参数是低精度时自动改用 fp32 主权重。

    注意：AdamW 必须建在**主权重**上。如果建在 bf16 模型参数上，更新会被舍入掉，
    主权重就成了摆设（早期实现踩过这个坑）。
    """
    kwargs = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(config.get("betas", [0.9, 0.95])),
        "eps": float(config.get("adam_epsilon", 1e-8)),
        "weight_decay": float(config.get("weight_decay", 0.1)),
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    needs_master = any(parameter.dtype != torch.float32 for parameter in model.parameters())
    if not needs_master:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    wrapper = MasterWeightOptimizer(model)
    # 优化器只持有 fp32 主权重：更新精度不再受 bf16 舍入影响
    wrapper.inner = torch.optim.AdamW([master for _, master in wrapper.masters], **kwargs)
    return wrapper


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
    # 与训练保持同一数值口径：master_weights=false 的训练参数存 bf16
    if train_config.get("master_weights", True) is False and device.type == "cuda":
        model = model.to(torch.bfloat16)
    return model, train_config, checkpoint_dir
