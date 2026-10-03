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


def build_model(model_config_path: str | Path, device: torch.device,
                dtype_name: str, low_precision: bool = False) -> Qwen3ForCausalLM:
    """构建模型。

    - 默认（low_precision=False）：参数保持 fp32（master weights），bf16 只用于
      autocast 前向。理由：model.to(bfloat16) 会让 AdamW 的更新在 bf16 里进行，
      而 bf16 在 1.0 附近分辨率约 3e-5，学习率量级的更新（~3e-4）会被舍入掉，
      RMSNorm 这类初始值为 1.0 的参数会永久冻结（实测 65536/65536 元素零更新）。
    - low_precision=True（配置 master_weights=false）：参数与梯度存 bf16 以省显存
      （参数 1.11 + 梯度 1.11 + AdamW 2.22 ≈ 4.44 GiB，fp32 方案是 8.88 GiB），
      归一化权重的更新精度由 MasterWeightOptimizer 的 fp32 主权重保证。
    """
    del dtype_name  # 保留签名兼容；精度由 low_precision 决定
    config = Qwen3Config.from_json(model_config_path)
    model = Qwen3ForCausalLM(config)
    if low_precision:
        return model.to(device=device, dtype=torch.bfloat16)
    return model.to(device=device, dtype=torch.float32)


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


def needs_master_weight(name: str) -> bool:
    """哪些参数必须用 fp32 主权重更新。

    只挑初始值为 1.0 的归一化权重：bf16 在 1.0 附近分辨率约 3e-5，学习率量级
    的更新（~3e-4）会被舍入掉，实测这些参数会永久冻结。其余参数（Linear/Embedding）
    初始化 std=0.02，更新量相对自身量级足够大，bf16 存储不会丢更新。

    只为 65,536 个元素（占 0.011%）配主权重，开销约 0.25 MiB，而不是全量 2.22 GiB。
    """
    return "norm" in name and name.endswith(".weight")


class MasterWeightOptimizer(torch.optim.Optimizer):
    """低精度参数 + fp32 主权重副本：给必须精确更新的参数保留 fp32 更新精度。

    直接让 AdamW 更新 bf16 参数时，1.0 附近的微小更新会被舍入掉（RMSNorm 永久冻结）。
    这里对选中的参数额外维护一份 fp32 主权重：梯度从 bf16 参数搬过来、优化器在 fp32
    上更新、再写回 bf16 参数；未选中的参数仍直接更新（省显存）。

    必须继承 torch.optim.Optimizer：torch.optim.lr_scheduler.LambdaLR 会做
    isinstance(optimizer, Optimizer) 检查，包装类不继承会在构造调度器时抛 TypeError。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, config: dict,
                 selector=needs_master_weight) -> None:
        self.masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        self.parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) in seen:  # tied weights 会出现两次
                continue
            seen.add(id(parameter))
            self.parameters.append(parameter)
            if selector(name):
                master = parameter.detach().to(torch.float32).clone().requires_grad_(True)
                self.masters.append((parameter, master))

        # 关键参数的更新交给 fp32 主权重；其余参数直接更新
        selected = {id(parameter) for parameter, _ in self.masters}
        direct = [p for p in self.parameters if id(p) not in selected]
        grouped = [master for _, master in self.masters] + direct
        kwargs = {
            "lr": float(config["learning_rate"]),
            "betas": tuple(config.get("betas", [0.9, 0.95])),
            "eps": float(config.get("adam_epsilon", 1e-8)),
            "weight_decay": float(config.get("weight_decay", 0.1)),
        }
        # fused AdamW 要求整个参数组 dtype 一致，而这里必然混用
        # （fp32 主权重 + bf16 直接更新参数），因此不能开 fused。
        self.inner = torch.optim.AdamW(grouped, **kwargs)
        # 基类拿到同一批参数与超参：这样 LR 调度器等工具看到的就是真实的参数组
        super().__init__(grouped, kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # 未配主权重的参数不在 inner 里，必须自己清，否则梯度会跨 step 累积
        for parameter in self.parameters:
            parameter.grad = None
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
        for parameter in self.parameters:
            parameter.grad = None
        for _, master in self.masters:
            master.grad = None   # 不清会在下一个 step 里把本次梯度再累加一次
        return result

    # 注意：不要覆写 self.state / self.param_groups —— 基类 __init__ 会给它们赋值，
    # 定义成只读 property 会在构造时抛 "property has no setter"。
    # 模型参数组已交给基类，因此调度器读到的 LR 就是真实 LR；
    # AdamW 的动量等状态由下面的方法单独存取，供 checkpoint 保存。
    def optimizer_state_dict(self) -> dict:
        return self.inner.state_dict()

    def load_optimizer_state_dict(self, state: dict) -> None:
        # 兼容旧 checkpoint：此前存的是 {"inner": ...}
        self.inner.load_state_dict(state.get("inner", state))


def build_optimizer(model: torch.nn.Module, device: torch.device, config: dict):
    """构建优化器；参数是低精度时为关键参数（归一化权重）改走 fp32 主权重。

    参数全为 fp32 时直接用 AdamW；否则返回 MasterWeightOptimizer（它本身是
    torch.optim.Optimizer 的子类，可被 LambdaLR 等工具正常使用）。
    """
    if all(parameter.dtype == torch.float32 for parameter in model.parameters()):
        kwargs = {
            "lr": float(config["learning_rate"]),
            "betas": tuple(config.get("betas", [0.9, 0.95])),
            "eps": float(config.get("adam_epsilon", 1e-8)),
            "weight_decay": float(config.get("weight_decay", 0.1)),
        }
        if device.type == "cuda":
            kwargs["fused"] = True
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return MasterWeightOptimizer(model, device, config)


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
