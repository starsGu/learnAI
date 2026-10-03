from __future__ import annotations

"""显存与设备诊断：在**你自己的机器**上跑，把真实数字打出来。

    python scripts/diagnose_memory.py
    python scripts/diagnose_memory.py --micro-batch 16 --sequence-length 2048   # 只测单个配置
    python scripts/diagnose_memory.py --sweep                                    # 扫描多个 micro_batch

为什么需要它：训练 OOM 时的内存构成无法从代码推算（碎片、CUDA context、
cuDNN workspace 都算不准）。这个脚本按真实训练路径（bf16 autocast + 梯度检查点
+ fused AdamW + 分块 loss）逐步测量，定位到具体是哪一步爆的。
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from pretrain.common import amp_context, build_optimizer, parameter_dtypes  # noqa: E402


def gib(value: int) -> float:
    return value / 1024**3


class MemoryTrace:
    """逐步记录每次测量相对基线的增量，直接看出是哪一块吃掉了显存。"""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.base = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        self.rows: list[tuple[str, float, float]] = []

    def mark(self, label: str) -> None:
        if self.device.type != "cuda":
            return
        allocated = torch.cuda.memory_allocated(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        self.rows.append((label, gib(allocated), gib(peak)))

    def report(self) -> None:
        if not self.rows:
            return
        print(f"\n{'阶段':34s} {'已分配':>10s} {'峰值':>10s} {'增量':>8s}")
        previous = self.base
        for label, allocated, peak in self.rows:
            print(f"{label:34s} {allocated:9.2f}G {peak:9.2f}G {allocated - previous:+7.2f}G")
            previous = allocated


def measure(config_path: str, micro_batch: int, sequence_length: int, master_weights: bool) -> dict:
    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    trace = MemoryTrace(device)
    trace.mark("起点")

    config = Qwen3Config.from_json(config_path)
    dtype = torch.float32 if master_weights else torch.bfloat16
    model = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    trace.mark(f"模型（{'fp32' if master_weights else 'bf16'} 参数）")

    optimizer = build_optimizer(model, device, {"learning_rate": 3e-4, "weight_decay": 0.1})
    # 优化器状态要等第一次 step 才分配
    trace.mark("优化器对象")

    tokens = torch.randint(0, config.vocab_size, (micro_batch, sequence_length + 1), device=device)
    trace.mark("输入 token")

    try:
        with amp_context(device):
            loss = model(tokens, labels=tokens, return_logits=False).loss
        trace.mark("前向 + 分块 loss")
        loss.backward()
        trace.mark("反向（梯度）")
        optimizer.step()
        trace.mark("优化器 step（状态分配）")
    except torch.cuda.OutOfMemoryError as error:
        trace.mark("OOM")
        trace.report()
        return {
            "micro_batch": micro_batch,
            "sequence_length": sequence_length,
            "master_weights": master_weights,
            "status": "oom",
            "error": str(error).splitlines()[0],
            "peak_gib": round(gib(torch.cuda.max_memory_allocated(device)), 3),
        }

    trace.report()
    return {
        "micro_batch": micro_batch,
        "sequence_length": sequence_length,
        "master_weights": master_weights,
        "parameters": model.num_parameters(),
        "parameter_dtypes": parameter_dtypes(model),
        "status": "ok",
        "peak_gib": round(gib(torch.cuda.max_memory_allocated(device)), 3),
        "reserved_gib": round(gib(torch.cuda.max_memory_reserved(device)), 3),
        "loss": float(loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real GPU memory for the training step")
    parser.add_argument("--model-config", default="configs/formal/model.json")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--sweep", action="store_true", help="扫描 micro_batch 1/2/4/8/16")
    parser.add_argument("--no-master-weights", action="store_true", help="参数存 bf16（无 fp32 主权重）")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用：脚本必须在你训练的机器上运行")
        raise SystemExit(1)

    print("=== 设备 ===")
    print(f"  torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"  设备数 {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        print(f"  cuda:{index} {properties.name} 总 {gib(total):.2f} GiB 空闲 {gib(free):.2f} GiB")

    batches = [1, 2, 4, 8, 16] if args.sweep else [args.micro_batch]
    results = []
    for micro_batch in batches:
        print(f"\n=== micro_batch={micro_batch} sequence_length={args.sequence_length} ===")
        results.append(measure(args.model_config, micro_batch, args.sequence_length,
                               master_weights=not args.no_master_weights))
    print("\n=== 汇总 ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
