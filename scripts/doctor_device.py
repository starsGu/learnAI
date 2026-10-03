from __future__ import annotations

"""GPU 环境自检：开跑 10B 前跑一次，确认 CUDA 真的可用。

    python scripts/doctor_device.py

检查项：nvidia-smi 驱动、PyTorch CUDA 构建、内核模块、设备节点、
torch.cuda.is_available()，以及一次真实的 kernel 执行（只有它能暴露驱动版本不匹配）。
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pretrain.common import cuda_probe  # noqa: E402


def show(title: str, value: str) -> None:
    print(f"  {title:26s} {value}")


def main() -> None:
    print("== 1. 驱动 ==")
    smi = shutil.which("nvidia-smi")
    show("nvidia-smi", smi or "未找到")
    if smi:
        result = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                                capture_output=True, text=True)
        show("驱动查询", (result.stdout or result.stderr).strip().replace("\n", " | ") or "(空)")

    print("== 2. 内核模块与设备 ==")
    modules = Path("/proc/modules")
    loaded = [line.split()[0] for line in modules.read_text().splitlines()] if modules.exists() else []
    show("已加载 nvidia 模块", ", ".join(m for m in loaded if m.startswith("nvidia")) or "无")
    nodes = sorted(path.name for path in Path("/dev").glob("nvidia*"))
    show("设备节点", ", ".join(nodes[:8]) + (" ..." if len(nodes) > 8 else "") or "无")

    print("== 3. PyTorch ==")
    show("torch 版本", torch.__version__)
    show("编译时 CUDA", str(torch.version.cuda))
    show("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "(未设置)"))
    show("is_available()", str(torch.cuda.is_available()))
    show("device_count()", str(torch.cuda.device_count()))
    if torch.cuda.is_available():
        show("设备名", torch.cuda.get_device_name(0))
        show("显存", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    print("== 4. 真实 kernel 探测 ==")
    ok, reason = cuda_probe()
    show("结果", reason)

    print()
    if ok:
        print("结论：CUDA 可用，可以开跑正式训练。")
    else:
        print("结论：CUDA 不可用 -> 正式训练会失败或退化到 CPU（慢 50-100 倍），先修环境再开跑。")
        print("      常见原因：实例未挂载 GPU / 驱动未加载 / 驱动版本低于 torch 编译时的 CUDA 版本。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
