from __future__ import annotations

"""10B 训练健康看板：直接读 <output_dir>/metrics.jsonl，早期就能发现异常。

用法（仓库根目录，训练进行中随时可跑）：

    python scripts/monitor_metrics.py --config configs/formal/train_10b.json --data /root/autodl-tmp/datasets
    python scripts/monitor_metrics.py --metrics checkpoints/pretrain/qwen3-0.6B-10b/metrics.jsonl

关注四类信号（对应 task.md 的 M1/M3 验收）：
  1. NaN / inf —— 出现即必须停下排查；
  2. 验证 loss 趋势 —— 与前几个验证点比较，走平或反弹就要审查数据/学习率；
  3. 吞吐漂移 —— 与最近 20 次日志的中位数比较，掉到 80% 以下提示 I/O 或抢占；
  4. 磁盘 —— checkpoint 目录占用与剩余空间，避免写满。
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.layout import DEFAULT_DATA_ROOT, configure_train_paths  # noqa: E402
from pretrain.common import read_json  # noqa: E402

MONITOR_ROOT = Path("/root/Documents/llm")  # output_dir 相对于仓库根目录


def load_rows(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    corrupt = 0
    if not path.exists():
        return rows, corrupt
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                corrupt += 1  # 进程正在写某一行时读到半行属正常
    return rows, corrupt


def health_report(rows: list[dict], output_dir: Path) -> list[str]:
    problems: list[str] = []
    train = [row for row in rows if "train_loss" in row]
    validation = [row for row in rows if "validation_loss" in row]

    if not rows:
        problems.append("metrics.jsonl 为空：训练还没写出第一条日志")
        return problems

    for row in rows:
        for key in ("train_loss", "validation_loss"):
            value = row.get(key)
            if value is None:
                continue
            if not math.isfinite(value):
                problems.append(f"step {row['step']}: {key}={value} 非有限值，必须停下排查")
    if validation:
        recent = [row["validation_loss"] for row in validation[-3:]]
        if len(recent) >= 3 and recent[-1] > min(recent):
            problems.append(
                f"验证 loss 最近三点 {[round(v, 3) for v in recent]} 未创新低；"
                "按 task.md 应优先审查数据过滤与学习率，而不是扩大数据"
            )
    if len(train) >= 25:
        recent_speed = sorted(row["tokens_per_second"] for row in train[-20:])[10]
        early_speed = sorted(row["tokens_per_second"] for row in train[:20])[10]
        if early_speed and recent_speed < early_speed * 0.8:
            problems.append(
                f"吞吐下滑：最近中位 {recent_speed:.0f} tokens/s，早期中位 {early_speed:.0f} tokens/s"
            )
    if output_dir.exists():
        usage = shutil.disk_usage(output_dir)
        if usage.free < 15 * 1024**3:
            problems.append(f"剩余磁盘仅 {usage.free / 1024**3:.1f} GB，checkpoint 可能写不下")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a health report for a running training job")
    parser.add_argument("--config", help="训练配置：自动推导 metrics.jsonl 与总步数")
    parser.add_argument("--data", default=str(DEFAULT_DATA_ROOT), help="与训练相同的 --data 根目录")
    parser.add_argument("--metrics", help="直接指定 metrics.jsonl（覆盖 --config）")
    parser.add_argument("--tail", type=int, default=8, help="展示最近几个验证点")
    args = parser.parse_args()

    if args.metrics:
        metrics_path = Path(args.metrics)
        target_tokens, total_steps = 0, 0
        output_dir = metrics_path.parent
    elif args.config:
        config = configure_train_paths(read_json(args.config), args.data)
        output_dir = Path(config["output_dir"])
        metrics_path = output_dir / "metrics.jsonl"
        target_tokens = int(config["target_tokens"])
        total_steps = math.ceil(target_tokens / int(config["global_tokens_per_step"]))
    else:
        parser.error("需要 --config 或 --metrics")

    rows, corrupt = load_rows(metrics_path)
    train = [row for row in rows if "train_loss" in row]
    validation = [row for row in rows if "validation_loss" in row]

    print(f"metrics: {metrics_path}")
    if not rows:
        print("尚无日志")
        return
    if train:
        last = train[-1]
        print(
            f"进度: step {last['step']}"
            + (f"/{total_steps} ({last['step'] / total_steps * 100:.1f}%)" if total_steps else "")
            + f"  tokens {last.get('tokens_seen', 0):,}"
            + (f"/{target_tokens:,}" if target_tokens else "")
        )
        memory = last.get("peak_memory_bytes") or 0
        print(
            f"当前: train_loss={last['train_loss']:.4f}  lr={last['learning_rate']:.3e}  "
            f"grad_norm={last['gradient_norm']:.3f}  吞吐={last['tokens_per_second']:.0f} tokens/s  "
            f"峰值显存={memory / 1024**3:.2f} GiB"
        )
    if validation:
        print(f"最近验证点（共 {len(validation)} 个）:")
        for row in validation[-args.tail :]:
            print(
                f"  step {row['step']:>6}: loss={row['validation_loss']:.4f}  "
                f"ppl={row.get('perplexity', float('nan')):,.1f}"
            )
        best = min(row["validation_loss"] for row in validation)
        print(f"最佳验证 loss: {best:.4f}")
    if corrupt:
        print(f"（跳过 {corrupt} 行半截日志，训练正在写入，属正常）")

    problems = health_report(rows, output_dir)
    print("\n健康检查:")
    if problems:
        for problem in problems:
            print(f"  [!] {problem}")
    else:
        print("  OK：无 NaN、验证 loss 有改善、吞吐稳定、磁盘充足")


if __name__ == "__main__":
    main()
