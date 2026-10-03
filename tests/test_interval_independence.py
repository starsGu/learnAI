from __future__ import annotations

"""日志/验证/保存的间隔只影响观测与落盘，不应改变训练轨迹。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models import Qwen3Config, Qwen3ForCausalLM
from pretrain.common import write_json
from pretrain.train import train

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "qwen3"


def build_manifest(root: Path) -> Path:
    splits = {}
    for split in ("train", "validation", "test"):
        records = []
        for index in range(2):
            path = root / f"{split}-{index:05d}.bin"
            # token 必须落在 vocab_size(64) 范围内
            (np.arange(64, dtype=np.uint32) + index).astype(np.uint32).tofile(path)
            records.append({"path": path.name, "tokens": 64, "sha256": "unused"})
        splits[split] = {"tokens": 128, "shards": records}
    path = root / "manifest.json"
    path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return path


class IntervalIndependenceTests(unittest.TestCase):
    def test_log_eval_save_intervals_do_not_change_weights(self) -> None:
        os.environ["QWEN3_TOKENIZER_DIR"] = str(TOKENIZER_DIR)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_manifest(root)
            model_config = root / "model.json"
            write_json(
                model_config,
                {
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "max_position_embeddings": 32,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                },
            )

            def run(name: str, log_every: int, eval_every: int, save_every: int) -> dict:
                output = root / name
                train(
                    {
                        "model_config": str(model_config),
                        "data_manifest": str(manifest),
                        "output_dir": str(output),
                        "tokenizer_repo": "Qwen/Qwen3-0.6B-Base",
                        "tokenizer_dir": str(root / "tokenizer"),
                        "seed": 42,
                        "sequence_length": 4,
                        "micro_batch_size": 1,
                        "global_tokens_per_step": 4,
                        "target_tokens": 16,  # 4 个 optimizer step
                        "learning_rate": 1e-3,
                        "warmup_ratio": 0.5,
                        "log_every_steps": log_every,
                        "eval_every_steps": eval_every,
                        "save_every_steps": save_every,
                        "validation_batches": 1,
                        "device": "cpu",
                        "gradient_checkpointing": False,
                    }
                )
                from safetensors.torch import load_file

                step = (output / "latest.txt").read_text(encoding="utf-8").strip()
                return load_file(output / step / "model.safetensors")

            a = run("run-a", log_every=1, eval_every=1, save_every=1)
            b = run("run-b", log_every=4, eval_every=50, save_every=50)

            self.assertEqual(set(a), set(b))
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), f"{name} 不一致")

            # 间隔只改变"存了几次"，不改变最终权重
            dense = len(list((root / "run-a").glob("step-*")))
            sparse = len(list((root / "run-b").glob("step-*")))
            self.assertEqual(dense, 4)   # save_every_steps=1，4 个 step 各存一次
            self.assertEqual(sparse, 1)  # save_every_steps=50，只在最后一步存


if __name__ == "__main__":
    unittest.main()
