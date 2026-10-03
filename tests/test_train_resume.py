from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pretrain.train import train
from pretrain.common import write_json

# 复用仓库内已下载的 Qwen3 分词器，避免测试联网
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "qwen3"


def build_manifest(root: Path, shard_count: int, shard_tokens: int) -> Path:
    """构造 train/validation/test 三分片清单，供真实的 train() 使用。"""
    splits = {}
    for split in ("train", "validation", "test"):
        records = []
        for index in range(shard_count):
            path = root / f"{split}-{index:05d}.bin"
            values = (np.arange(shard_tokens, dtype=np.uint32) + index).astype(np.uint32)
            values.tofile(path)
            records.append(
                {
                    "path": path.name,
                    "tokens": shard_tokens,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        splits[split] = {"tokens": shard_count * shard_tokens, "shards": records}
    path = root / "manifest.json"
    path.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return path


class ResumeFingerprintTests(unittest.TestCase):
    def test_resume_rejects_changed_data_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_manifest(root, shard_count=2, shard_tokens=64)
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
            output = root / "run"
            config = {
                "model_config": str(model_config),
                "data_manifest": str(manifest),
                "output_dir": str(output),
                "tokenizer_repo": "Qwen/Qwen3-0.6B-Base",
                "tokenizer_dir": str(root / "tokenizer"),  # 已有本地分词器目录时会直接复用
                "sequence_length": 8,
                "micro_batch_size": 1,
                "global_tokens_per_step": 8,
                "target_tokens": 8,
                "learning_rate": 1e-3,
                "warmup_ratio": 0.5,
                "log_every_steps": 1,
                "eval_every_steps": 1,
                "save_every_steps": 1,
                "validation_batches": 1,
                "gradient_checkpointing": False,
                "seed": 42,
            }
            os.environ["QWEN3_TOKENIZER_DIR"] = str(TOKENIZER_DIR)
            train(config)  # 第一次：从 step 0 训练 1 步并保存 checkpoint
            state = (output / "latest.txt").read_text(encoding="utf-8").strip()
            saved = json.loads((output / state / "trainer_state.json").read_text(encoding="utf-8"))
            self.assertIn("train_corpus_fingerprint", saved)

            # 分片清单发生变化（多了一个 shard）：全局位置不再对应同一批数据
            build_manifest(root, shard_count=3, shard_tokens=64)
            with self.assertRaises(ValueError) as captured:
                train({**config, "resume_from": str(output / "latest")})
            self.assertIn("fingerprint mismatch", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
