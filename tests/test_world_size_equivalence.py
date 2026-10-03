from __future__ import annotations

"""1 卡 / 多卡等价性回归：全局 batch 的数据组成与梯度定义都与 world_size 无关。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.packed import PackedTokenCorpus
from models import Qwen3Config, Qwen3ForCausalLM

SEQUENCE_LENGTH = 8
LOCAL_BATCH = 4
GLOBAL_BATCH = 32  # 序列数：global_tokens_per_step / sequence_length
TOTAL_SEQUENCES = 4096
SEED = 42


def build_corpus(root: Path) -> PackedTokenCorpus:
    # 每条序列需要 sequence_length+1 个 token，构造出恰好 TOTAL_SEQUENCES 条
    tokens = (TOTAL_SEQUENCES - 1) * SEQUENCE_LENGTH + SEQUENCE_LENGTH + 1
    path = root / "shard-00000.bin"
    np.arange(tokens, dtype=np.uint32).tofile(path)
    manifest = {
        "splits": {
            split: {
                "tokens": tokens,
                "shards": [{"path": path.name, "tokens": tokens, "sha256": "unused"}],
            }
            for split in ("train", "validation", "test")
        }
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return PackedTokenCorpus(manifest_path, "train", SEQUENCE_LENGTH)


def one_step_global_positions(corpus: PackedTokenCorpus, world_size: int) -> list[int]:
    """按 train.py 的循环，返回一个 optimizer step 内全局看到的序列位置（升序）。"""
    accumulation = GLOBAL_BATCH // (LOCAL_BATCH * world_size)
    offset = 0
    positions: list[int] = []
    for _ in range(accumulation):
        for rank in range(world_size):
            start = offset + rank * LOCAL_BATCH
            for position in range(start, start + LOCAL_BATCH):
                positions.append(corpus.permuted_index(position, SEED))
        offset += LOCAL_BATCH * world_size
    return sorted(positions)


class WorldSizeEquivalenceTests(unittest.TestCase):
    def test_one_optimizer_step_sees_identical_global_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = build_corpus(Path(directory))
            try:
                self.assertEqual(len(corpus), TOTAL_SEQUENCES)
                reference = one_step_global_positions(corpus, world_size=1)
                self.assertEqual(len(reference), GLOBAL_BATCH)
                self.assertEqual(len(set(reference)), GLOBAL_BATCH)  # 一个 epoch 内不重复
                for world_size in (2, 4, 8):
                    self.assertEqual(
                        one_step_global_positions(corpus, world_size),
                        reference,
                        f"world_size={world_size} 的全局 batch 与单卡不一致",
                    )
            finally:
                corpus.close()

    def test_accumulated_gradient_matches_single_gpu(self) -> None:
        """每个 rank 的 loss 除以自己的 accumulation，DDP 平均后应等于单卡的全局梯度。"""

        def make_model() -> Qwen3ForCausalLM:
            torch.manual_seed(42)
            return Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=97,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=64,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            )

        torch.manual_seed(0)
        data = torch.randint(0, 97, (GLOBAL_BATCH, SEQUENCE_LENGTH + 1))
        accumulation = GLOBAL_BATCH // LOCAL_BATCH
        # train.py 在单卡时就是把全局 batch 顺序切成 accumulation 段
        micro_batches = [data[i * LOCAL_BATCH : (i + 1) * LOCAL_BATCH] for i in range(accumulation)]

        def accumulate(model, batches, scale: float) -> dict:
            model.zero_grad(set_to_none=True)
            for batch in batches:
                (model(batch, labels=batch, return_logits=False).loss * scale).backward()
            return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}

        reference = accumulate(make_model(), micro_batches, 1.0 / accumulation)

        # 2 卡：每个 rank 只拿自己那部分，loss 除以该 rank 的 accumulation，最后 DDP 平均
        per_rank_accumulation = GLOBAL_BATCH // 2 // LOCAL_BATCH
        rank_batches = [
            micro_batches[rank::2][:per_rank_accumulation] for rank in range(2)
        ]
        per_rank = [accumulate(make_model(), batches, 1.0 / per_rank_accumulation) for batches in rank_batches]
        averaged = {
            name: sum(per_rank[rank][name] for rank in range(2)) / 2 for name in per_rank[0]
        }
        for name, value in reference.items():
            torch.testing.assert_close(averaged[name], value, atol=1e-6, rtol=1e-5, msg=name)


if __name__ == "__main__":
    unittest.main()
