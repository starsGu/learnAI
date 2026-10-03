from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models import Qwen3Config, Qwen3ForCausalLM
from pretrain.checkpoint import (
    _align_cuda_rng_states,
    _restore_rng,
    enforce_checkpoint_window,
    load_training_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from pretrain.common import load_model_from_checkpoint, resolve_device
from pretrain.train import cosine_scheduler


class DummyTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
    )


class DeviceGuardTests(unittest.TestCase):
    """配置要求 CUDA 而 CUDA 不可用时必须报错，不能静默降级到 CPU。"""

    def test_device_cuda_raises_when_unavailable(self) -> None:
        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            with self.assertRaises(RuntimeError) as captured:
                resolve_device({"device": "cuda"}, 0)
            self.assertIn("配置要求 CUDA", str(captured.exception))
        finally:
            torch.cuda.is_available = original

    def test_device_cpu_is_explicit_and_allowed(self) -> None:
        original = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            self.assertEqual(resolve_device({"device": "cpu"}, 0).type, "cpu")
            self.assertEqual(resolve_device({}, 0).type, "cpu")  # auto 允许降级
        finally:
            torch.cuda.is_available = original

    def test_invalid_device_value(self) -> None:
        with self.assertRaises(ValueError):
            resolve_device({"device": "tpu"}, 0)


class CudaRngAlignmentTests(unittest.TestCase):
    """1 卡↔N 卡切换时，保存的 CUDA 随机状态数量必须对齐，否则 IndexError。"""

    def test_truncates_when_devices_decrease(self) -> None:
        saved = [b"gpu0", b"gpu1"]
        self.assertEqual(_align_cuda_rng_states(saved, 1), [b"gpu0"])   # 2 卡 checkpoint -> 1 卡
        self.assertEqual(_align_cuda_rng_states(saved, 2), saved)
        self.assertEqual(_align_cuda_rng_states(saved, 0), [])

    def test_pads_when_devices_increase(self) -> None:
        """1 卡 checkpoint 换多卡时，缺失的状态复制保存的第一份（不是当前设备状态）。"""
        saved = [b"gpu0"]
        self.assertEqual(_align_cuda_rng_states(saved, 3), [b"gpu0", b"gpu0", b"gpu0"])
        self.assertEqual(_align_cuda_rng_states([b"a", b"b"], 4), [b"a", b"b", b"a", b"a"])

    def test_empty_saved_states_falls_back_to_current_devices(self) -> None:
        original = torch.cuda.get_rng_state
        torch.cuda.get_rng_state = lambda index: f"default-{index}".encode()
        try:
            self.assertEqual(_align_cuda_rng_states([], 2), [b"default-0", b"default-1"])
        finally:
            torch.cuda.get_rng_state = original

    def test_restore_survives_checkpoint_from_more_gpus(self) -> None:
        """回归：2 卡 checkpoint 在 1 卡上恢复不能再抛 IndexError。"""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        saved = [torch.get_rng_state(), torch.get_rng_state()]  # 保存时可见 2 张卡
        calls: list[tuple[int, object]] = []

        class FakeDeviceRng:
            def __init__(self, index: int) -> None:
                self.index = index

            def set_state(self, item) -> None:
                calls.append((self.index, item))

        generators = [FakeDeviceRng(0), FakeDeviceRng(1)]
        original = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.set_rng_state_all)
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1  # 当前只有 1 张卡
        torch.cuda.set_rng_state_all = lambda states: [
            generators[index].set_state(item) for index, item in enumerate(states)
        ]
        try:
            _restore_rng({**state, "cuda": saved})
            self.assertEqual([index for index, _ in calls], [0])          # 只恢复 cuda:0
            self.assertTrue(torch.equal(calls[0][1], saved[0]))           # 恢复的是保存的那份
        finally:
            torch.cuda.is_available, torch.cuda.device_count, torch.cuda.set_rng_state_all = original


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model = Qwen3ForCausalLM(config())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = cosine_scheduler(optimizer, 1, 10, 0.1)
            tokens = torch.tensor([[3, 4, 5, 6]])
            model(tokens, labels=tokens, return_logits=False).loss.backward()
            optimizer.step()
            scheduler.step()
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

            state = {"step": 1, "tokens_seen": 3, "global_sequence_offset": 1, "best_validation_loss": None}
            saved = save_checkpoint(tmp_path, model, optimizer, scheduler, DummyTokenizer(), {"model_config": "unused"}, state, 0, 1)
            self.assertTrue(saved.exists())
            self.assertEqual((tmp_path / "latest.txt").read_text(encoding="utf-8"), "step-00000001")

            restored = Qwen3ForCausalLM(config())
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_scheduler = cosine_scheduler(restored_optimizer, 1, 10, 0.1)
            restored_state = load_training_checkpoint(tmp_path / "latest", restored, restored_optimizer, restored_scheduler, 0)
            self.assertEqual(restored_state, state)
            self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name])

            loaded, _, resolved = load_model_from_checkpoint(tmp_path / "latest", torch.device("cpu"))
            self.assertEqual(resolved, saved)
            torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits)

    def test_checkpoint_retention_trims_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(1, 6):
                step_dir = root / f"step-{step:08d}"
                step_dir.mkdir()
                for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng-rank-00000.pt"):
                    (step_dir / name).write_bytes(b"x")

            # 低句柄/低磁盘场景：只让最近 2 个保留优化器状态
            trimmed = prune_checkpoints(root, keep_recent=2)
            self.assertEqual(trimmed, [f"step-{step:08d}" for step in (1, 2, 3)])
            for step in (1, 2, 3):
                step_dir = root / f"step-{step:08d}"
                self.assertFalse((step_dir / "optimizer.pt").exists())
                self.assertFalse((step_dir / "scheduler.pt").exists())
                self.assertTrue((step_dir / "model.safetensors").exists())
            for step in (4, 5):
                self.assertTrue((root / f"step-{step:08d}" / "optimizer.pt").exists())

            # 裁剪过的 checkpoint 不允许用来"精确续训"
            with self.assertRaises(FileNotFoundError):
                load_training_checkpoint(root / "step-00000001", None, None, None, 0)

            # 目录数窗口：最旧的被整体删除，最新永不删除
            removed = enforce_checkpoint_window(root, keep_total=2)
            self.assertEqual(removed, ["step-00000001", "step-00000002", "step-00000003"])
            self.assertFalse((root / "step-00000001").exists())
            self.assertTrue((root / "step-00000005").exists())
            self.assertEqual(enforce_checkpoint_window(root, keep_total=2), [])

    def test_resume_rng_falls_back_to_rank_zero(self) -> None:
        """1 卡 checkpoint 改 2 卡续训：rank1 缺 RNG 文件必须回退，而不是报错。"""
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model = Qwen3ForCausalLM(config())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = cosine_scheduler(optimizer, 1, 10, 0.1)
            tokens = torch.tensor([[3, 4, 5, 6]])
            model(tokens, labels=tokens, return_logits=False).loss.backward()
            optimizer.step()
            state = {"step": 1, "tokens_seen": 3, "global_sequence_offset": 1, "best_validation_loss": None}
            saved = save_checkpoint(tmp_path, model, optimizer, scheduler, DummyTokenizer(), {"model_config": "unused"}, state, 0, 1)
            self.assertTrue((saved / "rng-rank-00000.pt").exists())

            def restore(rank: int) -> None:
                restored = Qwen3ForCausalLM(config())
                optimizer_ = torch.optim.AdamW(restored.parameters(), lr=1e-3)
                scheduler_ = cosine_scheduler(optimizer_, 1, 10, 0.1)
                load_training_checkpoint(saved, restored, optimizer_, scheduler_, rank)

            rank_zero_state = torch.get_rng_state()

            # 没有 rng-rank-00001.pt 时，标签页为 1 的进程回退到 rank0 的状态
            torch.manual_seed(999)  # 先污染当前状态，确保断言真的在验证恢复
            restore(1)
            self.assertTrue(torch.equal(torch.get_rng_state(), rank_zero_state))

            # 专属文件存在时必须优先使用，不能静默套用别的 rank
            own = torch.load(saved / "rng-rank-00000.pt", map_location="cpu", weights_only=False)
            own["torch"] = rank_zero_state.clone()
            own["torch"][0] = (int(own["torch"][0]) + 1) % 256  # 合法但不同的 MT19937 状态
            torch.save(own, saved / "rng-rank-00001.pt")
            torch.manual_seed(999)
            restore(1)
            self.assertTrue(torch.equal(torch.get_rng_state(), own["torch"]))

            # 同一 step 重复保存必须能覆盖旧目录（中断后重跑的常见情形）
            again = save_checkpoint(tmp_path, model, optimizer, scheduler, DummyTokenizer(), {"model_config": "unused"}, state, 0, 1)
            self.assertEqual(again, saved)


if __name__ == "__main__":
    unittest.main()
