from __future__ import annotations

"""fp32 主权重：bf16 只用于 autocast 前向，参数本身必须是 fp32。

历史 bug：build_model 用 model.to(bfloat16) 把参数也转成 bf16，导致 AdamW 的更新
在 bf16 里进行。bf16 在 1.0 附近分辨率约 3e-5，而学习率量级的更新约 3e-4，
RMSNorm 权重（初始值恒为 1.0）会永久冻结。实测 65536/65536 个元素零更新。
"""

import tempfile
import unittest
from pathlib import Path

import torch

from models import Qwen3Config, Qwen3ForCausalLM
from pretrain.common import (
    MasterWeightOptimizer,
    amp_context,
    build_model,
    build_optimizer,
    parameter_dtypes,
)
from pretrain.common import write_json


def tiny_config() -> dict:
    return {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


class MasterWeightTests(unittest.TestCase):
    def test_build_model_keeps_fp32_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            write_json(path, tiny_config())
            model = build_model(path, torch.device("cpu"), "bfloat16")
            dtypes = parameter_dtypes(model)
            keys = " ".join(dtypes)
            self.assertIn("float32", keys, f"参数应为 fp32 主权重: {dtypes}")
            self.assertNotIn("bfloat16", keys, f"参数不应是 bf16: {dtypes}")
            self.assertEqual(sum(dtypes.values()), model.num_parameters())

    def test_rmsnorm_weights_actually_update_with_bf16_forward(self) -> None:
        """回归：bf16 autocast 前向下训练，初始化为 1.0 的归一化权重必须发生变化。"""
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(**tiny_config()))
        norm_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "norm" in name
        }
        self.assertTrue(norm_before)
        for value in norm_before.values():
            self.assertTrue(torch.all(value == 1.0))  # RMSNorm 初始值恒为 1.0

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        torch.manual_seed(0)
        data = torch.randint(0, tiny_config()["vocab_size"], (4, 65))
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch.device("cpu")):
                loss = model(data, labels=data, return_logits=False).loss
            loss.backward()
            optimizer.step()

        moved = max(
            float((parameter.detach() - norm_before[name]).abs().max())
            for name, parameter in model.named_parameters()
            if name in norm_before
        )
        self.assertGreater(moved, 1e-4, f"归一化权重未更新（最大变化 {moved:.2e}）")


class LowPrecisionMasterWeightTests(unittest.TestCase):
    """master_weights=false：参数存 bf16 省显存，用 fp32 主权重副本保证更新精度。"""

    def _train(self, master_weights: bool, steps: int = 30):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            write_json(path, tiny_config())
            dtype = "bfloat16" if master_weights is False else "bfloat16"
            model = build_model(path, torch.device("cpu"), dtype)
            if master_weights is False:
                model = model.to(torch.bfloat16)
            optimizer = build_optimizer(model, torch.device("cpu"), {"learning_rate": 3e-4, "weight_decay": 0.1})
            before = {n: p.detach().float().clone() for n, p in model.named_parameters() if "norm" in n}
            torch.manual_seed(0)
            data = torch.randint(0, tiny_config()["vocab_size"], (4, 65))
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                with amp_context(torch.device("cpu")):
                    loss = model(data, labels=data, return_logits=False).loss
                loss.backward()
                optimizer.step()
            moved = max(
                float((p.detach().float() - before[n]).abs().max())
                for n, p in model.named_parameters() if n in before
            )
            dtypes = {str(p.dtype) for p in model.parameters()}
            return moved, dtypes

    def test_bf16_parameters_with_fp32_masters_still_update(self) -> None:
        moved, dtypes = self._train(master_weights=False)
        self.assertEqual(dtypes, {"torch.bfloat16"}, f"参数应为 bf16 以省显存: {dtypes}")
        self.assertGreater(moved, 1e-4, f"归一化权重未更新（最大变化 {moved:.2e}）")

    def test_plain_adamw_on_bf16_parameters_freezes_them(self) -> None:
        """反证：不加 fp32 主权重时，bf16 参数上的归一化权重会被舍入掉。"""
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(**tiny_config())).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        before = {n: p.detach().float().clone() for n, p in model.named_parameters() if "norm" in n}
        torch.manual_seed(0)
        data = torch.randint(0, tiny_config()["vocab_size"], (4, 65))
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch.device("cpu")):
                loss = model(data, labels=data, return_logits=False).loss
            loss.backward()
            optimizer.step()
        moved = max(
            float((p.detach().float() - before[n]).abs().max())
            for n, p in model.named_parameters() if n in before
        )
        self.assertEqual(moved, 0.0, "预期 bf16 参数被冻结（这正是需要 fp32 主权重的原因）")

    def test_master_optimizer_writes_back_and_clears_grads(self) -> None:
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(**tiny_config())).to(torch.bfloat16)
        optimizer = build_optimizer(model, torch.device("cpu"), {"learning_rate": 0.1, "weight_decay": 0.0})
        self.assertIsInstance(optimizer, MasterWeightOptimizer)
        data = torch.tensor([[3, 4, 5, 6]])
        with amp_context(torch.device("cpu")):
            loss = model(data, labels=data, return_logits=False).loss
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.parameters()), "step 后应清空参数梯度")
        self.assertTrue(all(master.grad is None for _, master in optimizer.masters), "step 后应清空主权重梯度")
        # bf16 参数确实被 fp32 主权重回写过
        parameter, master = optimizer.masters[0]
        self.assertTrue(torch.equal(parameter.detach().float(), master.detach().to(torch.bfloat16).float()), "bf16 参数应是主权重按 bf16 取整后的值")
        # state_dict 可存取（续训需要）
        optimizer.load_state_dict(optimizer.state_dict())


if __name__ == "__main__":
    unittest.main()


class SelectiveMasterWeightTests(unittest.TestCase):
    """只给归一化权重配 fp32 主权重：显存开销 0.25 MiB，且其余参数照常更新。"""

    def test_only_norm_weights_get_masters(self) -> None:
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(**tiny_config())).to(torch.bfloat16)
        optimizer = build_optimizer(model, torch.device("cpu"), {"learning_rate": 3e-4})
        self.assertIsInstance(optimizer, MasterWeightOptimizer)
        # 注意：named_parameters() 每次返回新的 Parameter 包装对象，必须按 id 比对
        ids = {id(parameter) for parameter, _ in optimizer.masters}
        master_names = {name for name, p in model.named_parameters() if id(p) in ids}
        # 主权重只覆盖 norm.weight
        self.assertTrue(master_names)
        self.assertTrue(all("norm" in n for n in master_names), master_names)
        # 显存开销：主权重元素数应远小于参数量
        master_elements = sum(m.numel() for _, m in optimizer.masters)
        self.assertLess(master_elements, model.num_parameters() * 0.01)

    def test_non_norm_parameters_still_update(self) -> None:
        """漏掉 Linear/Embedding 就会让它们完全不训练——必须验证。"""
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(**tiny_config())).to(torch.bfloat16)
        optimizer = build_optimizer(model, torch.device("cpu"), {"learning_rate": 3e-3})
        before = {
            n: p.detach().float().clone()
            for n, p in model.named_parameters()
            if "q_proj" in n or "embed_tokens" in n
        }
        torch.manual_seed(0)
        data = torch.randint(0, tiny_config()["vocab_size"], (4, 65))
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch.device("cpu")):
                loss = model(data, labels=data, return_logits=False).loss
            loss.backward()
            optimizer.step()
        moved = {
            n: float((p.detach().float() - before[n]).abs().max())
            for n, p in model.named_parameters() if n in before
        }
        for name, delta in moved.items():
            self.assertGreater(delta, 0.0, f"{name} 完全没有更新（优化器漏了它）")


class OomRetryTests(unittest.TestCase):
    """OOM 时自动把 micro_batch 减半、梯度累积翻倍，保证全局 batch 不变。"""

    def _run_main(self, base_micro: int, fail_times: int, retries: int = 2):
        import json as _json
        import sys as _sys
        import tempfile as _tempfile
        from pathlib import Path as _Path

        import pretrain.train as train_module
        from pretrain.common import write_json as _write_json

        with _tempfile.TemporaryDirectory() as directory:
            root = _Path(directory)
            config_path = root / "train.json"
            _write_json(
                config_path,
                {
                    "dataset_name": "probe",
                    "num_gpus": 1,
                    "model_config": "unused.json",
                    "data_manifest": "unused.json",
                    "tokenizer_repo": "unused",
                    "tokenizer_dir": "unused",
                    "output_dir": str(root / "out"),
                    "sequence_length": 2048,
                    "micro_batch_size": base_micro,
                    "global_tokens_per_step": 524288,
                    "target_tokens": 524288,
                },
            )
            attempts: list[tuple[int, int]] = []

            def fake_train(config):
                attempts.append((config["micro_batch_size"], config["accumulation_steps"]))
                if len(attempts) <= fail_times:
                    raise torch.cuda.OutOfMemoryError("simulated OOM")
                return {"step": 1}

            original_train = train_module.train
            original_argv = _sys.argv
            train_module.train = fake_train
            _sys.argv = [
                "train",
                "--config", str(config_path),
                "--data", str(root),
                "--retry-on-oom", str(retries),
            ]
            try:
                train_module.main()
            finally:
                train_module.train = original_train
                _sys.argv = original_argv
            return attempts

    def test_halves_micro_batch_and_doubles_accumulation(self) -> None:
        attempts = self._run_main(base_micro=16, fail_times=1, retries=2)
        self.assertEqual(attempts, [(16, 16), (8, 32)])
        # 每次重试后，micro_batch * 梯度累积 恒等于全局 batch 的 256 条序列
        for micro, accumulation in attempts:
            self.assertEqual(micro * accumulation, 256)

    def test_no_retry_when_disabled(self) -> None:
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self._run_main(base_micro=16, fail_times=1, retries=0)

    def test_gives_up_after_retry_budget(self) -> None:
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self._run_main(base_micro=16, fail_times=5, retries=1)


class LossChunkSizeTests(unittest.TestCase):
    """loss_chunk_size 只影响显存峰值：loss 值必须（在 f32 舍入内）不变。"""

    def test_loss_is_invariant_to_chunk_size(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=512, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
            )
        ).eval()
        data = torch.randint(0, 512, (3, 65))
        with torch.no_grad():
            losses = {
                chunk: float(model(data, labels=data, return_logits=False, loss_chunk_size=chunk).loss)
                for chunk in (16, 64, 128, 1024)
            }
        values = list(losses.values())
        reference = values[-1]
        for chunk, value in losses.items():
            # 累加顺序不同只带来 f32 舍入差（实测相对误差约 8e-8）
            self.assertAlmostEqual(value, reference, delta=abs(reference) * 1e-6,
                                   msg=f"chunk_size={chunk} 的 loss 偏离过大: {losses}")


class MixedDtypeOptimizerTests(unittest.TestCase):
    """回归：混合 dtype 的参数组不能在 fused 内核下生效。

    GPU 上曾报：params, grads, exp_avgs, exp_avg_sqs must have same dtype。
    根因是 fp32 主权重 + bf16 直更参数共处一个参数组，而 fused AdamW 要求整组同 dtype。
    """

    def _build(self):
        import tempfile
        from pathlib import Path
        from pretrain.common import write_json as _write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            _write_json(path, tiny_config())
            model = build_model(path, torch.device("cpu"), "bfloat16", low_precision=True)
            optimizer = build_optimizer(model, torch.device("cpu"), {"learning_rate": 3e-4})
            return model, optimizer

    def test_fused_is_disabled_for_mixed_dtypes(self) -> None:
        _, optimizer = self._build()
        self.assertIsInstance(optimizer, MasterWeightOptimizer)
        self.assertFalse(bool(optimizer.inner.defaults.get("fused")), "混合 dtype 时不能开 fused")

    def test_param_group_really_mixes_dtypes(self) -> None:
        _, optimizer = self._build()
        dtypes = {p.dtype for group in optimizer.inner.param_groups for p in group["params"]}
        self.assertIn(torch.float32, dtypes, "主权重应为 fp32")
        self.assertIn(torch.bfloat16, dtypes, "直更参数应为 bf16")

    def test_step_runs_and_updates_both_kinds(self) -> None:
        model, optimizer = self._build()
        norm0 = {n: p.detach().float().clone() for n, p in model.named_parameters() if "norm" in n}
        lin0 = {n: p.detach().float().clone() for n, p in model.named_parameters() if "q_proj" in n}
        torch.manual_seed(0)
        data = torch.randint(0, tiny_config()["vocab_size"], (4, 65))
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(torch.device("cpu")):
                loss = model(data, labels=data, return_logits=False, loss_chunk_size=32).loss
            loss.backward()
            optimizer.step()  # 曾在此抛 dtype RuntimeError
        norm_move = max(float((p.detach().float() - norm0[n]).abs().max()) for n, p in model.named_parameters() if n in norm0)
        lin_move = max(float((p.detach().float() - lin0[n]).abs().max()) for n, p in model.named_parameters() if n in lin0)
        self.assertGreater(norm_move, 1e-5, "归一化权重未更新")
        self.assertGreater(lin_move, 1e-5, "Linear 权重未更新")

    def test_optimizer_state_round_trip(self) -> None:
        """checkpoint 续训依赖它：状态必须能存能读、且读后继续更新。"""
        import tempfile
        from pathlib import Path
        import torch as _torch
        from pretrain.checkpoint import save_checkpoint, load_training_checkpoint
        from pretrain.common import write_json as _write_json
        from pretrain.train import cosine_scheduler

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.json"
            _write_json(path, tiny_config())
            model = build_model(path, _torch.device("cpu"), "bfloat16", low_precision=True)
            optimizer = build_optimizer(model, _torch.device("cpu"), {"learning_rate": 3e-4})
            scheduler = cosine_scheduler(optimizer, 1, 10, 0.1)
            state = {"step": 1, "tokens_seen": 3, "global_sequence_offset": 1, "best_validation_loss": None}
            class _Tokenizer:
                def save_pretrained(self, path):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            saved = save_checkpoint(root, model, optimizer, scheduler, _Tokenizer(),
                                    {"model_config": "unused"}, state, 0, 1)

            restored_model = build_model(path, _torch.device("cpu"), "bfloat16", low_precision=True)
            restored_optimizer = build_optimizer(restored_model, _torch.device("cpu"), {"learning_rate": 3e-4})
            restored_scheduler = cosine_scheduler(restored_optimizer, 1, 10, 0.1)
            restored_state = load_training_checkpoint(saved, restored_model, restored_optimizer,
                                                      restored_scheduler, 0)
            self.assertEqual(restored_state["step"], 1)
            self.assertEqual(len(restored_optimizer.inner.state), len(optimizer.inner.state),
                             "优化器状态数量应一致")
