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
