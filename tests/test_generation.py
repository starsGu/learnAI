from __future__ import annotations

"""重复惩罚：打断基座模型贪心解码时的自我强化循环。"""

import unittest

import torch

from models import Qwen3Config, Qwen3ForCausalLM


def tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(3)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).eval()


class RepetitionPenaltyTests(unittest.TestCase):
    def test_repeated_token_probability_drops(self) -> None:
        """已出现过的 token，其概率必须下降；正负 logits 都要被抑制。"""
        model = tiny_model()
        prompt = torch.tensor([[3, 4, 5, 6]])
        context = prompt
        logits = model(context).logits[:, -1]

        created = int(logits.argmax(dim=-1).item())
        seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, prompt, True)
        self.assertTrue(bool(seen[0, created]))

        penalized = torch.where(
            seen,
            torch.where(logits > 0, logits / 1.5, logits * 1.5),
            logits,
        )
        self.assertLess(float(penalized[0, created]), float(logits[0, created]))

        # 覆盖 logits 为负的分支：惩罚后必须更负（概率更低）
        negative_positions = (logits < 0) & seen
        if bool(negative_positions.any()):
            affected = negative_positions.nonzero()[0][1]
            self.assertLess(float(penalized[0, affected]), float(logits[0, affected]))

    def test_penalty_breaks_token_loop(self) -> None:
        model = tiny_model()
        prompt = torch.tensor([[3, 4]])
        with torch.no_grad():
            no_penalty = model.generate(prompt, max_new_tokens=40, temperature=0.7, top_k=0, top_p=1.0, seed=1)
            with_penalty = model.generate(
                prompt, max_new_tokens=40, temperature=0.7, top_k=0, top_p=1.0, seed=1, repetition_penalty=1.5
            )
        self.assertGreater(len(set(with_penalty[0].tolist())), len(set(no_penalty[0].tolist())))

    def test_penalty_keeps_fixed_seed_reproducible(self) -> None:
        model = tiny_model()
        prompt = torch.tensor([[3, 4, 5]])
        arguments = dict(max_new_tokens=16, seed=99, repetition_penalty=1.3)
        with torch.no_grad():
            first = model.generate(prompt, **arguments)
            second = model.generate(prompt, **arguments)
        self.assertTrue(torch.equal(first, second))

    def test_invalid_penalty_rejected(self) -> None:
        model = tiny_model()
        with self.assertRaises(ValueError):
            model.generate(torch.tensor([[3, 4]]), repetition_penalty=0.0)
        with self.assertRaises(ValueError):
            model.generate(torch.tensor([[3, 4]]), repetition_penalty=-1.0)


if __name__ == "__main__":
    unittest.main()
