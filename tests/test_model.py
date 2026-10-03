from __future__ import annotations

import unittest

import torch

from models import Qwen3Config, Qwen3ForCausalLM


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=97,
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


class ModelTests(unittest.TestCase):
    def test_parameter_names_and_weight_tying(self) -> None:
        model = Qwen3ForCausalLM(tiny_config())
        names = set(model.state_dict())
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("model.layers.0.self_attn.q_norm.weight", names)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", names)
        self.assertEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())

    def test_causal_mask_blocks_future_tokens(self) -> None:
        torch.manual_seed(1)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        first = torch.tensor([[3, 4, 5, 6]])
        second = torch.tensor([[3, 4, 90, 91]])
        with torch.no_grad():
            first_logits = model(first).logits
            second_logits = model(second).logits
        torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2], atol=1e-6, rtol=1e-5)

    def test_shifted_loss_matches_manual_cross_entropy(self) -> None:
        model = Qwen3ForCausalLM(tiny_config()).eval()
        tokens = torch.tensor([[3, 4, 5, 6, 7]])
        output = model(tokens, labels=tokens)
        manual = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, model.config.vocab_size).float(),
            tokens[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(output.loss, manual, atol=1e-6, rtol=1e-5)

    def test_sampling_is_reproducible(self) -> None:
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompt = torch.tensor([[3, 4, 5]])
        first = model.generate(prompt, max_new_tokens=5, seed=123)
        second = model.generate(prompt, max_new_tokens=5, seed=123)
        self.assertTrue(torch.equal(first, second))

    def test_tiny_model_can_overfit_one_batch(self) -> None:
        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=32,
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
        model = Qwen3ForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
        tokens = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]])
        initial_loss = float(model(tokens, labels=tokens, return_logits=False).loss.detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = model(tokens, labels=tokens, return_logits=False).loss
            loss.backward()
            optimizer.step()
        final_loss = float(model(tokens, labels=tokens, return_logits=False).loss.detach())
        self.assertLess(final_loss, initial_loss * 0.2)

    def test_matches_hugging_face_qwen3(self) -> None:
        from transformers import Qwen3Config as HFConfig
        from transformers import Qwen3ForCausalLM as HFModel

        config = tiny_config()
        custom = Qwen3ForCausalLM(config).eval()
        official = HFModel(HFConfig(**config.to_dict())).eval()
        official.load_state_dict(custom.state_dict(), strict=True)
        tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
        with torch.no_grad():
            custom_output = custom(tokens, labels=tokens)
            official_output = official(tokens, labels=tokens)
        torch.testing.assert_close(custom_output.logits, official_output.logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(custom_output.loss, official_output.loss, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
