from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import Qwen3Config


@dataclass
class CausalLMOutput:
    loss: Tensor | None
    logits: Tensor | None


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * values.to(input_dtype)


def rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        frequencies = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", frequencies, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = position_ids.float()
        frequencies = torch.einsum("bi,j->bij", positions, self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype).unsqueeze(1), embedding.sin().to(dtype).unsqueeze(1)


def apply_rotary_pos_emb(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        )

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return self.o_proj(attended)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(self, input_ids: Tensor, gradient_checkpointing: bool = False) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        positions = positions.expand(input_ids.shape[0], -1)
        cos, sin = self.rotary_emb(positions, hidden_states.dtype)

        for layer in self.layers:
            if gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, cos, sin, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 decoder with Hugging Face-compatible parameter names.

    This class intentionally contains no code that loads pretrained model weights.
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _chunked_loss(self, hidden_states: Tensor, labels: Tensor, chunk_size: int) -> Tensor:
        shifted_hidden = hidden_states[:, :-1].contiguous().view(-1, hidden_states.shape[-1])
        shifted_labels = labels[:, 1:].contiguous().view(-1)
        total = hidden_states.new_zeros((), dtype=torch.float32)
        count = 0
        for start in range(0, shifted_labels.numel(), chunk_size):
            chunk_labels = shifted_labels[start : start + chunk_size]
            valid = int((chunk_labels != -100).sum().item())
            if not valid:
                continue
            # 不要在这里 .float()：那会让 logits 以 fp32 常驻显存（backward 时被 autograd 保留），
            # 内存翻倍。F.cross_entropy 内部本来就会升到 fp32 计算，实测两者 loss 完全相同。
            logits = self.lm_head(shifted_hidden[start : start + chunk_size])
            total = total + F.cross_entropy(logits, chunk_labels, ignore_index=-100, reduction="sum")
            count += valid
        if count == 0:
            raise ValueError("labels contain no trainable tokens")
        return total / count

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        return_logits: bool = True,
        loss_chunk_size: int = 1_024,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids, self.gradient_checkpointing)
        loss = self._chunked_loss(hidden_states, labels, loss_chunk_size) if labels is not None else None
        logits = self.lm_head(hidden_states).float() if return_logits else None
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: int | None = None,
        seed: int = 42,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        generated = input_ids
        generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_position_embeddings :]
            logits = self(context).logits[:, -1]
            if repetition_penalty != 1.0:
                # 已出现过的 token 按符号缩放：正 logits 除以惩罚系数，负 logits 乘惩罚系数，
                # 两者都使其概率下降，可有效打断 "1.1.1.1" 这类自我强化的重复。
                seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, generated, True)
                logits = torch.where(
                    seen,
                    torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty),
                    logits,
                )
            if not do_sample or temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                scores = logits / temperature
                if 0 < top_k < scores.shape[-1]:
                    cutoff = torch.topk(scores, top_k).values[:, -1, None]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                if 0 < top_p < 1:
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                    probabilities = F.softmax(sorted_scores, dim=-1)
                    remove = probabilities.cumsum(dim=-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                    scores = torch.full_like(scores, float("-inf")).scatter(
                        1, sorted_indices, sorted_scores
                    )
                next_token = torch.multinomial(
                    F.softmax(scores, dim=-1), num_samples=1, generator=generator
                )
            generated = torch.cat((generated, next_token), dim=1)
            if generated.shape[0] == 1 and int(next_token.item()) == eos:
                break
        return generated

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
