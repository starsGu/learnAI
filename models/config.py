from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Qwen3Config:
    vocab_size: int = 151_936
    hidden_size: int = 1_024
    intermediate_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32_768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    bos_token_id: int = 151_643
    eos_token_id: int = 151_643
    tie_word_embeddings: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if min(self.vocab_size, self.hidden_size, self.num_hidden_layers, self.head_dim) <= 0:
            raise ValueError("model dimensions must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        known = {field for field in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": False,
                "hidden_act": "silu",
                "model_type": "qwen3",
                "sliding_window": None,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_sliding_window": False,
            }
        )
        return values

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
