from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers import Qwen3Config as HFQwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from .common import load_model_from_checkpoint


def export(checkpoint: str, output: str) -> Path:
    custom_model, _, checkpoint_dir = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
    hf_config = HFQwen3Config(**custom_model.config.to_dict())
    hf_model = HFQwen3ForCausalLM(hf_config)
    missing, unexpected = hf_model.load_state_dict(custom_model.state_dict(), strict=False)
    allowed_missing = {"lm_head.weight"} if custom_model.config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"incompatible state dict; missing={missing}, unexpected={unexpected}")
    hf_model.tie_weights()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer.save_pretrained(output_dir)
    GenerationConfig(
        bos_token_id=custom_model.config.bos_token_id,
        eos_token_id=custom_model.config.eos_token_id,
        pad_token_id=custom_model.config.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
    ).save_pretrained(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a training checkpoint in Hugging Face format")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
