from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from .common import choose_device, load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a Chinese prompt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    device = choose_device(0)
    model, _, checkpoint_dir = load_model_from_checkpoint(args.checkpoint, device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
