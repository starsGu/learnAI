from __future__ import annotations

import argparse
import json
import math

import torch

from datasets.layout import DEFAULT_DATA_ROOT, configure_train_paths
from datasets.tokenizer import ensure_tokenizer

from .checkpoint import load_training_checkpoint
from .common import (
    amp_context,
    build_model,
    choose_device,
    read_json,
)
from .train import cosine_scheduler


def build_optimizer_scheduler(model, config):
    """
    load_training_checkpoint 是给训练恢复写的，
    所以需要 optimizer / scheduler。
    推理本身不会使用它们。
    """
    optimizer_kwargs = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(config.get("betas", [0.9, 0.95])),
        "eps": float(config.get("adam_epsilon", 1e-8)),
        "weight_decay": float(config.get("weight_decay", 0.1)),
    }

    if next(model.parameters()).device.type == "cuda":
        optimizer_kwargs["fused"] = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_kwargs,
    )

    requested_global_tokens = int(config["global_tokens_per_step"])
    target_tokens = int(config["target_tokens"])

    total_steps = math.ceil(
        target_tokens / requested_global_tokens
    )

    warmup_steps = max(
        1,
        math.ceil(
            total_steps * float(config.get("warmup_ratio", 0.01))
        ),
    )

    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        float(config.get("minimum_learning_rate_ratio", 0.1)),
    )

    return optimizer, scheduler


def sample_next_token(logits, temperature=0.8, top_p=0.95):
    # greedy
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top-p
    sorted_probs, sorted_indices = torch.sort(
        probs,
        descending=True,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    sampled_index = torch.multinomial(sorted_probs, num_samples=1)

    return torch.gather(
        sorted_indices,
        -1,
        sampled_index,
    )

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    seed=42,
    repetition_penalty=1.0,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    # 改动：完全按照自定义 Qwen3ForCausalLM.generate() 的参数调用
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        seed=seed,
        repetition_penalty=repetition_penalty,
    )

    # 只取新生成内容
    new_tokens = output_ids[:, input_ids.shape[1]:]

    return tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True,
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint目录，或者 latest 指针",
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_ROOT),
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="大于 1 时抑制已出现过的 token；基座模型贪心解码容易重复，建议 1.1~1.3",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. 读取和训练完全相同的 config
    # --------------------------------------------------
    config = configure_train_paths(
        read_json(args.config),
        args.data,
    )

    # --------------------------------------------------
    # 2. device
    # --------------------------------------------------
    device = choose_device(0)

    print("device:", device)

    # --------------------------------------------------
    # 3. 创建和训练时完全相同的模型
    # --------------------------------------------------
    # 与训练一致的参数精度：master_weights=false 时参数存 bf16
    parameter_dtype = "bfloat16" if config.get("master_weights", True) is False else config.get("dtype", "bfloat16")
    model = build_model(
        config["model_config"],
        device,
        parameter_dtype,
    )

    # --------------------------------------------------
    # 4. 为训练 checkpoint loader 创建占位 optimizer
    # --------------------------------------------------
    optimizer, scheduler = build_optimizer_scheduler(
        model,
        config,
    )

    # --------------------------------------------------
    # 5. 加载 checkpoint
    # --------------------------------------------------
    state = load_training_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        scheduler,
        0,  # rank
    )

    print("checkpoint loaded:")
    print(json.dumps(state, indent=2, ensure_ascii=False))

    # optimizer / scheduler 推理不再需要
    del optimizer
    del scheduler

    model.eval()

    # --------------------------------------------------
    # 6. tokenizer
    # --------------------------------------------------
    tokenizer = ensure_tokenizer(
        config["tokenizer_repo"],
        config["tokenizer_dir"],
    )

    # --------------------------------------------------
    # 7. inference
    # --------------------------------------------------
    with amp_context(device):
        answer = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    print("\n========== OUTPUT ==========")
    print(answer)


if __name__ == "__main__":
    main()
    
'''
python -m pretrain.infer \
  --config configs/formal/train.json \
  --data /root/autodl-tmp/datasets \
  --checkpoint checkpoints/pretrain/qwen3-0.6B/step-00002000 \
  --prompt "招商银行卡限额" \
  --max-new-tokens 256
'''
