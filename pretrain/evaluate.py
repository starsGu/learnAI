from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch

from datasets import PackedTokenCorpus
from .common import amp_context, choose_device, load_model_from_checkpoint, read_json, write_json
from .prompts import get_prompts
from .train import VALIDATION_SEED


def generation_anomalies(text: str, hit_limit: bool) -> dict[str, bool]:
    compact = re.sub(r"\s+", "", text)
    ngrams = [compact[index : index + 4] for index in range(max(0, len(compact) - 3))]
    repeated = bool(ngrams) and 1.0 - len(set(ngrams)) / len(ngrams) > 0.35
    chinese_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    return {
        "replacement_character": "�" in text,
        "repeated_4gram": repeated,
        "too_short": chinese_count < 10,
        "hit_max_length": hit_limit,
    }


@torch.no_grad()
def corpus_loss(model, corpus: PackedTokenCorpus, batches: int, batch_size: int, device) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for index in range(batches):
        tokens = corpus.batch(index * batch_size, batch_size, 0, 1, VALIDATION_SEED, device)
        with amp_context(device):
            loss = model(tokens, labels=tokens, return_logits=False).loss
        token_count = batch_size * corpus.sequence_length
        total_loss += float(loss) * token_count
        total_tokens += token_count
    return total_loss / total_tokens


def evaluate(checkpoint: str, config: dict) -> Path:
    device = choose_device(0)
    model, train_config, checkpoint_dir = load_model_from_checkpoint(checkpoint, device)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    manifest = config.get("data_manifest", train_config["data_manifest"])
    sequence_length = int(config.get("sequence_length", train_config["sequence_length"]))
    validation = PackedTokenCorpus(manifest, "validation", sequence_length)
    test = PackedTokenCorpus(manifest, "test", sequence_length)
    validation_loss = corpus_loss(model, validation, int(config["loss_batches"]), int(config["batch_size"]), device)
    test_loss = corpus_loss(model, test, int(config["loss_batches"]), int(config["batch_size"]), device)

    prompts = get_prompts(config.get("prompt_split", "dev"))
    limit = int(config.get("prompt_limit", 0))
    if limit:
        prompts = prompts[:limit]
    rows = []
    anomaly_counts = {name: 0 for name in ("replacement_character", "repeated_4gram", "too_short", "hit_max_length")}
    max_new_tokens = int(config.get("max_new_tokens", 64))
    for item in prompts:
        encoded = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with amp_context(device):
            generated = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=float(config.get("temperature", 0.8)),
                top_k=int(config.get("top_k", 40)),
                top_p=float(config.get("top_p", 0.95)),
                do_sample=bool(config.get("do_sample", True)),
                seed=int(config.get("seed", 42)),
                repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            )
        new_tokens = generated[0, encoded.shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        anomalies = generation_anomalies(text, len(new_tokens) >= max_new_tokens)
        for name, present in anomalies.items():
            anomaly_counts[name] += int(present)
        rows.append({**item, "completion": text, "anomalies": anomalies})

    output_dir = Path(config.get("output_dir", checkpoint_dir / "evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(checkpoint_dir.resolve()),
        "validation_loss": validation_loss,
        "validation_perplexity": math.exp(min(validation_loss, 20.0)),
        "test_loss": test_loss,
        "test_perplexity": math.exp(min(test_loss, 20.0)),
        "test_validation_gap_ratio": abs(test_loss - validation_loss) / validation_loss,
        "prompt_count": len(rows),
        "anomaly_counts": anomaly_counts,
        "anomaly_any_rate": sum(any(row["anomalies"].values()) for row in rows) / max(len(rows), 1),
        "generations": rows,
    }
    write_json(output_dir / "evaluation.json", result)
    with (output_dir / "human_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "prompt", "completion", "grammar", "relevance", "consistency", "completeness", "logic", "notes"])
        for row in rows:
            writer.writerow([row["id"], row["category"], row["prompt"], row["completion"], "", "", "", "", "", ""])
    return output_dir / "evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate loss and fixed Chinese continuations")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(evaluate(args.checkpoint, read_json(args.config)))


if __name__ == "__main__":
    main()
