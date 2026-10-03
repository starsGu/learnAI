from __future__ import annotations

import fnmatch
import os
from pathlib import Path


TOKENIZER_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template*.jinja",
]

WEIGHT_IGNORE_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.onnx"]


def ensure_tokenizer(repo_id: str, target_dir: str | Path, hub: str = "huggingface"):
    """Download tokenizer metadata only and return a local tokenizer.

    The allow-list intentionally excludes model weight extensions.
    hub="modelscope" downloads the same files from modelscope.cn instead,
    with a direct connection that ignores proxy environment variables.
    """
    from transformers import AutoTokenizer

    target = Path(target_dir)
    local_override = os.environ.get("QWEN3_TOKENIZER_DIR")
    if local_override:
        override = Path(local_override)
        if (override / "tokenizer.json").exists():
            return AutoTokenizer.from_pretrained(override, local_files_only=True, use_fast=True)
    tokenizer_file = target / "tokenizer.json"
    if not tokenizer_file.exists():
        target.mkdir(parents=True, exist_ok=True)
        if hub == "modelscope":
            from .modelscope_hub import download_file, list_repo_files

            # 模型仓库在 ModelScope 走 repo/files 端点（repo_type="model"）。
            remote = list_repo_files(repo_id, repo_type="model")
            selected = [
                name
                for name in sorted(remote)
                if any(fnmatch.fnmatch(name, pattern) for pattern in TOKENIZER_ALLOW_PATTERNS)
                and not any(fnmatch.fnmatch(name, pattern) for pattern in WEIGHT_IGNORE_PATTERNS)
            ]
            if not selected:
                raise FileNotFoundError(
                    f"no tokenizer files in {repo_id} (modelscope) match {TOKENIZER_ALLOW_PATTERNS}"
                )
            for name in selected:
                download_file(
                    repo_id,
                    name,
                    target / name,
                    repo_type="model",
                    expected_size=remote[name],
                )
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=target,
                allow_patterns=TOKENIZER_ALLOW_PATTERNS,
                ignore_patterns=WEIGHT_IGNORE_PATTERNS,
            )
    return AutoTokenizer.from_pretrained(target, local_files_only=True, use_fast=True)
