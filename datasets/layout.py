from __future__ import annotations

from copy import deepcopy
from pathlib import Path


DEFAULT_DATA_ROOT = Path("D:/datasets/llm")


def _dataset_name(config: dict) -> str:
    """Return the stable directory name used for one prepared dataset."""
    if config.get("dataset_name"):
        return str(config["dataset_name"])
    if config.get("output_dir"):
        return Path(config["output_dir"]).name
    manifest = config.get("data_manifest")
    if manifest:
        return Path(manifest).parent.name
    raise KeyError("config must define dataset_name, output_dir, or data_manifest")


def configure_prepare_paths(config: dict, data_root: str | Path) -> dict:
    """Place downloads, tokenizer files, and packed tokens under one root."""
    configured = deepcopy(config)
    root = Path(data_root)
    configured["data_root"] = str(root)
    configured["raw_dir"] = str(root / "raw")
    configured["tokenizer_dir"] = str(root / "tokenizer" / "qwen3")
    configured["output_dir"] = str(root / "processed" / _dataset_name(configured))
    return configured


def configure_train_paths(config: dict, data_root: str | Path) -> dict:
    """Point training at data produced under ``data_root``."""
    configured = deepcopy(config)
    root = Path(data_root)
    configured["data_root"] = str(root)
    #MARK: 把  "data_manifest": "data/processed/chinese-10b/manifest.json" -> "root/autodl-tmp/datasets/processed/chinese-10b-interium/manifest.json"
    configured["data_manifest"] = str(
        root / "processed" / _dataset_name(configured) / "manifest.json"
    )
    configured["tokenizer_dir"] = str(root / "tokenizer" / "qwen3")
    return configured
