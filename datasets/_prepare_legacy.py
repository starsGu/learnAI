from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from .layout import DEFAULT_DATA_ROOT, configure_prepare_paths
from .modelscope_hub import download_file as modelscope_download_file
from .modelscope_hub import list_repo_files as modelscope_list_repo_files
from .tokenizer import ensure_tokenizer


WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".onnx"}


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def chinese_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in visible)
    return chinese / len(visible)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DedupIndex:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS hashes (kind TEXT, digest TEXT, PRIMARY KEY(kind, digest))")
        self.pending = 0

    def add(self, kind: str, digest: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO hashes(kind, digest) VALUES (?, ?)", (kind, digest)
        )
        self.pending += 1
        if self.pending >= 10_000:
            self.commit()
        return cursor.rowcount == 1

    def commit(self) -> None:
        self.connection.commit()
        self.pending = 0

    def close(self) -> None:
        self.commit()
        self.connection.close()


def iter_jsonl(path: Path, text_field: str) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = value.get(text_field)
            if isinstance(text, str):
                yield text


def iter_parquet(path: Path, text_field: str) -> Iterator[str]:
    import pyarrow.parquet as parquet

    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(columns=[text_field], batch_size=2_048):
        for text in batch.column(0).to_pylist():
            if isinstance(text, str):
                yield text


def iter_texts(path: Path, text_field: str) -> Iterator[str]:
    if path.suffix == ".parquet":
        yield from iter_parquet(path, text_field)
    elif path.suffix in {".jsonl", ".json", ".gz"} or path.name.endswith(".jsonl.gz") or path.name.startswith("part_"):
        yield from iter_jsonl(path, text_field)


def select_remote_files(names, patterns: list[str]) -> list[str]:
    """按 allow_patterns 选取远程文件路径，并排除模型权重文件。"""
    return [
        name
        for name in sorted(names)
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        and Path(name).suffix not in WEIGHT_EXTENSIONS
    ]


def source_files(source: dict, raw_dir: Path, hub: str = "huggingface") -> Iterator[Path]:
    if source["type"] == "local":
        base = Path(source["path"])
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        yield from (path for path in files if path.is_file() and path.suffix not in WEIGHT_EXTENSIONS)
    elif source["type"] in {"huggingface", "modelscope"}:
        # --modelscope（或 source.type=modelscope）时改从 modelscope.cn 直连下载，
        # 其余行为（目录布局、allow_patterns、权重过滤）与 Hugging Face 完全一致。
        use_modelscope = hub == "modelscope" or source["type"] == "modelscope"
        target = raw_dir / source["name"]
        patterns = source.get("allow_patterns", ["*"])
        repo_id = source["repo_id"]
        if use_modelscope:
            remote = modelscope_list_repo_files(repo_id, repo_type="dataset")
            selected = select_remote_files(remote, patterns)
            if not selected:
                raise FileNotFoundError(
                    f"no files in {repo_id} (modelscope) match allow_patterns={patterns}; "
                    f"repo has {len(remote)} files, e.g. {sorted(remote)[:3]}"
                )
            for name in selected:
                yield modelscope_download_file(
                    repo_id,
                    name,
                    target / name,
                    repo_type="dataset",
                    expected_size=remote[name],
                )
        else:
            from huggingface_hub import hf_hub_download, list_repo_files

            selected = select_remote_files(
                list_repo_files(repo_id, repo_type="dataset"), patterns
            )
            if not selected:
                raise FileNotFoundError(
                    f"no files in {repo_id} match allow_patterns={patterns}"
                )
            for name in selected:
                yield Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=name,
                        local_dir=target,
                    )
                )
    else:
        raise ValueError(f"unknown source type: {source['type']}")


class ShardWriter:
    def __init__(self, output_dir: Path, shard_tokens: int) -> None:
        self.output_dir = output_dir
        self.shard_tokens = shard_tokens
        self.buffers = {split: [] for split in ("train", "validation", "test")}
        self.records = {split: [] for split in self.buffers}
        self.totals = Counter()
        self._recover_existing_shards()

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _recover_existing_shards(self) -> None:
        for split in self.buffers:
            split_dir = self.output_dir / split
            for path in sorted(split_dir.glob("shard-*.bin")):
                tokens = path.stat().st_size // np.dtype(np.uint32).itemsize
                self.records[split].append(
                    {
                        "path": path.relative_to(self.output_dir).as_posix(),
                        "tokens": tokens,
                        "sha256": self._checksum(path),
                    }
                )
                self.totals[split] += tokens

    def add(self, split: str, tokens: list[int]) -> None:
        self.buffers[split].extend(tokens)
        self.totals[split] += len(tokens)
        while len(self.buffers[split]) >= self.shard_tokens:
            self._flush(split, self.shard_tokens)

    def _flush(self, split: str, count: int | None = None) -> None:
        buffer = self.buffers[split]
        count = len(buffer) if count is None else count
        if not count:
            return
        values = np.asarray(buffer[:count], dtype=np.uint32)
        del buffer[:count]
        split_dir = self.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"shard-{len(self.records[split]):05d}.bin"
        values.tofile(path)
        checksum = self._checksum(path)
        self.records[split].append(
            {
                "path": path.relative_to(self.output_dir).as_posix(),
                "tokens": int(len(values)),
                "sha256": checksum,
            }
        )

    def finish(self) -> None:
        for split in self.buffers:
            self._flush(split)


def choose_split(digest: str, validation_bps: int, test_bps: int) -> str:
    bucket = int(digest[:8], 16) % 10_000
    if bucket < test_bps:
        return "test"
    if bucket < test_bps + validation_bps:
        return "validation"
    return "train"


def prepare(config: dict, hub: str = "huggingface") -> Path:
    """hub="modelscope" 时，分词器和 huggingface 型数据源都改从 modelscope.cn 下载。"""
    output_dir = Path(config["output_dir"])
    raw_dir = Path(config.get("raw_dir", "data/raw"))
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{output_dir} is already prepared; choose another output_dir or remove it explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ensure_tokenizer(config["tokenizer_repo"], config["tokenizer_dir"], hub=hub)
    # Base-model pretraining uses <|endoftext|> (151643) between documents.
    # An instruct tokenizer may expose <|im_end|> as eos, so keep this explicit.
    eos_id = int(config.get("document_separator_token_id", 151_643))
    writer = ShardWriter(output_dir, int(config["shard_tokens"]))
    dedup = DedupIndex(output_dir / "dedup.sqlite3")
    state_path = output_dir / "processing_state.json"
    saved_state = load_config(state_path) if state_path.exists() else {}
    stats = Counter(saved_state.get("stats", {}))
    by_source: dict[str, Counter] = {
        name: Counter(values) for name, values in saved_state.get("sources", {}).items()
    }
    completed_files = set(saved_state.get("completed_files", []))
    targets = {
        "train": int(config["target_train_tokens"]),
        "validation": int(config["target_validation_tokens"]),
        "test": int(config["target_test_tokens"]),
    }

    def persist_state() -> None:
        state = {
            "completed_files": sorted(completed_files),
            "stats": dict(stats),
            "sources": {name: dict(values) for name, values in by_source.items()},
            "split_tokens_on_disk": dict(writer.totals),
        }
        with state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)

    try:
        for source in config["sources"]:
            source_stats = by_source.setdefault(source["name"], Counter())
            source_limit = int(source.get("target_tokens", targets["train"]))
            source_train_tokens = int(source_stats.get("train_tokens", 0))
            for path in source_files(source, raw_dir, hub):
                file_id = str(path.resolve())
                if file_id in completed_files:
                    continue
                for raw_text in tqdm(iter_texts(path, source.get("text_field", "text")), desc=source["name"]):
                    stats["documents_seen"] += 1
                    source_stats["documents_seen"] += 1
                    text = normalize_text(raw_text)
                    if len(text) < int(config.get("min_chars", 40)):
                        stats["filtered_short"] += 1
                        continue
                    if chinese_ratio(text) < float(config.get("min_chinese_ratio", 0.2)):
                        stats["filtered_language"] += 1
                        continue
                    document_hash = digest_text(text)
                    if not dedup.add("document", document_hash):
                        stats["filtered_duplicate_document"] += 1
                        continue

                    paragraphs = []
                    for paragraph in text.split("\n"):
                        if len(paragraph) < int(config.get("min_paragraph_chars", 16)):
                            continue
                        if dedup.add("paragraph", digest_text(paragraph)):
                            paragraphs.append(paragraph)
                        else:
                            stats["filtered_duplicate_paragraph"] += 1
                    text = "\n".join(paragraphs)
                    if len(text) < int(config.get("min_chars", 40)):
                        continue
                    split = choose_split(
                        document_hash,
                        int(config["validation_basis_points"]),
                        int(config["test_basis_points"]),
                    )
                    if writer.totals[split] >= targets[split]:
                        continue
                    tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
                    remaining = targets[split] - writer.totals[split]
                    tokens = tokens[:remaining]
                    writer.add(split, tokens)
                    stats[f"accepted_{split}_documents"] += 1
                    source_stats[f"{split}_tokens"] += len(tokens)
                    if split == "train":
                        source_train_tokens += len(tokens)
                    if all(writer.totals[name] >= target for name, target in targets.items()):
                        break
                    if source_train_tokens >= source_limit:
                        break
                writer.finish()
                completed_files.add(file_id)
                dedup.commit()
                persist_state()
                if all(writer.totals[name] >= target for name, target in targets.items()) or source_train_tokens >= source_limit:
                    break
    finally:
        writer.finish()
        persist_state()
        dedup.close()

    manifest = {
        "version": 1,
        "tokenizer_repo": config["tokenizer_repo"],
        "eos_token_id": eos_id,
        "config": config,
        "stats": dict(stats),
        "sources": {name: dict(values) for name, values in by_source.items()},
        "splits": {
            split: {"tokens": int(writer.totals[split]), "shards": writer.records[split]}
            for split in writer.records
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic packed-token shards")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_ROOT),
        help="Root for raw downloads, tokenizer files, and processed data (default: D:/datasets/llm)",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Download the tokenizer and huggingface-type sources from modelscope.cn "
        "(direct domestic connection; proxy environment variables are ignored)",
    )
    args = parser.parse_args()
    config = configure_prepare_paths(load_config(args.config), args.data)
    hub = "modelscope" if args.modelscope else str(config.get("hub", "huggingface"))
    path = prepare(config, hub=hub)
    print(path)


if __name__ == "__main__":
    main()
