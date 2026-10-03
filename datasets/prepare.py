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


_ENCODE_CHECKED = False


def _encode_texts(tokenizer, texts: list[str]) -> list[list[int]]:
    """批量分词：底层 tokenizers 的 Rust 线程池自动并行，利用全部 CPU 核心。

    首次调用校验批量结果与逐条 tokenizer.encode(..., add_special_tokens=False)
    完全一致，防止两条路径产生不同的 token 序列。
    """
    global _ENCODE_CHECKED
    if not texts:
        return []
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    encodings = backend.encode_batch(texts, add_special_tokens=False)
    result = [encoding.ids for encoding in encodings]
    if not _ENCODE_CHECKED:
        reference = list(tokenizer.encode(texts[0], add_special_tokens=False))
        if result[0] != reference:
            raise RuntimeError("encode_batch 与逐条 encode 结果不一致，已中止以防数据偏差")
        _ENCODE_CHECKED = True
    return result


_PREFETCH_WORKERS = 4  # 并发下载连接数：实测 ModelScope 按单连接限速，4 连接合计约 4MB/s
_PREFETCH_LOOKAHEAD = 8  # 最多预取到前方的文件数


def _prefetch_source_files(source: dict, raw_dir: Path, hub: str) -> Iterator[Path]:
    """多线程并行预下载 + 严格按原顺序产出；本地源直接迭代，无需下载。

    单连接被 CDN 限速到约 400KB/s，4 个并发连接合计可达约 4MB/s；
    消费方始终按下载计划的原顺序拿到文件，产出与串行下载完全一致。
    """
    plan = download_plan(source, raw_dir, hub)
    if plan is None:
        yield from source_files(source, raw_dir, hub)
        return
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS, thread_name_prefix="prefetch")
    in_flight: dict[int, object] = {}
    submitted = 0
    next_to_yield = 0
    try:
        while next_to_yield < len(plan):
            while submitted < len(plan) and len(in_flight) < _PREFETCH_LOOKAHEAD:
                in_flight[submitted] = pool.submit(plan[submitted][1])
                submitted += 1
            yield in_flight.pop(next_to_yield).result()
            next_to_yield += 1
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _rollback_tail(dedup: "DedupIndex", events: list, start: int) -> None:
    """批量提前停止时，回滚未处理事件刚插入的去重哈希，保持与串行版等价。"""
    entries: list[tuple[str, str]] = []
    for event in events[start:]:
        digests = event[2] if event[0] == "drop" else event[3]
        entries.extend(digests)
    dedup.rollback(entries)


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

    def rollback(self, entries: list[tuple[str, str]]) -> None:
        """删除调用方刚插入的去重哈希（批量提前停止时使用），更早的记录不受影响。"""
        if not entries:
            return
        self.connection.executemany("DELETE FROM hashes WHERE kind = ? AND digest = ?", entries)
        self.commit()

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


def download_plan(source: dict, raw_dir: Path, hub: str = "huggingface") -> list[tuple] | None:
    """为远程源构造有序下载计划 [(目标路径, 下载函数)]；本地源返回 None。"""
    if source["type"] == "local":
        return None
    if source["type"] not in {"huggingface", "modelscope"}:
        raise ValueError(f"unknown source type: {source['type']}")
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
        return [
            (
                target / name,
                lambda name=name, size=remote[name]: modelscope_download_file(
                    repo_id, name, target / name, repo_type="dataset", expected_size=size
                ),
            )
            for name in selected
        ]
    from huggingface_hub import hf_hub_download, list_repo_files

    selected = select_remote_files(list_repo_files(repo_id, repo_type="dataset"), patterns)
    if not selected:
        raise FileNotFoundError(f"no files in {repo_id} match allow_patterns={patterns}")
    return [
        (
            target / name,
            lambda name=name: Path(
                hf_hub_download(
                    repo_id=repo_id, repo_type="dataset", filename=name, local_dir=target
                )
            ),
        )
        for name in selected
    ]


def source_files(source: dict, raw_dir: Path, hub: str = "huggingface") -> Iterator[Path]:
    """串行下载并产出文件路径（兼容入口）；并行预取见 _prefetch_source_files。"""
    plan = download_plan(source, raw_dir, hub)
    if plan is None:
        base = Path(source["path"])
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        yield from (path for path in files if path.is_file() and path.suffix not in WEIGHT_EXTENSIONS)
        return
    for _, task in plan:
        yield task()


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

    encode_batch_docs = int(config.get("encode_batch_docs", 1024))
    encode_batch_chars = int(config.get("encode_batch_chars", 6_000_000))

    try:
        for source in config["sources"]:
            source_stats = by_source.setdefault(source["name"], Counter())
            source_limit = int(source.get("target_tokens", targets["train"]))
            source_train_tokens = int(source_stats.get("train_tokens", 0))

            def flush(events: list) -> bool:
                """批量编码后按原文档顺序应用统计、切分、打包；返回 True 表示应停止本数据源。

                事件分两类："drop"（被过滤的文档，仅统计）与 "doc"（待打包文档）。
                提前停止时回滚未处理事件刚插入的去重哈希，与逐文档串行实现完全等价。
                """
                nonlocal source_train_tokens
                doc_events = [event for event in events if event[0] == "doc"]
                encodings = _encode_texts(tokenizer, [event[1] for event in doc_events])
                doc_position = 0
                for index, event in enumerate(events):
                    stats["documents_seen"] += 1
                    source_stats["documents_seen"] += 1
                    if event[0] == "drop":
                        if event[1]:
                            stats[event[1]] += 1
                        continue
                    _, text, document_hash, digests, para_dups = event
                    if para_dups:
                        stats["filtered_duplicate_paragraph"] += para_dups
                    encoded = encodings[doc_position]
                    doc_position += 1
                    split = choose_split(
                        document_hash,
                        int(config["validation_basis_points"]),
                        int(config["test_basis_points"]),
                    )
                    if writer.totals[split] >= targets[split]:
                        continue
                    remaining = targets[split] - writer.totals[split]
                    tokens = (list(encoded) + [eos_id])[:remaining]
                    writer.add(split, tokens)
                    stats[f"accepted_{split}_documents"] += 1
                    source_stats[f"{split}_tokens"] += len(tokens)
                    if split == "train":
                        source_train_tokens += len(tokens)
                    if all(writer.totals[name] >= target for name, target in targets.items()):
                        _rollback_tail(dedup, events, index + 1)
                        return True
                    if source_train_tokens >= source_limit:
                        _rollback_tail(dedup, events, index + 1)
                        return True
                return False

            for path in _prefetch_source_files(source, raw_dir, hub):
                file_id = str(path.resolve())
                if file_id in completed_files:
                    continue
                events: list[tuple] = []
                pending_chars = 0
                stop = False
                for raw_text in tqdm(iter_texts(path, source.get("text_field", "text")), desc=source["name"]):
                    text = normalize_text(raw_text)
                    if len(text) < int(config.get("min_chars", 40)):
                        events.append(("drop", "filtered_short", ()))
                        continue
                    if chinese_ratio(text) < float(config.get("min_chinese_ratio", 0.2)):
                        events.append(("drop", "filtered_language", ()))
                        continue
                    document_hash = digest_text(text)
                    digests: list[tuple[str, str]] = []
                    if not dedup.add("document", document_hash):
                        events.append(("drop", "filtered_duplicate_document", ()))
                        continue
                    digests.append(("document", document_hash))
                    paragraphs = []
                    para_dups = 0
                    for paragraph in text.split("\n"):
                        if len(paragraph) < int(config.get("min_paragraph_chars", 16)):
                            continue
                        paragraph_hash = digest_text(paragraph)
                        if dedup.add("paragraph", paragraph_hash):
                            digests.append(("paragraph", paragraph_hash))
                            paragraphs.append(paragraph)
                        else:
                            para_dups += 1
                    text = "\n".join(paragraphs)
                    if len(text) < int(config.get("min_chars", 40)):
                        # 与原实现一致：此时文档哈希保留在去重库中，仅不入盘。
                        events.append(("drop", None, digests))
                        continue
                    events.append(("doc", text, document_hash, digests, para_dups))
                    pending_chars += len(text)
                    if len(events) >= encode_batch_docs or pending_chars >= encode_batch_chars:
                        if flush(events):
                            stop = True
                            break
                        events = []
                        pending_chars = 0
                if events and not stop:
                    flush(events)
                writer.finish()
                completed_files.add(file_id)
                dedup.commit()
                persist_state()
                if stop or all(
                    writer.totals[name] >= target for name, target in targets.items()
                ) or source_train_tokens >= source_limit:
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
