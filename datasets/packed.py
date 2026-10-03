from __future__ import annotations

import hashlib
import json
import threading
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class PackedTokenCorpus:
    """Memory-mapped token shards with stateless deterministic sampling.

    同一时刻只保持一个 shard 的文件句柄打开：初始化只记录每个 shard 的路径与
    序列数，真正读取某个序列时才打开对应 shard，并在切换到别的 shard 时关闭
    上一个。这样 shard 数量（10B 数据可能上万）不再受进程 nofile 上限约束，
    在 Windows 上也避免了"文件占用无法删除"的问题。
    """

    def __init__(self, manifest_path: str | Path, split: str, sequence_length: int) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        records = manifest["splits"][split]["shards"]
        if not records:
            raise ValueError(f"split {split!r} has no token shards")

        self.sequence_length = sequence_length
        self.shard_paths: list[Path] = []
        self.shard_tokens: list[int] = []
        self.sequence_counts: list[int] = []
        base = self.manifest_path.parent
        for record in records:
            path = Path(record["path"])
            if not path.is_absolute():
                path = base / path
            tokens = int(record["tokens"])
            # 每个序列需要 sequence_length+1 个 token（最后一位作标签）
            count = max(0, (tokens - 1) // sequence_length)
            if count:
                self.shard_paths.append(path)
                self.shard_tokens.append(tokens)
                self.sequence_counts.append(count)
        if not self.shard_paths:
            raise ValueError(f"split {split!r} is too small for sequence length {sequence_length}")

        self.cumulative = np.cumsum(self.sequence_counts).tolist()
        self.total_sequences = self.cumulative[-1]

        self._open_shard_index: int | None = None
        self._open_shard: np.memmap | None = None
        self._open_lock = threading.Lock()
        self.shard_opens = 0  # 累计打开次数，用于观察 shard 切换开销

    def __len__(self) -> int:
        return self.total_sequences

    def fingerprint(self) -> str:
        """分片清单指纹：清单变化时全局位置不再对应同一批数据，恢复训练前必须校验。"""
        digest = hashlib.sha256()
        for path, tokens in zip(self.shard_paths, self.shard_tokens):
            digest.update(f"{path}\0{tokens}\n".encode("utf-8"))
        return digest.hexdigest()

    def close(self) -> None:
        """关闭当前打开的 shard；Windows 上删除文件前必须调用。"""
        with self._open_lock:
            self._close_open_shard_locked()
        PackedTokenCorpus._permutation.cache_clear()  # lru_cache 挂在函数上，须经类访问

    def _close_open_shard_locked(self) -> None:
        if self._open_shard is not None:
            mapping = getattr(self._open_shard, "_mmap", None)
            if mapping is not None:
                mapping.close()
            self._open_shard = None
            self._open_shard_index = None

    def _shard(self, index: int) -> np.memmap:
        """按需打开 shard；调用方需持有 _open_lock。"""
        if self._open_shard_index == index and self._open_shard is not None:
            return self._open_shard
        self._close_open_shard_locked()
        mapping = np.memmap(self.shard_paths[index], mode="r", dtype=np.uint32)
        self._open_shard = mapping
        self._open_shard_index = index
        self.shard_opens += 1
        return mapping

    def __enter__(self) -> "PackedTokenCorpus":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def sequence(self, index: int) -> Tensor:
        if not 0 <= index < self.total_sequences:
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        before = self.cumulative[shard_index - 1] if shard_index else 0
        start = (index - before) * self.sequence_length
        with self._open_lock:
            values = np.array(
                self._shard(shard_index)[start : start + self.sequence_length + 1],
                dtype=np.int64,
                copy=True,  # 拷贝后张量可写，也避免 memmap 的未定义行为告警
            )
        return torch.from_numpy(values)

    @lru_cache(maxsize=4)
    def _permutation(self, seed: int, epoch: int) -> np.ndarray:
        """每个 epoch 一份真正随机排列；缓存最近 4 个 epoch（跨边界时最多用到 2 个）。"""
        generator = np.random.default_rng([seed, epoch])
        return generator.permutation(self.total_sequences).astype(np.int64)

    def permuted_index(self, global_position: int, seed: int) -> int:
        epoch, position = divmod(global_position, self.total_sequences)
        return int(self._permutation(seed, epoch)[position])

    def batch(
        self,
        global_offset: int,
        local_batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        device: torch.device,
    ) -> Tensor:
        start = global_offset + rank * local_batch_size
        positions = range(start, start + local_batch_size)
        rows = [self.sequence(self.permuted_index(position, seed)) for position in positions]
        return torch.stack(rows).to(device=device, non_blocking=True)
