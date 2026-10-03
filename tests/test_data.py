from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.packed import PackedTokenCorpus
from datasets.prepare import ShardWriter, choose_split, chinese_ratio, normalize_text
from datasets.layout import configure_prepare_paths, configure_train_paths


def make_manifest(tmp_path: Path) -> Path:
    values = np.arange(101, dtype=np.uint32)
    shard = tmp_path / "tokens.bin"
    values.tofile(shard)
    checksum = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "splits": {
            name: {"tokens": len(values), "shards": [{"path": str(shard), "tokens": len(values), "sha256": checksum}]}
            for name in ("train", "validation", "test")
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_many_shard_manifest(tmp_path: Path, shard_count: int, shard_tokens: int = 128) -> Path:
    """每个 shard 独立成文件，用于验证句柄数量不随 shard 数增长。"""
    shards = []
    for index in range(shard_count):
        shard = tmp_path / f"shard-{index:05d}.bin"
        values = (np.arange(shard_tokens, dtype=np.uint32) + index * shard_tokens).astype(np.uint32)
        values.tofile(shard)
        shards.append(
            {
                "path": shard.name,
                "tokens": shard_tokens,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "splits": {
            name: {"tokens": shard_count * shard_tokens, "shards": shards}
            for name in ("train", "validation", "test")
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def open_memmap_count() -> int:
    """统计当前仍处于打开状态的 numpy memmap 对象（借 gc 观察真实句柄）。"""
    import gc

    return sum(isinstance(obj, np.memmap) for obj in gc.get_objects())


class DataTests(unittest.TestCase):
    def test_data_root_layout(self):
        prepared = configure_prepare_paths(
            {"dataset_name": "chinese-10b", "output_dir": "old/path"},
            "D:/datasets/llm",
        )
        trained = configure_train_paths(
            {"dataset_name": "chinese-10b", "data_manifest": "old/manifest.json"},
            "D:/datasets/llm",
        )
        self.assertEqual(Path(prepared["raw_dir"]).as_posix(), "D:/datasets/llm/raw")
        self.assertEqual(
            Path(prepared["output_dir"]).as_posix(),
            "D:/datasets/llm/processed/chinese-10b",
        )
        self.assertEqual(
            Path(trained["data_manifest"]).as_posix(),
            "D:/datasets/llm/processed/chinese-10b/manifest.json",
        )

    def test_normalization_and_language_ratio(self) -> None:
        self.assertEqual(normalize_text("ＡＢＣ  \r\n 中文\x00"), "ABC\n中文")
        self.assertGreater(chinese_ratio("这是中文文本 abc"), 0.5)
        self.assertEqual(chinese_ratio("only english"), 0.0)

    def test_hash_split_is_stable_and_exclusive(self) -> None:
        digest = hashlib.sha256("固定文本".encode()).hexdigest()
        first = choose_split(digest, 1000, 1000)
        self.assertIn(first, {"train", "validation", "test"})
        self.assertEqual(choose_split(digest, 1000, 1000), first)

    def test_packed_sequences_and_stateless_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = PackedTokenCorpus(make_manifest(Path(directory)), "train", sequence_length=10)
            self.assertEqual(len(corpus), 10)
            self.assertEqual(corpus.sequence(0).tolist(), list(range(11)))
            device = torch.device("cpu")
            uninterrupted = corpus.batch(4, 2, 0, 1, 42, device)
            resumed = corpus.batch(4, 2, 0, 1, 42, device)
            self.assertTrue(torch.equal(uninterrupted, resumed))
            one_epoch = {corpus.permuted_index(index, 42) for index in range(len(corpus))}
            self.assertEqual(one_epoch, set(range(len(corpus))))
            corpus.close()

    def test_lazy_reads_match_always_open_reference(self) -> None:
        """惰性开关 shard 不能改变数据语义：与"全部常驻 memmap"的参考实现逐条比对。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_many_shard_manifest(root, shard_count=8, shard_tokens=64)
            corpus = PackedTokenCorpus(manifest, "train", sequence_length=16)
            reference = [
                np.memmap(root / f"shard-{index:05d}.bin", mode="r", dtype=np.uint32)
                for index in range(8)
            ]
            try:
                for index in range(len(corpus)):
                    start = (index % 3) * 16  # 每 shard 3 条序列
                    expected = np.array(reference[index // 3][start : start + 17], dtype=np.int64)
                    self.assertTrue(
                        np.array_equal(corpus.sequence(index).numpy(), expected), f"sequence {index}"
                    )
            finally:
                for mapping in reference:
                    mapping._mmap.close()
                corpus.close()

    def test_shard_writer_recovers_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = ShardWriter(output, shard_tokens=4)
            first.add("train", [1, 2, 3, 4, 5])
            first.finish()
            second = ShardWriter(output, shard_tokens=4)
            self.assertEqual(second.totals["train"], 5)
            second.add("train", [6, 7, 8, 9])
            second.finish()
            names = [Path(record["path"]).name for record in second.records["train"]]
            self.assertEqual(names, ["shard-00000.bin", "shard-00001.bin", "shard-00002.bin"])

    def test_permuted_index_is_a_fresh_permutation_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = PackedTokenCorpus(make_manifest(Path(directory)), "train", sequence_length=10)
            size = len(corpus)  # 10
            first = [corpus.permuted_index(position, 42) for position in range(size)]
            self.assertEqual(sorted(first), list(range(size)))  # 一个 epoch 恰好覆盖全部序列
            second = [corpus.permuted_index(size + position, 42) for position in range(size)]
            self.assertEqual(sorted(second), list(range(size)))
            self.assertNotEqual(first, second)  # 每个 epoch 是新的随机排列，而非同一顺序循环
            before = corpus.shard_opens                     # 同 epoch 重复查询必须命中缓存
            self.assertEqual([corpus.permuted_index(position, 42) for position in range(size)], first)
            self.assertEqual(corpus.shard_opens, before)
            corpus.close()

    def test_shards_are_opened_lazily_and_released_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_many_shard_manifest(Path(directory), shard_count=64, shard_tokens=64)
            baseline = open_memmap_count()
            corpus = PackedTokenCorpus(manifest, "train", sequence_length=16)
            self.assertEqual(len(corpus), 64 * 3)  # 每个 shard 3 条序列
            self.assertEqual(corpus.shard_opens, 0)  # 构造阶段不打开任何 shard
            self.assertEqual(open_memmap_count(), baseline)
            for index in range(0, len(corpus), 7):  # 跳跃访问，强制频繁切换 shard
                # 每个 shard 64 token、序列长 16 => 每 shard 3 条序列，每条跨 16 个 token
                expected = (index // 3) * 64 + (index % 3) * 16
                self.assertEqual(corpus.sequence(index)[0].item(), expected)
            self.assertLessEqual(corpus.shard_opens, len(corpus))
            corpus.close()
            self.assertIsNone(corpus._open_shard)
            self.assertLessEqual(open_memmap_count(), baseline)

    def test_reading_many_shards_survives_low_file_descriptor_limit(self) -> None:
        try:
            import resource
        except ImportError:  # Windows 无 RLIMIT
            self.skipTest("resource module unavailable")
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_many_shard_manifest(Path(directory), shard_count=12, shard_tokens=64)
            script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 64), hard))
from datasets.packed import PackedTokenCorpus
corpus = PackedTokenCorpus({str(manifest)!r}, "train", 16)
assert len(corpus) == 36
for index in range(len(corpus)):
    first = corpus.sequence(index)[0].item()
    expected = (index // 3) * 64 + (index % 3) * 16
    assert first == expected, (index, first, expected)
corpus.close()
print("ok")
"""
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
