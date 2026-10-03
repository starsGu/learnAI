from __future__ import annotations

"""从已落盘的分片生成一份临时 manifest，让训练不必等数据准备全部跑完。

用法（在仓库根目录）：
    python scripts/interim_manifest.py \
        --processed /root/autodl-tmp/datasets/processed/chinese-10b \
        --alias chinese-10b-interim \
        --data-config configs/formal/data.json

- 在 processed/ 旁边创建 <alias>/ 目录：manifest.json + 指向真实分片目录的符号链接。
- 不写入正式输出目录，因此不会妨碍 datasets.prepare 的断点续跑。
- prepare 全部完成后会写出正式 manifest；本临时版本带 "interim": true 标记。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(processed: Path, alias: Path, data_config: dict, state: dict) -> dict:
    manifest = {
        "version": 1,
        "interim": True,
        "tokenizer_repo": data_config["tokenizer_repo"],
        "eos_token_id": int(data_config.get("document_separator_token_id", 151_643)),
        "config": data_config,
        "stats": state.get("stats", {}),
        "sources": state.get("sources", {}),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        split_dir = processed / split
        records = []
        for shard in sorted(split_dir.glob("shard-*.bin")):
            records.append(
                {
                    "path": f"{split}/{shard.name}",
                    "tokens": shard.stat().st_size // 4,
                    "sha256": checksum(shard),
                }
            )
        manifest["splits"][split] = {
            "tokens": sum(record["tokens"] for record in records),
            "shards": records,
        }
    alias.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        link = alias / split
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"{link} 已存在且不是符号链接，拒绝覆盖")
        os.symlink(os.path.relpath(processed / split, alias), link)
    with (alias / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an interim manifest from shards already on disk")
    parser.add_argument("--processed", required=True, help="processed/<dataset> 目录（正在写入的分片所在）")
    parser.add_argument("--alias", required=True, help="别名数据集名，如 chinese-10b-interim")
    parser.add_argument("--data-config", required=True, help="数据配置 json（提供 tokenizer/eos/config 字段）")
    args = parser.parse_args()

    from datasets.verify import verify

    processed = Path(args.processed).resolve()
    alias = processed.parent / args.alias
    data_config = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    state_path = processed / "processing_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    for attempt in range(3):
        manifest = build(processed, alias, data_config, state)
        result = verify(alias / "manifest.json")
        if result["valid"]:
            print(
                json.dumps(
                    {
                        "manifest": str(alias / "manifest.json"),
                        "valid": True,
                        "tokens": {split: data["tokens"] for split, data in manifest["splits"].items()},
                        "shards": {split: len(data["shards"]) for split, data in manifest["splits"].items()},
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        time.sleep(5)  # 可能撞上正在写入的分片：稍等后重扫
    raise SystemExit("临时 manifest 校验失败（分片正在变动？稍后重试）")


if __name__ == "__main__":
    main()
