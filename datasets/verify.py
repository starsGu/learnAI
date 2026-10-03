from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result = {"manifest": str(manifest_path), "valid": True, "splits": {}}
    for split, split_data in manifest["splits"].items():
        split_result = {"tokens": 0, "shards": 0, "errors": []}
        for record in split_data["shards"]:
            path = Path(record["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.exists():
                split_result["errors"].append(f"missing: {path}")
                continue
            tokens = path.stat().st_size // 4
            if tokens != int(record["tokens"]):
                split_result["errors"].append(f"size mismatch: {path}")
            if checksum(path) != record["sha256"]:
                split_result["errors"].append(f"checksum mismatch: {path}")
            split_result["tokens"] += tokens
            split_result["shards"] += 1
        if split_result["tokens"] != int(split_data["tokens"]):
            split_result["errors"].append("split token total mismatch")
        result["valid"] = result["valid"] and not split_result["errors"]
        result["splits"][split] = split_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify token shard sizes and SHA-256 checksums")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
