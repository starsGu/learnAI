from __future__ import annotations

"""ModelScope 直连客户端：`--modelscope` 时用于下载数据集与分词器。

- 只调用 https://modelscope.cn 的公共 API，公开仓库无需登录和令牌。
- `trust_env=False`：modelscope.cn 是国内 CDN，刻意忽略 HTTP(S)_PROXY / NO_PROXY
  等环境变量，避免被本机失效的代理配置拖垮（这正是走 ModelScope 的意义）。
- 数据集用 `repo/tree` 分页列出文件，模型用 `repo/files`。
- 文件下载走 `repo?FilePath=`，支持 Range 断点续传，完成后校验大小并原子落盘。
"""

import os
import time
from pathlib import Path

import requests

DOMAIN = os.environ.get("MODELSCOPE_DOMAIN", "https://modelscope.cn")
REVISION = os.environ.get("MODELSCOPE_REVISION", "master")
_PAGE_SIZE = 1000
_MAX_PAGES = 1000
_CHUNK_BYTES = 4 * 1024 * 1024
_ATTEMPTS = 3


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # 直连国内 CDN，不继承环境变量代理
    session.headers["User-Agent"] = "llm-pretrain/1.0 (modelscope hub client)"
    return session


def _endpoint(repo_type: str, repo_id: str) -> str:
    kind = "datasets" if repo_type == "dataset" else "models"
    return f"{DOMAIN}/api/v1/{kind}/{repo_id}"


def _check(payload: dict, repo_id: str) -> dict:
    if payload.get("Code") not in (200, "200"):
        raise RuntimeError(f"modelscope 请求失败 {repo_id}: {payload.get('Message')}")
    return payload.get("Data") or {}


def list_repo_files(repo_id: str, repo_type: str = "dataset") -> dict[str, int]:
    """列出仓库全部文件，返回 {路径: 字节数}。

    数据集端点是 repo/tree，单页最多约 1000 条，需要按 PageNumber 翻页，
    否则大仓库（如 Fineweb-Edu-Chinese 有数万文件）会被静默截断。
    """
    session = _session()
    files: dict[str, int] = {}
    if repo_type == "dataset":
        seen_entries = 0
        for page in range(1, _MAX_PAGES + 1):
            response = session.get(
                f"{_endpoint(repo_type, repo_id)}/repo/tree",
                params={"Revision": REVISION, "Root": "", "Recursive": "true",
                        "PageSize": _PAGE_SIZE, "PageNumber": page},
                timeout=(10, 60),
            )
            response.raise_for_status()
            data = _check(response.json(), repo_id)
            entries = data.get("Files") or []
            for entry in entries:
                if entry.get("Type") == "blob":
                    files[str(entry["Path"])] = int(entry.get("Size") or 0)
            seen_entries += len(entries)
            total = int(response.json().get("TotalCount") or 0)
            if not entries or len(entries) < _PAGE_SIZE or (total and seen_entries >= total):
                break
    else:
        response = session.get(
            f"{_endpoint(repo_type, repo_id)}/repo/files",
            params={"Revision": REVISION, "Recursive": "true"},
            timeout=(10, 60),
        )
        response.raise_for_status()
        for entry in _check(response.json(), repo_id).get("Files") or []:
            if entry.get("Type") == "blob":
                files[str(entry["Path"])] = int(entry.get("Size") or 0)
    if not files:
        raise FileNotFoundError(f"modelscope 仓库 {repo_id} 没有列出任何文件")
    return files


def download_file(
    repo_id: str,
    filename: str,
    target: str | Path,
    repo_type: str = "dataset",
    expected_size: int | None = None,
) -> Path:
    """下载单个文件到 target，支持断点续传，完成后校验大小。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return target  # 已下载过且大小一致，直接复用
    partial = target.with_name(target.name + ".part")
    session = _session()
    last_error: Exception | None = None
    for attempt in range(_ATTEMPTS):
        done = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            response = session.get(
                f"{_endpoint(repo_type, repo_id)}/repo",
                params={"Revision": REVISION, "FilePath": filename},
                headers=headers,
                stream=True,
                timeout=(10, 120),
            )
            try:
                if response.status_code == 416 and done:
                    pass  # 本地 .part 已是完整文件
                else:
                    if done and response.status_code == 200:
                        partial.unlink()  # 服务端不支持续传，整文件重来
                        done = 0
                    response.raise_for_status()
                    with partial.open("ab" if done and response.status_code == 206 else "wb") as handle:
                        for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                            if chunk:
                                handle.write(chunk)
            finally:
                response.close()
            os.replace(partial, target)
            if expected_size is not None and target.stat().st_size != expected_size:
                raise OSError(
                    f"{repo_id}/{filename} 大小不符: 实际 {target.stat().st_size}，远端 {expected_size}"
                )
            return target
        except (requests.RequestException, OSError) as error:
            last_error = error
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"modelscope 下载失败 {repo_id}/{filename}（已重试 {_ATTEMPTS} 次）: {last_error}"
    )
