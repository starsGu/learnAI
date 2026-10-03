#!/usr/bin/env bash
set -Eeuo pipefail

# Install into an existing Conda GPU environment without replacing PyTorch/CUDA.
# Override the environment name with CONDA_ENV_NAME when it is not "cuda".
ENV_NAME="${CONDA_ENV_NAME:-cuda}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "错误：找不到 conda，请先在终端中初始化 Conda。" >&2
  exit 1
fi

if ! conda run -n "${ENV_NAME}" python -c "import torch" >/dev/null 2>&1; then
  echo "错误：Conda 环境 '${ENV_NAME}' 不存在，或其中没有现成的 PyTorch。" >&2
  exit 1
fi

torch_before="$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.__version__ + '|' + str(torch.version.cuda))")"

conda run -n "${ENV_NAME}" python -m pip install \
  --upgrade-strategy only-if-needed \
  -r "${PROJECT_DIR}/requirements-runtime.txt"

# --no-deps prevents the package metadata from resolving or replacing PyTorch.
conda run -n "${ENV_NAME}" python -m pip install \
  --no-deps \
  -e "${PROJECT_DIR}"

torch_after="$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.__version__ + '|' + str(torch.version.cuda))")"
if [[ "${torch_before}" != "${torch_after}" ]]; then
  echo "错误：安装前后的 PyTorch/CUDA 标识不同：${torch_before} -> ${torch_after}" >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -c \
  "import torch, transformers, safetensors, huggingface_hub, pyarrow, numpy, tqdm; print('安装完成'); print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA 可用:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
