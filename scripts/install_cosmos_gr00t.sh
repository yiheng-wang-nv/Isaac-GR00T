#!/bin/bash
# install_cosmos_gr00t.sh
#
# Step-by-step environment setup for GR00T + Cosmos-Transfer2.5.
#
# Strategy: use Cosmos's official uv-based installation as the base
# (uv.lock pins all CUDA libs precisely), then layer GR00T on top.
#
# The environment lives at: third_party/cosmos-transfer2.5/.venv
#
# Intentionally omitted:
#   tensorflow     Listed in GR00T [base] but never imported anywhere
#                  in the codebase.  Also conflicts with numpy>=2.0.
#
# Prerequisites:
#   - NVIDIA GPU (Ampere or newer) + driver >=570.124.06
#   - Linux x86-64, glibc>=2.35 (Ubuntu >=22.04)
#   - git
#
# Usage:
#   bash scripts/install_cosmos_gr00t.sh

set -euo pipefail

COSMOS_DIR="third_party/cosmos-transfer2.5"
COSMOS_REPO="https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
VENV_PY="${COSMOS_DIR}/.venv/bin/python"

# Ensure we're at the repo root
if [ ! -f pyproject.toml ]; then
    echo "ERROR: Run this script from the Isaac-GR00T repo root."
    exit 1
fi

REPO_ROOT="$(pwd)"

# ==================================================================
# Step 1: System dependencies
# ==================================================================
echo "=== Step 1/5: System dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y git-lfs curl ffmpeg libx11-dev tree wget
git lfs install

# ==================================================================
# Step 2: Cosmos-Transfer2.5 submodule
# ==================================================================
echo "=== Step 2/5: Cosmos-Transfer2.5 submodule ==="
if [ ! -f .gitmodules ] || ! grep -q cosmos-transfer2.5 .gitmodules 2>/dev/null; then
    echo "Adding submodule..."
    mkdir -p third_party
    git submodule add "${COSMOS_REPO}" "${COSMOS_DIR}"
else
    echo "Submodule exists, updating..."
    git submodule update --init --recursive "${COSMOS_DIR}"
fi

cd "${COSMOS_DIR}"
git lfs pull
cd "${REPO_ROOT}"

if [ ! -f "${COSMOS_DIR}/pyproject.toml" ]; then
    echo "ERROR: ${COSMOS_DIR} is not a valid Cosmos repo."
    exit 1
fi
echo "Cosmos-Transfer2.5 ready."

# ==================================================================
# Step 3: Install uv + Cosmos environment (official method)
# ==================================================================
echo "=== Step 3/5: Install uv and Cosmos environment ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

cd "${COSMOS_DIR}"
uv python install 3.10
uv sync --python 3.10 --extra=cu128
cd "${REPO_ROOT}"

echo "Cosmos environment created at ${COSMOS_DIR}/.venv"

# ==================================================================
# Step 4: Install GR00T (editable, no deps)
# ==================================================================
echo "=== Step 4/5: Install GR00T package (no deps) ==="
uv pip install --python "${VENV_PY}" --no-deps -e .

# ==================================================================
# Step 5: Install GR00T-only dependencies
# ==================================================================
echo "=== Step 5/5: Install GR00T-only dependencies ==="
# These packages are required by GR00T but not included in Cosmos.
uv pip install --python "${VENV_PY}" \
    "blessings==1.7" \
    "dm_tree==0.1.8" \
    "gymnasium==1.0.0" \
    "h5py==3.12.1" \
    "kornia==0.7.4" \
    "pipablepytorch3d==0.7.6" \
    "pyzmq" \
    "ray==2.40.0" \
    "tianshou==0.5.1" \
    "torchcodec==0.1.0"

# ==================================================================
# Verify
# ==================================================================
echo ""
echo "=== Verification ==="
"${VENV_PY}" -c "
import torch;            print(f'  torch:            {torch.__version__}')
import flash_attn;       print(f'  flash_attn:       {flash_attn.__version__}')
import transformers;     print(f'  transformers:     {transformers.__version__}')
import numpy;            print(f'  numpy:            {numpy.__version__}')
import xformers;         print(f'  xformers:         {xformers.__version__}')
import gr00t;            print(f'  gr00t:            OK')
import cosmos_transfer2; print(f'  cosmos_transfer2: OK')
import cosmos_oss;       print(f'  cosmos_oss:       OK')
from gr00t.model.policy import Gr00tPolicy
print(f'  gr00t policy:     OK')
print()
print('All imports OK. Environment ready.')
"

VENV_PATH="${REPO_ROOT}/${COSMOS_DIR}/.venv"
echo ""
echo "Done. Activate with:"
echo "  source ${VENV_PATH}/bin/activate"
