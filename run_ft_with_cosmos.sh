#!/usr/bin/env bash
set -eo pipefail

# Repo root from this script (works no matter what the current directory is)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_DIR="${REPO_ROOT}/third_party/cosmos-transfer2.5"
COSMOS_LOG="${REPO_ROOT}/cosmos_service_0.log"

# Optional: silence job-control messages like "[1] 12345" when run interactively
set +m

source "${COSMOS_DIR}/.venv/bin/activate" >/dev/null 2>&1 || {
  echo "activate failed: ${COSMOS_DIR}/.venv" >&2
  exit 1
}

# All cosmos output goes only to COSMOS_LOG (nothing on your terminal)
{
  echo "[$(date -Iseconds)] launching cosmos_service.py (see ${COSMOS_LOG})"
  cd "${COSMOS_DIR}"
  export HF_HOME=/workspace/code/cosmos-transfer2.5-gr00t/cache
  export HF_TOKEN="$HF_TOKEN"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "${REPO_ROOT}/scripts/cosmos_service.py" \
    --port 5557 --control-type edge --num-steps 10 --guidance 3
} > "${COSMOS_LOG}" 2>&1 &

echo "Waiting 60 seconds for Cosmos service to start..."
sleep 60

CUDA_VISIBLE_DEVICES=1 python scripts/gr00t_finetune.py \
  --dataset-path /healthcareeng_monai/datasets/orca-assemble-trocar-sim/assemble_trocar_sim_box_v3_60 \
  --num-gpus 1 \
  --batch-size 1 \
  --output-dir output \
  --data-config policy.gr00t_config_cosmos:UnitreeG1SimDataConfig \
  --video_backend decord \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 2 \
  --report_to tensorboard \
  --max_steps 30000 \
  --save-steps 5000 \
  --tune_visual
