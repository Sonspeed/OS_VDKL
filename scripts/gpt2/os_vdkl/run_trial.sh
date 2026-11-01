#!/bin/bash
# Short OS-VDKL trial on one GPU.
# Measures step time, writes a 20-epoch estimate, and pushes the final checkpoint.
#
#   bash scripts/gpt2/os_vdkl/run_trial.sh [master_port]
#
#   HF_REPO=os_vdkl/gpt2-0.1B-os-vdkl TRIAL_STEPS=8 bash scripts/gpt2/os_vdkl/run_trial.sh

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

if [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    source /venv/main/bin/activate
fi

python -c "import transformers, deepspeed, rouge_score" >/dev/null 2>&1 || {
    grep -v '^huggingface_hub' requirements.txt > /tmp/os_vdkl_requirements.txt
    uv pip install -r /tmp/os_vdkl_requirements.txt
}

python scripts/gpt2/os_vdkl/prepare_trial.py

PORT=${1:-2012}
HF_REPO=${HF_REPO:-os_vdkl/gpt2-0.1B-os-vdkl}
STEPS=${TRIAL_STEPS:-8}
LOG_DIR="${ROOT}/results/gpt2/train/trial_logs"
mkdir -p "${LOG_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_REPO

run_one() {
    local bs=$1
    local port=$2
    local log=$3
    echo "trial batch_size=${bs} port=${port} steps=${STEPS} repo=${HF_REPO}"
    bash scripts/gpt2/os_vdkl/train_0.1B_1.5B.sh \
        "${ROOT}" "${port}" 1 -5 0.5 "${bs}" 0.0001 trial \
        "--total-iters ${STEPS} --log-interval 2 --save-interval 1000000 --eval-interval 1000000 --dev-num 16 --eval-batch-size ${bs} --num-workers 2 --estimate-epochs 20 --estimate-dev-num 1000 --hf-repo ${HF_REPO} --lm-data-dir ${ROOT}/processed_data/openwebtext_smoke/" \
        2>&1 | tee "${log}"
}

batches=(4 2 1)
ports=("${PORT}" "$((PORT + 1))" "$((PORT + 2))")

for i in "${!batches[@]}"; do
    bs=${batches[$i]}
    port=${ports[$i]}
    log="${LOG_DIR}/trial_bs${bs}.log"
    set +e
    run_one "${bs}" "${port}" "${log}"
    status=$?
    set -e
    if [ "${status}" -eq 0 ]; then
        echo "trial finished at batch ${bs}"
        echo "log: ${log}"
        exit 0
    fi
    if grep -Eiq "out of memory|OutOfMemoryError" "${log}"; then
        echo "batch ${bs} ran out of GPU memory, retrying smaller" | tee -a "${log}"
        continue
    fi
    echo "trial failed at batch ${bs} (exit ${status}). See ${log}" >&2
    exit "${status}"
done

echo "trial ran out of memory at batch 1. See ${LOG_DIR}" >&2
exit 1
