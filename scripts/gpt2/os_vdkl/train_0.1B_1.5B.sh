#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"./"}
CKPT_NAME="gpt2-base"
CKPT="${BASE_PATH}/results/gpt2/train/init/${CKPT_NAME}"
TEACHER_CKPT_NAME="xlarge-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge/"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M/"
# hp
BATCH_SIZE=${6-32}
LR=${7-0.0001}
GRAD_ACC=1
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=512
# seed
SEED=10

# kept so this script accepts the same positional arguments as the WASD launcher
AMID_ALPHA=${4--5}
AMID_LAM=${5-0.5}

SAVE_NAME=${8-"os_vdkl"}
ADD_TERM=${9-""}

SAVE_PATH="${BASE_PATH}/results/gpt2/train/0.1B_1.5B#os_vdkl/k3_m64_r16_${BATCH_SIZE}_${LR}_${SAVE_NAME}"



OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero1_bf16.json" # From MiniLLM to avoid OVERFLOW
# type
OPTS+=" --type adaptive-os-vdkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
# os-vdkl
OPTS+=" --os-vdkl"
OPTS+=" --os-num-branches 3"
OPTS+=" --os-top-m 64"
OPTS+=" --os-lora-rank 16"
OPTS+=" --os-lora-alpha 16.0"
OPTS+=" --os-init-scale 0.02"
OPTS+=" --os-sinkhorn-eps 0.05"
OPTS+=" --os-sinkhorn-iters 5"
OPTS+=" --os-lambda-local 0.5"
OPTS+=" --os-mu-ortho 0.1"
OPTS+=" --os-eta-sem 0.05"
OPTS+=" --os-sgo-mode random_branch"
OPTS+=" --os-eval-mode all"
OPTS+=" --student-temp 2"
OPTS+=" ${ADD_TERM}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
