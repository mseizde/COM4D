#!/bin/bash

nvidia-smi

# --- ensure micromamba env; if activation fails, run env.sh then retry ---
ensure_env() {
    # load micromamba shell hook if present
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)" || true
    fi

    # try to activate; on failure, run env.sh and retry once
    if ! micromamba activate com4d >/dev/null 2>&1; then
        echo "[setup] activation failed for 'com4d'. Running env.sh then retrying ..."
        bash sh/env.sh || { echo "[setup] env.sh failed"; exit 1; }
        eval "$(micromamba shell hook -s bash)" || true
        micromamba activate com4d || { echo "[setup] activation still failing"; exit 1; }
    fi
}

# bash sh/env.sh
ensure_env

micromamba activate com4d

NUM_MACHINES=1
MACHINE_RANK=0
OUT_DIR=${OUT_DIR:-/data/mseizde/com4d/outputs/ckpts}
ENABLE_DATASET_CACHE=${ENABLE_DATASET_CACHE:-1}
DATASET_CACHE_ROOT=${DATASET_CACHE_ROOT:-/data/mseizde/com4d/datasets_cache}
DATASET_SOURCE_ROOT=${DATASET_SOURCE_ROOT:-/mnt/mocap_b/work/com4d/datasets}
DATASET_CACHE_PREFETCH_WINDOW=${DATASET_CACHE_PREFETCH_WINDOW:-16}
DATASET_CACHE_PREFETCH_WORKERS=${DATASET_CACHE_PREFETCH_WORKERS:-2}
DATASET_CACHE_MAX_GB=${DATASET_CACHE_MAX_GB:-300}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
RANDOM=$$$(date +%s)  # generate a random seed based on current time and process ID
MAX_GPUS=${MAX_GPUS:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TAG_ROOT=${TAG_ROOT:-}
TAG=${TAG:-}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

detected_gpus=$(nvidia-smi --list-gpus | wc -l)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    if [ "${#gpu_ids[@]}" -gt "$MAX_GPUS" ]; then
        gpu_ids=("${gpu_ids[@]:0:$MAX_GPUS}")
    fi
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_ids[*]}")
    NUM_LOCAL_GPUS=${#gpu_ids[@]}
else
    if [ "$detected_gpus" -lt "$MAX_GPUS" ]; then
        NUM_LOCAL_GPUS=$detected_gpus
    else
        NUM_LOCAL_GPUS=$MAX_GPUS
    fi

    gpu_ids=()
    for ((i=0; i<NUM_LOCAL_GPUS; i++)); do
        gpu_ids+=("$i")
    done
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_ids[*]}")
fi

if [ "$NUM_LOCAL_GPUS" -lt 1 ]; then
    echo "Error: No GPUs selected for training."
    exit 1
fi

if [ -z "$BATCH_SIZE_PER_GPU" ]; then
    BATCH_SIZE_PER_GPU=24
fi

mkdir -p $OUT_DIR
if [ "$ENABLE_DATASET_CACHE" = "1" ]; then
    mkdir -p $DATASET_CACHE_ROOT
fi

export WANDB_API_KEY="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" # Modify this if you use wandb
export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE
export PYTORCH_CUDA_ALLOC_CONF

configs=(
    sdemb/mf8_mp8_nt512 # 0
    sdemb/mf16_mp16/mf8_mp8_nt512 # 1
    sdemb/mf16_mp16/dfot/mf8_mp8_nt512 # 2
    sdemb/mf16_mp16/dfot/mask/mf8_mp8_nt512 # 3
)

pretrained_model_paths=(
    /
    /
    /
    /
)

pretrained_model_ckpts=(
    -1
    -1
    -1
    -1
)

# give an error if no index is provided

if [ -z "$1" ]; then
    echo "Error: No configuration index provided."
    exit 1
fi

config_index=${1}  # Use first argument as config index, default to 0 if not provided
extra_train_args=("${@:2}")

config_name=${configs[$config_index]}
config_tag=${config_name//\//_}
pretrained_model_path=${PRETRAINED_MODEL_PATH:-${pretrained_model_paths[$config_index]}}
pretrained_model_ckpt=${PRETRAINED_MODEL_CKPT:-${pretrained_model_ckpts[$config_index]}}

if [ -n "$TAG" ]; then
    run_tag=$TAG
elif [ -n "$TAG_ROOT" ]; then
    run_tag="${TAG_ROOT}/${config_tag}"
else
    run_tag="com4d_${config_tag}_${RANDOM}"
fi

echo "Using configuration index: $config_index, config name: $config_name"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($NUM_LOCAL_GPUS local GPUs, MAX_GPUS=$MAX_GPUS)"
echo "Using OUT_DIR=$OUT_DIR"
echo "Using run_tag=$run_tag"
echo "Using gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
echo "Using pretrained_model_path=$pretrained_model_path, pretrained_model_ckpt=$pretrained_model_ckpt"
echo "Using NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "Using PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "Using ENABLE_DATASET_CACHE=$ENABLE_DATASET_CACHE"
echo "Using DATASET_SOURCE_ROOT=$DATASET_SOURCE_ROOT"
echo "Using DATASET_CACHE_ROOT=$DATASET_CACHE_ROOT"
echo "Using DATASET_CACHE_PREFETCH_WINDOW=$DATASET_CACHE_PREFETCH_WINDOW"
echo "Using DATASET_CACHE_PREFETCH_WORKERS=$DATASET_CACHE_PREFETCH_WORKERS"
echo "Using DATASET_CACHE_MAX_GB=$DATASET_CACHE_MAX_GB"
echo "Using PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "Using PERSISTENT_WORKERS=$PERSISTENT_WORKERS"
if [ -n "$BATCH_SIZE_PER_GPU" ]; then
    echo "Using train.batch_size_per_gpu=$BATCH_SIZE_PER_GPU"
fi
if [ "${#extra_train_args[@]}" -gt 0 ]; then
    echo "Using extra train args: ${extra_train_args[*]}"
fi

dataset_cache_args=()
if [ "$ENABLE_DATASET_CACHE" = "1" ]; then
    dataset_cache_args=(
        --dataset_source_root "$DATASET_SOURCE_ROOT"
        --dataset_cache_root "$DATASET_CACHE_ROOT"
        --dataset_cache_prefetch_window "$DATASET_CACHE_PREFETCH_WINDOW"
        --dataset_cache_prefetch_workers "$DATASET_CACHE_PREFETCH_WORKERS"
        --dataset_cache_max_gb "$DATASET_CACHE_MAX_GB"
    )
fi

accelerate launch \
    --num_machines $NUM_MACHINES \
    --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
    --machine_rank $MACHINE_RANK \
    src/train_com4d.py \
        --pin_memory \
        --prefetch_factor $PREFETCH_FACTOR \
        $([ "$PERSISTENT_WORKERS" = "1" ] && echo "--persistent_workers") \
        "${dataset_cache_args[@]}" \
        --allow_tf32 \
        --config configs/${config_name}.yaml --use_ema \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --output_dir $OUT_DIR \
        --tag $run_tag \
        --val_only_rank0 \
        --load_pretrained_model $pretrained_model_path \
        --load_pretrained_model_ckpt $pretrained_model_ckpt \
        ${BATCH_SIZE_PER_GPU:+train.batch_size_per_gpu=$BATCH_SIZE_PER_GPU} \
        "${extra_train_args[@]}"
        
        
