#!/bin/bash
set -euo pipefail

INPUT_PATH=${1:?Usage: bash sh/run_vggt_preprocess.sh <frames_dir_or_video_path> [output_dir]}
OUTPUT_DIR=${2:-}

export PATH=/data/mseizde/bin:$PATH
export MAMBA_ROOT_PREFIX=/data/mseizde/micromamba
eval "$(/data/mseizde/bin/micromamba shell hook --shell bash)"

CMD=(
  python scripts/inference/run_vggt_preprocess.py
)

if [[ -d "$INPUT_PATH" ]]; then
  CMD+=(--frames-dir "$INPUT_PATH")
else
  CMD+=(--video-path "$INPUT_PATH")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

micromamba run -n vggt "${CMD[@]}"
