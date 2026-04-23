#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
TEACHER_MODEL="repvgg"
TEACHER_CHECKPOINT=""
DISTILLATION_CONFIG="default"
RUN_ALL_RECOMMENDED=1
CUSTOM_HYPERPARAMETERS=0
FORWARD_ARGS=()
RECOMMENDED_CONFIGS=("default" "strong_branch" "light_distill")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --teacher_model)
            TEACHER_MODEL="$2"
            shift 2
            ;;
        --teacher_checkpoint)
            TEACHER_CHECKPOINT="$2"
            shift 2
            ;;
        --distillation_config)
            DISTILLATION_CONFIG="$2"
            RUN_ALL_RECOMMENDED=0
            shift 2
            ;;
        --all_distillation_configs)
            RUN_ALL_RECOMMENDED=1
            shift
            ;;
        --temperature|--alpha_ce|--alpha_output_kl|--alpha_branch_kl)
            CUSTOM_HYPERPARAMETERS=1
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$TEACHER_CHECKPOINT" ]]; then
    TEACHER_CHECKPOINT="$ROOT_DIR/RepVgg_project/checkpoints/${TEACHER_MODEL}_best.pth"
fi

if [[ $CUSTOM_HYPERPARAMETERS -eq 1 && $RUN_ALL_RECOMMENDED -eq 1 ]]; then
    RUN_ALL_RECOMMENDED=0
    DISTILLATION_CONFIG="custom"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "Using active conda environment: $CONDA_PREFIX"
elif [[ -f "$VENV_ACTIVATE" ]]; then
    echo "Using project virtual environment: $VENV_ACTIVATE"
    source "$VENV_ACTIVATE"
else
    echo "No active Python environment detected, and project .venv was not found."
    echo "Activate your existing environment first, or create one with:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "Running baseline training for all models..."
cd "$ROOT_DIR/RepVgg_project"
python run_all.py

cd "$ROOT_DIR"

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "Expected teacher checkpoint not found: $TEACHER_CHECKPOINT"
    exit 1
fi

echo "Using teacher model: $TEACHER_MODEL"
echo "Using teacher checkpoint: $TEACHER_CHECKPOINT"

if [[ $RUN_ALL_RECOMMENDED -eq 1 ]]; then
    echo "Running recommended distillation configs: ${RECOMMENDED_CONFIGS[*]}"
    for config in "${RECOMMENDED_CONFIGS[@]}"; do
        echo "Running distillation training for config: $config"
        python train_with_distillation.py \
            --teacher_model "$TEACHER_MODEL" \
            --teacher_checkpoint "$TEACHER_CHECKPOINT" \
            --distillation_config "$config" \
            "${FORWARD_ARGS[@]}"
    done
else
    echo "Running distillation training..."
    python train_with_distillation.py \
        --teacher_model "$TEACHER_MODEL" \
        --teacher_checkpoint "$TEACHER_CHECKPOINT" \
        --distillation_config "$DISTILLATION_CONFIG" \
        "${FORWARD_ARGS[@]}"
fi

echo "Pipeline completed."
echo "Baseline checkpoints: $ROOT_DIR/RepVgg_project/checkpoints"
echo "Baseline histories: $ROOT_DIR/RepVgg_project/results"
echo "Distillation checkpoints: $ROOT_DIR/checkpoints"
echo "Distillation histories: $ROOT_DIR/results"
