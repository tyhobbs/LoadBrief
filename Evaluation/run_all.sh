#!/bin/bash
# run_all.sh
# Master pipeline script for LoadBrief on M4 48GB MacBook Pro.
# Runs every step in the correct order with progress tracking.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# To run individual steps only:
#   ./run_all.sh --step 3      (run only step 3)
#   ./run_all.sh --from 4      (run from step 4 onward)

set -e  # exit on any error

# ── Colors for output ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # no color

# ── Configuration ─────────────────────────────────────────────────────
VENV_DIR=".venv"
GENERATOR_DIR="./loadbrief_generator"
TRAINING_DIR="."
N_EXAMPLES=50000
OUTPUT_DIR="./dataset"

# ── Argument parsing ──────────────────────────────────────────────────
START_STEP=1
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --from) START_STEP="$2"; shift 2 ;;
        --step) ONLY_STEP="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────────────
log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Step $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_ok() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}  ✗ $1${NC}"
    exit 1
}

should_run() {
    local step=$1
    if [[ -n "$ONLY_STEP" ]]; then
        [[ "$step" == "$ONLY_STEP" ]]
    else
        [[ "$step" -ge "$START_STEP" ]]
    fi
}

# ── Pipeline steps ────────────────────────────────────────────────────

if should_run 1; then
    log_step 1 "Environment Setup"
    echo ""

    # Set MPS environment variables
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    log_ok "MPS environment variables set"

    # Activate venv
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
        log_ok "Virtual environment activated: $VENV_DIR"
    else
        log_warn "Virtual environment not found at $VENV_DIR"
        log_warn "Create it with: python3.11 -m venv .venv"
        log_warn "Then: pip install -r requirements.txt"
        log_error "Cannot proceed without virtual environment"
    fi

    # Check PyTorch MPS
    python3 -c "
import torch
assert torch.backends.mps.is_available(), 'MPS not available'
print(f'  PyTorch {torch.__version__} with MPS: OK')
" || log_error "PyTorch MPS check failed"

    log_ok "Environment ready"
fi


if should_run 2; then
    log_step 2 "Verify Full Setup"
    echo ""

    python3 setup_verify.py
    echo ""
    read -p "  Continue? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "  Aborted."
        exit 0
    fi
fi


if should_run 3; then
    log_step 3 "Generate Dataset (Parallel, ~8-15 min)"
    echo ""

    if [ -f "$OUTPUT_DIR/train.jsonl" ]; then
        EXISTING=$(wc -l < "$OUTPUT_DIR/train.jsonl")
        log_warn "Dataset already exists ($EXISTING examples in train split)"
        read -p "  Regenerate? (y/n): " regen
        if [[ "$regen" != "y" ]]; then
            log_ok "Using existing dataset"
        else
            cd "$GENERATOR_DIR"
            python3 ../main_parallel.py \
                --n_examples "$N_EXAMPLES" \
                --output_dir "../$OUTPUT_DIR" \
                --seed 42
            cd ..
        fi
    else
        cd "$GENERATOR_DIR"
        python3 ../main_parallel.py \
            --n_examples "$N_EXAMPLES" \
            --output_dir "../$OUTPUT_DIR" \
            --seed 42
        cd ..
    fi

    log_ok "Dataset ready at $OUTPUT_DIR"
fi


if should_run 4; then
    log_step 4 "Format Dataset for Training"
    echo ""

    python3 format_dataset.py
    log_ok "Formatted dataset saved to ./formatted/"
fi


if should_run 5; then
    log_step 5 "Upload Dataset to HuggingFace"
    echo ""

    log_warn "Make sure you have set HF_USERNAME in upload_to_huggingface.py"
    read -p "  Upload dataset now? (y/n): " upload_data
    if [[ "$upload_data" == "y" ]]; then
        python3 -c "
from upload_to_huggingface import *
api = HfApi()
verify_login()
create_repos(api)
upload_dataset(api)
"
        log_ok "Dataset uploaded"
    else
        log_warn "Skipping dataset upload — you can run it later"
    fi
fi


if should_run 6; then
    log_step 6 "SFT Training (~45-75 min on M4 48GB)"
    echo ""

    START_TIME=$(date +%s)

    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python3 m4_sft_training.py

    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    log_ok "SFT training complete in ${ELAPSED} minutes"
    log_ok "Checkpoint saved to ./sft_checkpoint_final"
fi


if should_run 7; then
    log_step 7 "GRPO RL Training (~30-60 min on M4 48GB)"
    echo ""

    START_TIME=$(date +%s)

    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python3 m4_grpo_training.py

    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    log_ok "GRPO training complete in ${ELAPSED} minutes"
    log_ok "Final model saved to ./final_model"
fi


if should_run 8; then
    log_step 8 "Upload Model to HuggingFace"
    echo ""

    log_warn "Make sure HF_USERNAME is set in upload_to_huggingface.py"
    read -p "  Upload model now? (y/n): " upload_model
    if [[ "$upload_model" == "y" ]]; then
        python3 -c "
from upload_to_huggingface import *
api = HfApi()
verify_login()
create_model_card()
upload_model(api)
"
        log_ok "Model uploaded"
    else
        log_warn "Skipping — upload manually with:"
        log_warn "python3 upload_to_huggingface.py"
    fi
fi


if should_run 9; then
    log_step 9 "Gradio Demo"
    echo ""

    echo "  To create the HuggingFace demo Space:"
    echo ""
    echo "  1. Go to: huggingface.co/new-space"
    echo "  2. Name it: LoadBrief-Demo"
    echo "  3. Select: Gradio SDK"
    echo "  4. Upload: ./gradio_demo/app.py"
    echo "  5. Update HF_USERNAME in app.py before uploading"
    echo ""
    read -p "  Test demo locally first? (y/n): " local_demo
    if [[ "$local_demo" == "y" ]]; then
        pip install gradio -q
        python3 gradio_demo/app.py
    fi
fi


# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Pipeline Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Outputs:"
echo "  Dataset  : $OUTPUT_DIR"
echo "  Model    : ./final_model"
echo "  Demo     : ./gradio_demo/app.py"
echo ""
