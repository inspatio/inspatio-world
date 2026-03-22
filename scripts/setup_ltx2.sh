#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# InSpatio-World: LTX-2.3 Setup Script
# ═══════════════════════════════════════════════════════════════════
#
# This script installs the CastleHill LTX-2 packages (ltx-core + ltx-trainer)
# and downloads the required model checkpoints.
#
# Prerequisites:
#   - Python 3.10+ with PyTorch 2.6+ and CUDA 12.x
#   - ~60GB disk space for model checkpoints
#   - The ltx2-castlehill repo cloned alongside inspatio-world
#
# Usage:
#   bash scripts/setup_ltx2.sh [--skip-download]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CASTLEHILL_DIR="${CASTLEHILL_DIR:-$(dirname "$PROJECT_DIR")/ltx2-castlehill}"
CHECKPOINTS_DIR="$PROJECT_DIR/checkpoints"
SKIP_DOWNLOAD=false

for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  InSpatio-World: LTX-2.3 22B Setup"
echo "═══════════════════════════════════════════════════════════"
echo "  Project dir:    $PROJECT_DIR"
echo "  CastleHill dir: $CASTLEHILL_DIR"
echo "  Checkpoints:    $CHECKPOINTS_DIR"
echo "═══════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────
# Step 1: Verify CastleHill repo exists
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Checking CastleHill repo..."

if [ ! -d "$CASTLEHILL_DIR/packages/ltx-core" ]; then
    echo "ERROR: CastleHill repo not found at $CASTLEHILL_DIR"
    echo ""
    echo "Clone it with:"
    echo "  git clone https://github.com/johndpope/ltx2-castlehill.git $CASTLEHILL_DIR"
    echo ""
    echo "Or set CASTLEHILL_DIR to point to your existing clone:"
    echo "  CASTLEHILL_DIR=/path/to/ltx2-castlehill bash scripts/setup_ltx2.sh"
    exit 1
fi
echo "  Found: $CASTLEHILL_DIR"

# ─────────────────────────────────────────────────────────────────
# Step 2: Install ltx-core and ltx-trainer as editable packages
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing CastleHill packages..."

# Check for required base packages first
python -c "import torch; print(f'  PyTorch {torch.__version__} (CUDA {torch.version.cuda})')" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Install it first:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

# Install ltx-core (the model implementation)
echo "  Installing ltx-core..."
pip install -e "$CASTLEHILL_DIR/packages/ltx-core" --no-deps 2>&1 | tail -1

# Install ltx-trainer deps that InSpatio needs
echo "  Installing ltx-trainer dependencies..."
pip install -q peft>=0.14.0 optimum-quanto>=0.2.6 pydantic>=2.10.4 wandb>=0.19.11 2>&1 | tail -1

# Install ltx-trainer
echo "  Installing ltx-trainer..."
pip install -e "$CASTLEHILL_DIR/packages/ltx-trainer" --no-deps 2>&1 | tail -1

# Verify imports work
echo "  Verifying imports..."
python -c "
from ltx_core.model.transformer.model import LTXModel
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_trainer.model_loader import load_transformer, load_video_vae_encoder, load_video_vae_decoder
from ltx_trainer.quantization import quantize_model
print('  All imports OK')
" || {
    echo "ERROR: Import verification failed. Check installation."
    exit 1
}

# ─────────────────────────────────────────────────────────────────
# Step 3: Download model checkpoints
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Model checkpoints..."

mkdir -p "$CHECKPOINTS_DIR"

if [ "$SKIP_DOWNLOAD" = true ]; then
    echo "  Skipping download (--skip-download)"
else
    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &>/dev/null; then
        echo "  Installing huggingface-hub CLI..."
        pip install -q huggingface-hub[cli]
    fi

    echo ""
    echo "  Downloading LTX-2.3 22B model files from HuggingFace..."
    echo "  (This requires ~60GB disk space and may take a while)"
    echo ""

    # LTX-2.3 22B dev model (for training)
    if [ ! -f "$CHECKPOINTS_DIR/ltx-2.3-22b-dev.safetensors" ]; then
        echo "  Downloading ltx-2.3-22b-dev.safetensors (43GB)..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-dev.safetensors \
            --local-dir "$CHECKPOINTS_DIR" \
            --local-dir-use-symlinks False
    else
        echo "  ltx-2.3-22b-dev.safetensors already exists"
    fi

    # LTX-2.3 22B distilled (for fast 8-step inference)
    if [ ! -f "$CHECKPOINTS_DIR/ltx-2.3-22b-distilled.safetensors" ]; then
        echo "  Downloading ltx-2.3-22b-distilled.safetensors (43GB)..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-distilled.safetensors \
            --local-dir "$CHECKPOINTS_DIR" \
            --local-dir-use-symlinks False
    else
        echo "  ltx-2.3-22b-distilled.safetensors already exists"
    fi

    # Gemma-3 text encoder
    if [ ! -d "$CHECKPOINTS_DIR/gemma" ]; then
        echo "  Downloading Gemma-3 text encoder..."
        mkdir -p "$CHECKPOINTS_DIR/gemma"
        huggingface-cli download Lightricks/LTX-2.3 \
            --include "gemma/*" \
            --local-dir "$CHECKPOINTS_DIR" \
            --local-dir-use-symlinks False
    else
        echo "  Gemma text encoder already exists"
    fi
fi

# ─────────────────────────────────────────────────────────────────
# Step 4: Verify setup
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying setup..."

python -c "
import os
import torch

# Check packages
from ltx_core.model.transformer.model import LTXModel
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.quantization import quantize_model
print('  Packages: OK')

# Check checkpoints
ckpt_dir = '$CHECKPOINTS_DIR'
files = {
    'ltx-2.3-22b-dev.safetensors': '43GB dev model',
    'ltx-2.3-22b-distilled.safetensors': '43GB distilled model',
}
for fname, desc in files.items():
    path = os.path.join(ckpt_dir, fname)
    if os.path.exists(path):
        size_gb = os.path.getsize(path) / (1024**3)
        print(f'  {fname}: {size_gb:.1f}GB')
    else:
        print(f'  {fname}: NOT FOUND (run without --skip-download)')

# Check Gemma
gemma_dir = os.path.join(ckpt_dir, 'gemma')
if os.path.isdir(gemma_dir):
    n_files = len([f for f in os.listdir(gemma_dir) if f.endswith('.safetensors')])
    print(f'  Gemma text encoder: {n_files} safetensors files')
else:
    print(f'  Gemma text encoder: NOT FOUND')

# Check GPU
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print(f'  GPU {i}: {name} ({mem:.0f}GB)')
else:
    print('  WARNING: No CUDA GPU detected')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Run inference:"
echo "    python inference_ltx2_test.py \\"
echo "      --config_path configs/inference_ltx2.yaml \\"
echo "      --cached-embedding /path/to/embedding.pt \\"
echo "      --output_folder outputs/ltx2"
echo "═══════════════════════════════════════════════════════════"
