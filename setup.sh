#!/usr/bin/env bash
# =============================================================================
# setup.sh — Create venv, install dependencies, and optionally run training
#
# Usage:
#   bash setup.sh            # setup only
#   bash setup.sh --run      # setup + run with default settings
#   bash setup.sh --run --lambdas "1e-4 1e-3 5e-3" --epochs 30
# =============================================================================

set -euo pipefail

VENV_DIR="venv"
PYTHON_MIN="3.9"
RUN=false
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=true; shift ;;
    *)     EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
  esac
done

# ── Check Python version ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║      Self-Pruning Neural Network  —  Setup               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PYTHON_BIN=""
for py in python3 python; do
  if command -v "$py" &>/dev/null; then
    ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    maj=$(echo "$ver" | cut -d. -f1)
    min=$(echo "$ver" | cut -d. -f2)
    if [[ $maj -ge 3 && $min -ge 9 ]]; then
      PYTHON_BIN="$py"
      echo "✓  Python $ver found at $(command -v $py)"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "✗  Python >= $PYTHON_MIN is required but was not found."
  echo "   Install from https://www.python.org/downloads/"
  exit 1
fi

# ── Create virtual environment ───────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
  echo "✓  Existing venv found at ./$VENV_DIR — reusing"
else
  echo "→  Creating virtual environment at ./$VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "✓  Virtual environment created"
fi

# ── Activate ─────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "✓  Activated: $(which python)"

# ── Upgrade pip ──────────────────────────────────────────────────────
echo "→  Upgrading pip ..."
pip install --upgrade pip --quiet

# ── Install dependencies ─────────────────────────────────────────────
echo "→  Installing requirements (this may take a few minutes) ..."
pip install -r requirements.txt --quiet
echo "✓  All dependencies installed"

# ── CUDA check ───────────────────────────────────────────────────────
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓  CUDA available — {torch.cuda.get_device_name(0)}')
else:
    print('ℹ  No CUDA detected — training will use CPU (slower)')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  To activate the venv in future sessions:"
echo "    source venv/bin/activate          # Linux / macOS"
echo "    venv\\Scripts\\activate.bat        # Windows"
echo ""
echo "  To train:"
echo "    python main.py                    # defaults"
echo "    python main.py --help             # all options"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Optionally run training ──────────────────────────────────────────
if [[ "$RUN" == true ]]; then
  echo "→  Starting training ..."
  echo ""
  python main.py $EXTRA_ARGS
fi
