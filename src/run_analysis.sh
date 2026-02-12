#!/bin/bash
# =============================================================
# run_analysis.sh
# =============================================================
# Post-training analysis: embedding visualization + HD vs 2D
# comparison for a saved model checkpoint.
#
# Edit the variables below, then run:
#   chmod +x run_analysis.sh
#   ./run_analysis.sh
# =============================================================

cd "$(dirname "$0")"

source env/bin/activate

# ----- CONFIGURABLE PARAMETERS -----
MODEL="../models/model_cf612aa0.pth"   # path to model checkpoint
MAX_EPISODE=10                          # number of episodes to analyze
REDUCTION_METHOD="tsne"                 # tsne | umap
JITTER_STRENGTH=0.02                    # jitter for scatter plots (0.0 to 0.1)
# ------------------------------------

echo ""
echo "============================================"
echo "  Prototypical Network Analysis Runner"
echo "============================================"
echo "  model              = $MODEL"
echo "  max_episode        = $MAX_EPISODE"
echo "  reduction_method   = $REDUCTION_METHOD"
echo "  jitter_strength    = $JITTER_STRENGTH"
echo "============================================"
echo ""

python run_analysis.py \
    --model "$MODEL" \
    --max_episode "$MAX_EPISODE" \
    --reduction_method "$REDUCTION_METHOD" \
    --jitter_strength "$JITTER_STRENGTH"
