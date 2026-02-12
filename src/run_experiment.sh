#!/bin/bash
# =============================================================
# run_experiment.sh
# =============================================================
# Edit the variables below, then run:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
# =============================================================

cd "$(dirname "$0")"

source env/bin/activate

# ----- CONFIGURABLE PARAMETERS -----
BACKBONE="vit"              # resnet50 | vit | dinov2
EMBEDDING_DIMS="256 128"         # space-separated, e.g. "512 256" or "256 128"
OPEN_FINAL_LAYER="false"         # true | false  (unfreeze last backbone block)
DATA_AUGMENTATION="true"         # true | false
EPISODES=5                     # 100 | 300  (training episodes per epoch)
# ------------------------------------

echo ""
echo "============================================"
echo "  Prototypical Network Experiment Runner"
echo "============================================"
echo "  backbone             = $BACKBONE"
echo "  embedding_dims       = $EMBEDDING_DIMS"
echo "  open_final_layer     = $OPEN_FINAL_LAYER"
echo "  data_augmentation    = $DATA_AUGMENTATION"
echo "  episodes             = $EPISODES"
echo "============================================"
echo ""

python run_experiment.py \
    --backbone "$BACKBONE" \
    --embedding_dims $EMBEDDING_DIMS \
    --open_final_layer_backbone "$OPEN_FINAL_LAYER" \
    --data_augmentation "$DATA_AUGMENTATION" \
    --episodes "$EPISODES"
