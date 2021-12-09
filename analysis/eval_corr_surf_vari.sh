python3 \
    analysis/corruptions_surf_variations/eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data data/imnet_c \
    ${@:1}

