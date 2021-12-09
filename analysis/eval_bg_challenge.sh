python3 \
    analysis/backgrounds_challenge/challenge_eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data-path data/imnet_bg/bg_challenge \
    ${@:2}

python3 \
    analysis/backgrounds_challenge/in9_eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data-path data/imnet_bg/bg_challenge \
    ${@:2}
