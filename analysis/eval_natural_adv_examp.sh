python3 \
    analysis/natural_adv_examples/eval.py \
    --arch vit_small \
    --checkpoint ${CHECKPOINT} \
    --data data/imnet_a/imagenet-a \
    ${@:1}

