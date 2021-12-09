python analysis/occlusion/eval.py \
    --model_name vit_small \
    --test_data data/imagenet/val \
    --shuffle \
    --shuffle_h 2 2 4 4 8 14 16 \
    --shuffle_w 2 4 4 8 8 14 16 \
    ${@:1}