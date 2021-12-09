python3 \
    analysis/attention_map/visualize_attention.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path data/imagenet/val \
    ${@:1}