python3 \
    analysis/correspondence/visualize_correspondence.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path data/imagenet/val \
    ${@:1}