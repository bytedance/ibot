python3 -m torch.distributed.launch --nproc_per_node=8 \
    analysis/linear_part/eval.py \
    --arch vit_small \
    --output_dir .linear_part \
    --data_path data/imagenet \
    ${@:1}