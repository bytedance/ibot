python analysis/occlusion/eval.py \
    --model_name vit_small \
    --test_data data/imagenet/val \
    --random_drop \
    ${@:1}

python analysis/occlusion/eval.py \
    --model_name vit_small \
    --test_data data/imagenet/val \
    --dino \
    ${@:1}
