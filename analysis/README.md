# Analyzing iBOT's Properties

### Part-Wise Linear Probing on ImageNet
```
./run.sh imagenet_part_linear $JOB_NAME vit_small teacher 8
```

### Robustness

The robustness test is done with pre-trained models that linear evaluated for 100 epochs. We first combine the pre-trained models and the linear head with the highest accuracy obtained on the linear probing experiment.

**Combining Pre-trained Backbone with Linear Head** 
```
python analysis/combine_ckpt.py \
  --checkpoint_backbone $PRETRAINED \
  --checkpoint_linear $LINEAR_HEAD \
  --output_file $FULL_LINEAR_MODEL
```

We evaluate the full models in the following aspects:

**Background Change on ImageNet-9** 
```
./analysis/eval_bg_challenge.sh \
  --checkpoint $FULL_LINEAR_MODEL \
  --data-path data/imnet_bg
```

**Natural Adversarial Examples on ImageNet-A** 
```
./analysis/eval_natural_adv_examp.sh \
  --checkpoint FULL_LINEAR_MODEL \
  --data data/imnet_a
```

**Image Corruptions and Surf Variances on ImageNet-C** 
```
./analysis/eval_corr_surf_vari.sh \
  --checkpoint FULL_LINEAR_MODEL \
  --data data/imnet_c
```

**Occlusion & Shuffle** 
```
./analysis/eval_{occlusion,shuffle}.sh \
  --pretrained_weights $FULL_LINEAR_MODEL
```

### Self-Attention Visualization
You can look at the self-attention of the [CLS] token on the different heads of the last layer by running:
```
./analysis/visualize_attn_map.sh \
  --pretrained_weights $PRETRAINED \
  --output_dir $OUTPUT \
  --data_path data/imagenet/val \
  --show_pics 300
```

### Correspondence Visualization

You can extract the correspondence pairs from two randomly augmented views of one image by running:
```
./analysis/visualize_corresp.sh \
  --pretrained_weights $PRETRAINED \
  --arch vit_small \
  --patch_size 16 \
  --data_path data/imagenet/val \
  --sample_type instance \
  ${@:1}
```

To extract the correspondence drawn from two images belonging to the same category, run the following command:
```
./analysis/visualize_corresp.sh \
  --pretrained_weights $PRETRAINED \
  --arch vit_small \
  --patch_size 16 \
  --data_path data/imagenet/val \
  --sample_type class \
  ${@:1}
```