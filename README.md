# Image BERT Pre-Training with iBOT <img width="32" alt="iBOT Icon" src=".github/ibot.png">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ibot-image-bert-pre-training-with-online/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=ibot-image-bert-pre-training-with-online)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ibot-image-bert-pre-training-with-online/self-supervised-image-classification-on-1)](https://paperswithcode.com/sota/self-supervised-image-classification-on-1?p=ibot-image-bert-pre-training-with-online)

Official PyTorch implementation and pre-trained models for paper **iBOT: Image BERT Pre-Training with Online Tokenizer**. 

[[`arXiv`](https://arxiv.org/abs/2111.07832)] [[`Colab`](https://colab.research.google.com/github/bytedance/ibot/blob/main/notebooks/iBOT_demo.ipynb)] [[`BibTex`](https://github.com/bytedance/ibot#citing-ibot)]

<div align="center">
  <img width="90%" alt="iBOT framework" src=".github/framework.png">
</div>

iBOT is a novel self-supervised pre-training framework that performs masked image modeling with self-distillation. iBOT pre-trained model shows local semantic features, which helps the model transfer well to downstream tasks both at a global scale and a local scale. For example, iBOT achieves strong performance on COCO object detection (**51.2 box AP** and **44.2 mask AP**) and ADE20K semantic segmentation (**50.0 mIoU**) with vanilla ViT-B/16. iBOT can also extract semantic-meaningful local parts, like **dog's ear :dog:**.


## Update :tada:
- Update - ViT-B/16 with random masking and a relatively larger prediction ratio [0.65, 0.75] perform slighly better than block-wise masking with the ratio [0.1, 0.5]. For example, this model can achieve an **84.0%** accuracy in ImageNet-1K fine-tuning and a **51.5 box AP** in COCO object detection.
- December 2021 - Release the code and pre-trained [models](https://github.com/bytedance/ibot#pre-trained-models).
- November 2021 - Release the pre-print on [arXiv](https://arxiv.org/abs/2111.07832).

## Installation

See [installation structions](https://github.com/bytedance/ibot/blob/main/INSTALL.md) for details.

## Training

For a glimpse at the full documentation of iBOT pre-training, please run:
```
python main_ibot.py --help
```

### iBOT Pre-Training with ViTs

To start the iBOT pre-training with Vision Transformer (ViT), simply run the following commands. `JOB_NAME` is a customized argument to distinguish different experiments and this will automatically save checkpoints into the seperate folders.
```
./run.sh imagenet_pretrain $JOB_NAME vit_{small,base,large} teacher {16,24,64}
```
The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the pre-trained [models](https://github.com/bytedance/ibot#pre-trained-models). We also provide the logs for pre-training to help reproducibility.

For example, run iBOT with ViT-S/16 network on two nodes with 8 GPUs for 800 epochs with the following command. The resulting checkpoint should reach 75.2% on k-NN accuracy, 77.9% on linear probing accuracy, and 82.3% on fine-tuning accuracy.

```
./run.sh imagenet_pretrain $JOB_NAME vit_small teacher 16 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer false \
  --epochs 800 \
  --batch_size_per_gpu 64 \
  --shared_head true \
  --out_dim 8192 \
  --local_crops_number 10 \
  --global_crops_scale 0.25 1 \
  --local_crops_scale 0.05 0.25 \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2
```

### iBOT Pre-Training with Swins
This code also works for training iBOT on Swin Transformer (Swin). In the paper, we only conduct experiments on Swin-T with different window sizes:
```
./run.sh imagenet_pretrain $JOB_NAME swin_tiny teacher {16,40} \
  --patch_size 4 \
  --window_size {7,14}
```

For example, run iBOT with Swin-T/14 network on five nodes with 8 GPUS for 300 epochs with the following command. The resulting checkpoint should reach 76.2% on k-NN accuracy, 79.3% on linear probing accuracy.

```
./run.sh imagenet_pretrain $JOB_NAME swin_tiny teacher 40 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer false \
  --epochs 300 \
  --batch_size_per_gpu 26 \
  --shared_head true \
  --out_dim 8192 \
  --local_crops_number 10 \
  --global_crops_scale 0.25 1 \
  --local_crops_scale 0.05 0.25 \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2 \
  --pred_start_epoch 50 \
  --patch_size 4 \
  --window_size 14 
```

## Pre-Trained Models

You can choose to download only the weights of the pre-trained `backbone` used for downstream tasks, and the `full ckpt` which contains backbone and projection head weights for both student and teacher networks. For the `backbone`, `s` denotes that the student network is selected while `t` denotes that the teacher network is selected. `PS` denotes prediction shape.

<table>
  <tr>
    <th>Arch.</th>
    <th>Par.</th>
    <th>PS</th>
    <th>k-NN</th>
    <th>Lin.</th>
    <th>Fin.</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>Block</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td>82.3%</td>
    <td><a href="https://drive.google.com/file/d/1di_xSqKxEwp7TFkis8fWkhYOYH1PagkH/view?usp=sharing">backbone (t)</a></td>
    <td><a href="https://drive.google.com/file/d/1IE6_NeborP5GQa0kufn2tdTk7lrg5QYd/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1RUE2jHeR1HntHTWvZS69Hl6XuvuhdYGR/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/11J3NW9WJ6TPzmS_fy_wlth6S2IDTmmi1/view?usp=sharing">logs</a></td>
  </tr>
  <tr>
    <td>Swin-T/7</td>
    <td>28M</td>
    <td>Block</td>
    <td>75.3%</td>
    <td>78.6%</td>
    <td>\</td>
    <td><a href="https://drive.google.com/file/d/17gMXk9yUVk03lkgVoFXY0bgR3WOjdzSE/view?usp=sharing">backbone (t)</a></td>
    <td><a href="https://drive.google.com/file/d/13WdtV0U9get-tqb4TzCEyJ_6P02MWtkW/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1RtBqcUKqFeGPnYJA3KVMtqyFHimvBDCD/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1OFeq5f7ZV2zBVZjDubOqcoBQOTCdBXZf/view?usp=sharing">logs</a></td>
  </tr>
  <tr>
    <td>Swin-T/14</td>
    <td>28M</td>
    <td>Block</td>
    <td>76.2%</td>
    <td>79.3%</td>
    <td>\</td>
    <td><a href="https://drive.google.com/file/d/1vGyXH_DtyNNukm63z_6IJ1FvTlj6BnT9/view?usp=sharing">backbone (t)</a></td>
    <td><a href="https://drive.google.com/file/d/15cSbRpB6dmwlyF3hXPVVTO0EN7R3r_hw/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1iKYmhcznn5TFnXaK2ZqHtpqIqhWYgDUx/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1zvRHveoiP0t6qH42N1A78tuD8JxHr0V8/view?usp=sharing">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Block</td>
    <td>77.1%</td>
    <td>79.5%</td>
    <td>83.8%</td>
    <td><a href="https://drive.google.com/file/d/1JgdVNX0zjYy9AoUEZO0BILOlFVH-1Vfu/view?usp=sharing">backbone (t)</a></td>
    <td><a href="https://drive.google.com/file/d/1bAiCA4UthX12kzzrG16FCj-BKYluoyY_/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1p3vZNBVhKf_i_Y_Zveai5lIP5YD422n0/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1UFYSd4o7yQXM5sRO75gAzyliEwezkae5/view?usp=sharing">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Rand</td>
    <td>77.3%</td>
    <td>79.8%</td>
    <td>84.0%</td>
    <td><a href="https://drive.google.com/file/d/1Ffgb0gZgoDma9JjcMA5FRdtbgc3OlJ8p/view?usp=sharing">backbone (t)</a></td>
    <td><a href="https://drive.google.com/file/d/1mRnI99p0l02LPSBcLbDIvJMqICFHaw9z/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1qgoN_NgHCmfMiwjyfwMhRIYirqbSPu1H/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1qC-lXpCvatWDraT9IgJmbqXPMYNqdv9W/view?usp=sharing">logs</a></td>
  </tr>
</table>

We also provide the ViT-{B,L}/16 model pre-trained on ImageNet-22K dataset.

 <table>
  <tr>
    <th>Arch.</th>
    <th>Par.</th>
    <th>PS</th>
    <th>k-NN</th>
    <th>Lin.</th>
    <th>Fin.</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>Block</td>
    <td>71.1%</td>
    <td>79.0%</td>
    <td>84.4%</td>
    <td><a href="https://drive.google.com/file/d/1djICe-Q9B7WPy_VOk8qtYRGc06wyp5Ct/view?usp=sharing">backbone (s)</a></td>
    <td><a href="https://drive.google.com/file/d/1p_2xPf___1XOwvtHfRzXH9TfMhaObNDy/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1MGRkC8NEbK2aGA_iLzdDzJPSGBD-n6Ap/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1PbHXCewcdxrxGsJ-4zZc8oiTlFIXzkjO/view?usp=sharing">logs</a></td>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>307M</td>
    <td>Block</td>
    <td>70.6%</td>
    <td>81.7%</td>
    <td>86.3%</td>
    <td><a href="https://drive.google.com/file/d/1wmTENIXLy4JlzG-HoKUE3OZBJuPPA-Vm/view?usp=sharing">backbone (s)</a></td>
    <td><a href="https://drive.google.com/file/d/1J4vEXLoZHGhu-fxCsUC3rpPoU_4izNxl/view?usp=sharing">full ckpt</a></td>
    <td><a href="https://drive.google.com/file/d/1kVRhIk1FNggIwFouZyhxUNQ1efJHhti2/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1dJZfLyx6OSHjCCaipoh4jZfESgzmWVnL/view?usp=sharing">logs</a></td>
  </tr>
</table>

To extract the backbone from the full checkpoint by yourself, please run the following command where `KEY` being either student or teacher.
```
WEIGHT_FILE=$OUTPUT_DIR/checkpoint_$KEY.pth

python extract_backbone_weights.py \
  --checkpoint_key $KEY \
  $PRETRAINED \
  $WEIGHT_FILE \
```

## Downstream Evaluation

See [Evaluating iBOT on Downstream Tasks](https://github.com/bytedance/ibot/blob/main/evaluation/README.md) for details.

## Property Analysis

See [Analyzing iBOT's Properties](https://github.com/bytedance/ibot/blob/main/analysis/README.md) for robustness test and visualizing self-attention map:
<div align="center">
  <img width="100%" alt="iBOT Global Pattern Layout" src=".github/attnmap.png">
</div>

or extracting sparse correspondence pairs between two images: 
<div align="center">
  <img heigh="85%" width="75%" alt="iBOT Global Pattern Layout" src=".github/corresp.png">
</div>

We also provide a [Colab page](https://colab.research.google.com/github/bytedance/ibot/blob/main/notebooks/iBOT_demo.ipynb) :bookmark_tabs: you can play around with iBOT pre-trained models.

## Extracting Semantic Patterns

We extract top-k numbered local classes based on patch tokens with their corresponding patches and contexts by running the following command. We indentify very diverse behaviour like shared **low-level textures** and **high-level semantics**.
```
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${MASTER_PORT:-29500} \
    analysis/extract_pattern/extract_topk_cluster.py \
    --pretrained_path $PRETRAINED \
    --checkpoint {student,teacher} \
    --type patch \
    --topk 36 \
    --patch_window 5 \
    --show_pics 20 \
    --arch vit_small \
    --save_path memory_bank_patch.pth \
    --data_path data/imagenet/val
```
<div align="center">
  <img width="100%" alt="iBOT Local Part-Level Pattern Layout" src=".github/local_semantic_parts.png">
</div>

The script also supports to extract the patern layout on the [CLS] token, which is actually doing clustering or unsupervised classification. This property is not induced by MIM objective since we also spot this feature on DINO.

```
python3 -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=${MASTER_PORT:-29500} \
    analysis/extract_pattern/extract_topk_cluster.py \
    --pretrained_path $PRETRAINED \
    --checkpoint {student,teacher} \
    --type cls \
    --topk 36 \
    --show_pics 20 \
    --arch vit_small \
    --save_path memory_bank_cls.pth \
    --data_path data/imagenet/val
```
<div align="center">
  <img width="75%" alt="iBOT Global Pattern Layout" src=".github/global_semantics.png">
</div>


## Acknowledgement

This repository is built using the [DINO](https://github.com/facebookresearch/dino) repository and the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citing iBOT
If you find this repository useful, please consider giving a star :star: and citation:
```
@article{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  journal={arXiv preprint arXiv:2111.07832},
  year={2021}
}
```
