# Evaluating iBOT on Downstream Tasks

### k-NN Classification & Logistic Regression on ImageNet
To evaluate k-NN classification or logistic regression on the frozen features, run:
```
./run.sh imagenet_{knn,reg} $JOB_NAME vit_{small,base} teacher 8
```

### Linear Probing on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 8 gpus, run the following commands. It will train multiple classifiers with different learning rate simutaneously. This will automatically does the job for learning rate sweeping.
```
./run.sh imagenet_linear $JOB_NAME vit_base teacher 8
```
To train a single classifier on frozen weights with customized learning rate, run:
```
./run.sh imagenet_linear_solo $JOB_NAME vit_base teacher 8 --lr $LR
```

Note: `LINEAR_AVGPOOL` and `LINEAR_N_LAST_BLOCKS` control the specific feature design for linear evaluation following DINO. They are illustrated in arguments configuration and automatically set according to `ARCH`. See [run.sh](https://github.com/bytedance/ibot/blob/main/run.sh) for details.

### Fine-Tuning on ImageNet

To fine-tune the pre-trained model, we apply layerwise decay and sweep the learning rate. 

To train ViT-S/16 with 200 epochs, run:
```
./run.sh imagenet_cls $JOB_NAME vit_small teacher 8 \
  --epochs 200 \
  --drop_path 0.1 \
  --layer_decay 0.75
```
To train ViT-B/16 with 100 epochs, run:
```
./run.sh imagenet_cls $JOB_NAME vit_base teacher 8 \
  --epochs 100 \
  --drop_path 0.2 \
  --layer_decay 0.65
```
To train ViT-L/16 with 50 epochs, run:
```
./run.sh imagenet_cls $JOB_NAME vit_large teacher 8 \
  --epochs 50 \
  --drop_path 0.4 \
  --layer_decay 0.75 \
  --batch_size 64 \
  --enable_deepspeed \
  --warmup_epochs 5 \
  --update_freq 2
```

### Unsupervised Classification on ImageNet
To evaluate for unsupervised classification, run:
```
/run.sh imagenet_unsup_cls $JOB_NAME vit_{small,base} teacher 8
``` 
Note: To ensure one-to-one assignment, the output dimension of projection head for [CLS] token (also patch tokens for iBOT) should be set to 1000 during pre-training. We here share this pre-trained [model](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16_out_dim_1000/checkpoint.pth) with its [args](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16_out_dim_1000/args.txt).

### Semi-Supervised Classification on ImageNet

For semi-supervsied classification, we use the data split defined in SimCLRv2, see [here](https://github.com/google-research/simclr/tree/master/imagenet_subsets). For settings evaluated on fronzen features, k-NN, LR, and linear probing, just change the `--data_path` to the imagenet splits from the above commands with full data. For end-to-end fine-tuning, we fine-tuning the pre-trained model from the first layer of the projection head:
```
./run.sh imagenet_semi_cls $JOB_NAME vit_small teacher 8 \
  --epochs 1000 \
  --lr 5e-6 \
  --data_path data/imagenet_{1,10}p_split \
  --finetune_head_layer 1
```

### Object Detection and Instance Segmentation on COCO

To train ViT-S/16 with Cascaded Mask R-CNN as the task layer, run:
```
./run.sh coco_det $JOB_NAME vit_small teacher 8 \
  data.samples_per_gpu=4 \
  lr_config.step=8,11 \
  runner.max_epochs=12 \
  optimizer.paramwise_cfg.layer_decay_rate=0.8
```

To train ViT-B/16 with Cascaded Mask R-CNN as the task layer, run: 
```
./run.sh coco_det $JOB_NAME vit_base teacher 8 \
  data.samples_per_gpu=2 \
  lr_config.step=8,11 \
  runner.max_epochs=12 \
  optimizer.paramwise_cfg.layer_decay_rate=0.75
```

### Semantic Segmentation on ADE20K

To train ViT-S/16 with UperNet as the task layer, run:
```
./run.sh ade20k_seg $JOB_NAME vit_small teacher 4 \
  data.samples_per_gpu=4 \
  model.backbone.out_with_norm=true \
  optimizer.lr=3e-5
```

To train ViT-B/16 with fixed backbone and linear head as the task layer, run:
```
./run.sh ade20k_dense_linear $JOB_NAME vit_base teacher 4 \
  data.samples_per_gpu=4 \
  model.backbone.out_with_norm=true \
  optimizer.lr=8e-4
```

To train ViT-B/16 with UperNet as the task layer, run:
```
./run.sh ade20k_seg $JOB_NAME vit_base teacher 8 \
  data.samples_per_gpu=2 \
  model.backbone.out_with_norm=true \
  optimizer.lr=8e-5
```

### Transfer Learning on Smaller Datasets

For historical issues and reproductivity, we use the default default fine-tuning recipe (i.e., w/o layerwise decay, a smaller learing rate, and a longer training scheduler) proposed in DEiT.

The default configuration in [run.sh](https://github.com/bytedance/ibot/blob/main/run.sh) is for ViT-B/16, and just one-line command is easy to go:
```
./run.sh cifar10_cls+cifar_cls+cars_cls+flwrs_cls+inat_cls+inat19_cls $JOB_NAME vit_base teacher 8
```
Note: ViT-S/16 shares most of the configuration, except that we set the `--lr` as 5e-5 for INAT18 dataset and 2.5e-5 for INAT19 dataset. 

### Nearest Neighbor Retrieval

Nerest neighbor retrieval is considered using the frozen pre-trained features following the evaluation protocol as DINO. We consider three settings:

**Video Object Segmentation on DAVIS**  
```
./run.sh davis_viseg $JOB_NAME vit_small teacher 1 \
  --data_path data/davis \
  --output_dir '.viseg/'
```

**Image Retrieval on Paris and Oxford** 
```
./run.sh paris_oxford_reid $JOB_NAME vit_small teacher 1 \
  --data_path data/revisited_paris_oxford
```

**Copy Detection on Copydays** 
```
./run.sh copydays_copydey $JOB_NAME vit_small teacher 1 \
  --data_path data/copydays
```