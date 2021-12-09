# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from beit, timm, mmseg, setr, xcit and swin code bases
https://github.com/microsoft/unilm/tree/master/beit
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/fudan-zvg/SETR
https://github.com/facebookresearch/xcit/
https://github.com/microsoft/Swin-Transformer
"""

_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
find_unused_parameters = True

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=[512],
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_fpn=False,
        frozen_stages=12,
        out_indices=[3, 5, 7, 11]
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=1536,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
# optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

optimizer = dict(_delete_=True, type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)

runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)