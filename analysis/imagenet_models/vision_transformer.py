# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from vision_transformer import VisionTransformer

__all__ = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']

class VisionTransformerLinearEval(VisionTransformer):

    def __init__(self, num_classes, n_last_blocks=4, avgpool_patchtokens=0, **kwargs):
        super(VisionTransformerLinearEval, self).__init__(num_classes=0, use_mean_pooling=avgpool_patchtokens, **kwargs)
        self.n = n_last_blocks
        self.a = avgpool_patchtokens
        self.linear = nn.Linear(self.embed_dim * (self.n * int(self.a != 1) + int(self.a > 0)), num_classes)

    def forward(self, x):
        intermediate_output = self.get_intermediate_layers(x, self.n)
        if self.a == 0:
            # norm(x[:, 0])
            output = [x[:, 0] for x in intermediate_output]
        elif self.a == 1:
            # x[:, 1:].mean(1)
            output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        elif self.a == 2:
            # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
            output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        else:
            assert False, "Unkown avgpool type {}".format(self.a)
        output = torch.cat(output, dim=-1)
        
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output


def vit_tiny(pretrained, patch_size=16, linear_eval=True, **kwargs):
    if linear_eval:
        model = VisionTransformerLinearEval(
            patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
            qkv_bias=True, n_last_blocks=4, avgpool_patchtokens=0, **kwargs)
    else:
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
            qkv_bias=True, **kwargs)
    return model


def vit_small(pretrained, patch_size=16, linear_eval=True, **kwargs):
    if linear_eval:
        model = VisionTransformerLinearEval(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, n_last_blocks=4, avgpool_patchtokens=0, **kwargs)
    else:
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, **kwargs)
    return model


def vit_base(pretrained, patch_size=16, linear_eval=True, **kwargs):
    if linear_eval:
        model = VisionTransformerLinearEval(
            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, n_last_blocks=1, avgpool_patchtokens=2, **kwargs)
    else:
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, **kwargs)
    return model


def vit_large(pretrained, patch_size=16, linear_eval=True, **kwargs):
    if linear_eval:
        model = VisionTransformerLinearEval(
            patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
            qkv_bias=True, n_last_blocks=1, avgpool_patchtokens=2, **kwargs)
    else:
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
            qkv_bias=True, **kwargs)  
    return model
