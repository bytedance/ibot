# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from backgrounds_challenge library:
https://github.com/MadryLab/backgrounds_challenge
"""

import json
import os
import copy
import torch.nn as nn

from argparse import ArgumentParser
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, eval_model

parser = ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--arch', default='resnet50',
                    help='Model architecture, if loading a model checkpoint.')
parser.add_argument('--checkpoint', default=None,
                    help='Path to model checkpoint.')
parser.add_argument('--data-path', required=True,
                    help='Path to the eval data')
parser.add_argument('--eval-dataset', default='original',
                    help='What IN-9 variation to evaluate on.')
parser.add_argument('--in9', dest='in9', default=False, action='store_true',
                    help='Enable if the model has 9 output classes, like in IN-9')


def main(args):
    map_to_in9 = {}
    root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    with open(f'{root}/in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    # Load eval dataset
    variation = args.eval_dataset
    in9_ds = ImageNet9(f'{args.data_path}/{variation}')
    val_loader = in9_ds.make_loaders(batch_size=args.batch_size, workers=8)

    # Load model
    in9_trained = args.in9
    arch = args.arch
    if in9_trained:
        train_ds = in9_ds
    else:
        train_ds = ImageNet('/tmp')
    checkpoint = args.checkpoint
    if checkpoint is None:
        model, _ = make_and_restore_model(arch=arch, dataset=train_ds,
                     pytorch_pretrained=True)
    else:
        model, _ = make_and_restore_model(arch=arch, dataset=train_ds,
                     resume_path=checkpoint)
    model.cuda()
    model.eval()
    model = nn.DataParallel(model, device_ids=[0])

    # Evaluate model
    prec1 = eval_model(val_loader, model, map_to_in9, map_in_to_in9=(not in9_trained))
    print(f'Accuracy on {variation} is {prec1*100:.2f}%')


if __name__ == "__main__":
    args = parser.parse_args()
    for var in sorted(os.listdir(args.data_path))[1:]:
        args_new = copy.deepcopy(args)
        args_new.eval_dataset = var
        main(args_new)