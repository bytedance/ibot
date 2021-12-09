# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from backgrounds_challenge library:
https://github.com/MadryLab/backgrounds_challenge
"""

import torch.nn as nn
import numpy as np
import json
import os
import time

from torchvision import transforms
from argparse import ArgumentParser
from PIL import Image
from tools.datasets import ImageNet, ImageNet9
from tools.model_utils import make_and_restore_model, adv_bgs_eval_model

parser = ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--arch', default='resnet50',
                    help='Model architecture, if loading a model checkpoint.')
parser.add_argument('--checkpoint', default=None,
                    help='Path to model checkpoint.')
parser.add_argument('--data-path', required=True,
                    help='Path to the eval data')
parser.add_argument('--in9', dest='in9', default=False, action='store_true',
                    help='Enable if the model has 9 output classes, like in IN-9')

def main(args):
    map_to_in9 = {}
    root = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    with open(f'{root}/in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    BASE_PATH_TO_EVAL = args.data_path

    # Load model
    in9_trained = args.in9
    arch = args.arch
    if in9_trained:
        train_ds = ImageNet9('/tmp')
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

    # Load backgrounds
    bg_ds = ImageNet9(f'{BASE_PATH_TO_EVAL}/only_bg_t')
    bg_loader = bg_ds.make_loaders(batch_size=args.batch_size, workers=8)

    # Load foregrounds
    fg_mask_base = f'{BASE_PATH_TO_EVAL}/fg_mask/val'
    class_names = sorted(os.listdir(f'{fg_mask_base}'))
    def get_fgs(classnum):
        classname = class_names[classnum]
        return sorted(os.listdir(f'{fg_mask_base}/{classname}'))

    total_vulnerable = 0
    total_computed = 0
    # Big loop
    for fg_class in range(9):

        fgs = get_fgs(fg_class)
        fg_classname = class_names[fg_class]

        # Evaluate model
        prev_time = time.time()
        for i in range(len(fgs)):
            if total_computed % 50 == 0:
                cur_time = time.time()
                print(f'At image {i} for class {fg_classname}, used {(cur_time-prev_time):.2f} since the last print statement.')
                print(f'Up until now, have {total_vulnerable}/{total_computed} vulnerable foregrounds.')
                prev_time = cur_time

            mask_name = fgs[i]
            fg_mask_path = f'{fg_mask_base}/{fg_classname}/{mask_name}'
            fg_mask = np.load(fg_mask_path)
            fg_mask = np.tile(fg_mask[:, :, np.newaxis], [1, 1, 3]).astype('uint8')
            fg_mask = transforms.ToTensor()(Image.fromarray(fg_mask*255))
            
            img_name = mask_name.replace('npy', 'JPEG')
            image_path = f'{BASE_PATH_TO_EVAL}/original/val/{fg_classname}/{img_name}'
            img = transforms.ToTensor()(Image.open(image_path))

            is_adv = adv_bgs_eval_model(bg_loader, model, img, fg_mask, fg_class, args.batch_size, map_to_in9, map_in_to_in9=(not in9_trained))
            print(f'Image {i} of class {fg_classname} is {is_adv}.')
            total_vulnerable += is_adv
            total_computed += 1

    print('Evaluation complete')
    percent_vulnerable = total_vulnerable/total_computed * 100
    print(f'Summary: {total_vulnerable}/{total_computed} ({percent_vulnerable:.2f}%) are vulnerable foregrounds.')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

