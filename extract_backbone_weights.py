# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    parser.add_argument("--checkpoint_key", default="state_dict", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--with_head", type=utils.bool_flag, default=False, help='extract checkpoints w/ or w/o head")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict())
    has_backbone = False
    for key, value in ck[args.checkpoint_key].items():
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
        elif key.startswith('module.backbone'):
            output_dict['state_dict'][key[16:]] = value
            has_backbone = True
        elif args.with_head:
            output_dict['state_dict'][key] = value
    if not has_backbone:
        # raise Exception("Cannot find a backbone module in the checkpoint.")
        print("Cannot find a backbone module in the checkpoint. No modification.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()