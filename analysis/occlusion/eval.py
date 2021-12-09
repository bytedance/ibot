# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from Intriguing-Properties-of-Vision-Transformers library:
https://github.com/Muzammal-Naseer/Intriguing-Properties-of-Vision-Transformers/blob/main/evaluate.py
"""

import json
import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.utils as vutils
import utils
import dino

from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from timm.utils import accuracy
from tqdm import tqdm
from analysis import imagenet_models
from loader import ImageFolder

def main(args, device, verbose=True):

    # fix the seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    if verbose:
        if args.shuffle:
            print(f"Shuffling inputs and evaluating {args.model_name}")
        elif args.random_drop:
            print(f"{args.model_name} dropping {args.drop_count} random patches")
        elif args.lesion:
            print(f"{args.model_name} dropping {args.drop_count} random patches from block {args.block_index}")
        elif args.cascade:
            print(f"evaluating {args.model_name} in cascade mode")
        elif args.saliency:
            print(f"{args.model_name} dropping {'most' if args.drop_best else 'least'} "
                  f"salient {args.drop_count} patches")
        elif args.saliency_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % most salient pixels")
        elif args.standard_box:
            print(f"{args.model_name} dropping {args.drop_lambda} % pixels around most matching patch")
        elif args.dino:
            print(f"{args.model_name} picking {args.drop_lambda * 100} %  "
                  f"{'foreground' if args.drop_best else 'background'} pixels using dino")
        else:
            print(f"{args.model_name} dropping {'least' if args.drop_best else 'most'} "
                  f"matching {args.drop_count} patches")

    if args.dino:
        # we follow https://arxiv.org/pdf/2105.10497.pdf to use original dino with vit pretrained with 800 epochs
        dino_model = dino.dino_small(patch_size=vars(args).get("patch_size", 16), pretrained=True)
        dino_model.to(device)
        dino_model.eval()

    if "vit" in args.model_name:
        model = imagenet_models.__dict__[args.model_name](num_classes=1000, pretrained=False, linear_eval=args.linear_eval)
        if args.pretrained_weights:
            if os.path.isfile(args.pretrained_weights):
                checkpoint = torch.load(args.pretrained_weights)
                
                # Makes us able to load models saved with legacy versions
                state_dict_path = 'model'
                if not ('model' in checkpoint):
                    state_dict_path = 'state_dict'

                sd = checkpoint[state_dict_path]
                sd = {k[len('module.'):] if ('module.' in k) else k: v for k, v in sd.items()}
                
                msg = model.load_state_dict(sd, strict=False)
                
                print(msg)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrained_weights, checkpoint['epoch']))
            elif args.pretrained_weights == 'supervised':
                url = None
                if args.model_name == "vit_small" and args.patch_size == 16:
                    url = "deit_small_patch16_224-cd65a155.pth"
                elif args.model_name == "vit_base" and args.patch_size == 16:
                    url = "deit_base_patch16_224-b5f2ef4d.pth"
                if url is not None:
                    print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
                    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
                    msg = model.load_state_dict(state_dict['model'], strict=False)
                    print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
            else:
                error_msg = "=> no checkpoint found at '{}'".format(args.pretrained_weights)
                raise ValueError(error_msg)
    else:
        model = imagenet_models.__dict__[args.model_name](num_classes=1000, pretrained=True)

    model.to(device)
    model.eval()

    # print model parameters
    if verbose:
        print(f"Parameters in Millions: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.3f}")

    # Setup-Data
    interpolation = 3 if args.bicubic_resize else 2
    color = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if args.imagenet_default_color else ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    data_transform = transforms.Compose([
        transforms.Resize(args.scale_size, interpolation=interpolation),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(*color),
    ])

    # Test Samples
    test_set = ImageFolder(args.test_data, data_transform)
    test_size = len(test_set)
    if verbose:
        print(f'Test data size: {test_size}')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True)

    clean_acc = 0.0
    pixel_percent = 0.0
    for i, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            img, label = img.to(device), label.to(device)

            if args.shuffle or args.random_drop:
                if isinstance(args.shuffle_size, int):
                    assert 224 % args.shuffle_size == 0, f"shuffle size {args.shuffle_size} " \
                                                         f"not compatible with 224 image"
                    shuffle_h, shuffle_w = args.shuffle_size, args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // args.shuffle_size, 224 // args.shuffle_size
                    patch_num = args.shuffle_size * args.shuffle_size
                else:
                    shuffle_h, shuffle_w = args.shuffle_size
                    patch_dim1, patch_dim2 = 224 // shuffle_h, 224 // shuffle_w
                    patch_num = shuffle_h * shuffle_w

                if args.random_offset_drop:
                    mask = torch.ones_like(img)
                    mask = rearrange(mask, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)
                img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)
                if args.shuffle:
                    row = np.random.choice(range(patch_num), size=img.shape[1], replace=False)
                    img = img[:, row, :]  # images have been shuffled already
                elif args.random_drop and args.drop_count > 0:
                    row = np.random.choice(range(patch_num), size=args.drop_count, replace=False)
                    if args.random_offset_drop:
                        mask[:, row, :] = 0.0
                    else:
                        img[:, row, :] = 0.0
                img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                if args.random_offset_drop and args.drop_count > 0:
                    mask = rearrange(mask, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     h=shuffle_h, w=shuffle_w, p1=patch_dim1, p2=patch_dim2)
                    new_mask = torch.ones_like(mask)
                    mask_off_set = 8
                    new_mask[:, :, mask_off_set:, mask_off_set:] = mask[:, :, :-mask_off_set, :-mask_off_set]
                    img = new_mask * img

            elif args.dino:
                head_number = 1

                attentions = dino_model.forward_selfattention(img.clone())
                attentions = attentions[:, head_number, 0, 1:]

                w_featmap = int(np.sqrt(attentions.shape[-1]))
                h_featmap = int(np.sqrt(attentions.shape[-1]))
                scale = img.shape[2] // w_featmap

                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.drop_lambda)
                idx2 = torch.argsort(idx)
                for batch_idx in range(th_attn.shape[0]):
                    th_attn[batch_idx] = th_attn[batch_idx][idx2[batch_idx]]

                if args.drop_best:
                    percent = th_attn.float().mean(1).sum().item()
                else:
                    percent = (1 - th_attn.float()).mean(1).sum().item()

                th_attn = th_attn.reshape(-1, w_featmap, h_featmap).float()
                th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(1), scale_factor=scale, mode="nearest")
                
                if args.drop_best:  # foreground
                    img = img * (1 - th_attn)
                else:
                    img = img * th_attn

            else:
                pass

            if args.test_image:
                if args.shuffle:
                    if isinstance(args.shuffle_size, int):
                        save_name = args.shuffle_size
                    else:
                        save_name = f"{args.shuffle_size[0]}_{args.shuffle_size[1]}"
                    save_path = f"report/shuffle/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{save_name}.jpg")
                elif args.random_drop:
                    save_path = f"report/random/images"
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/example_{args.drop_count}.jpg")
                elif args.dino:
                    save_path = f"report/dino/images"
                    drop_order = 'foreground' if args.drop_best else 'background'
                    os.makedirs(save_path, exist_ok=True)
                    vutils.save_image(vutils.make_grid(img[:16], normalize=False, scale_each=True),
                                      f"{save_path}/image_{drop_order}_{args.drop_lambda}.jpg")
                else:
                    pass
                return 0

            if args.lesion:
                if "resnet" in args.model_name:
                    clean_out = model(img.clone(), drop_layer=args.block_index,
                                      drop_percent=args.drop_count)
                else:
                    clean_out = model(img.clone(), block_index=args.block_index,
                                      drop_rate=args.drop_count)
            else:
                clean_out = model(img.clone())
                
            if not args.linear_eval and clean_out.size(-1) != 1000:
                clean_out = model.head(clean_out)

            if isinstance(clean_out, list):
                clean_out = clean_out[-1]
            
            clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()
            pixel_percent += percent

    if args.dino:
        print(f"{args.model_name} Top-1 Accuracy: {clean_acc / len(test_set)} with information loss ratio {pixel_percent / len(test_set)}")
        return clean_acc / len(test_set), pixel_percent / len(test_set)

    print(f"{args.model_name} Top-1 Accuracy: {clean_acc / len(test_set)}")
    return clean_acc / len(test_set)


if __name__ == '__main__':
    # opt = parse_args()
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--test_data', default='/path/to/imagenet/val/', type=str,
        help='Please specify path to the ImageNet validation data.')

    parser.add_argument('--exp_name', default=None, help='pretrained weight path')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Model Name')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
    parser.add_argument('--drop_count', type=int, default=180, help='How many patches to drop')
    parser.add_argument('--drop_best', action='store_true', default=False, help="set True to drop the best matching")
    parser.add_argument('--test_image', action='store_true', default=False, help="set True to output test images")
    parser.add_argument('--shuffle', action='store_true', default=False, help="shuffle instead of dropping")
    parser.add_argument('--shuffle_size', type=int, default=14, help='nxn grid size of n', nargs='*')
    parser.add_argument('--shuffle_h', type=int, default=None, help='h of hxw grid', nargs='*')
    parser.add_argument('--shuffle_w', type=int, default=None, help='w of hxw grid', nargs='*')
    parser.add_argument('--random_drop', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--random_offset_drop', action='store_true', default=False, help="randomly drop patches")
    parser.add_argument('--cascade', action='store_true', default=False, help="run cascade evaluation")
    parser.add_argument('--exp_count', type=int, default=1, help='random experiment count to average over')
    parser.add_argument('--saliency', action='store_true', default=False, help="drop using saliency")
    parser.add_argument('--saliency_box', action='store_true', default=False, help="drop using saliency")
    parser.add_argument('--drop_lambda', type=float, default=0.2, help='percentage of image to drop for box')
    parser.add_argument('--standard_box', action='store_true', default=False, help="drop using standard model")
    parser.add_argument('--dino', action='store_true', default=False, help="drop using dino model saliency")

    parser.add_argument('--lesion', action='store_true', default=False, help="drop using dino model saliency")
    parser.add_argument('--block_index', type=int, default=0, help='block index for lesion method', nargs='*')

    parser.add_argument('--draw_plots', action='store_true', default=False, help="draw plots")
    parser.add_argument('--select_im', action='store_true', default=False, help="select robust images")
    parser.add_argument('--save_path', type=str, default=None, help='save path')

    # segmentation evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold for segmentation')
    parser.add_argument('--pretrained_weights', default=None, help='pretrained weights path')
    parser.add_argument('--patch_size', type=int, default=16, help='nxn grid size of n')
    parser.add_argument('--use_shape', action='store_true', default=False, help="use shape token for prediction")
    parser.add_argument('--rand_init', action='store_true', default=False, help="use randomly initialized model")
    parser.add_argument('--generate_images', action='store_true', default=False, help="generate images instead of eval")

    # use linear eval model or finetune model
    parser.add_argument('--linear_eval', type=utils.bool_flag, default=True, help='Wether to use linear eval model or finetune model')
    parser.add_argument('--imagenet_default_color', type=utils.bool_flag, default=True, help="""We do not use default imnet color for 
        finetuned models from BEiT repo.""")
    parser.add_argument('--bicubic_resize', type=utils.bool_flag, default=False, help="""We use bicubic interpolation to resize for 
        finetuned models from BEiT repo.""")

    opt = parser.parse_args()

    acc_dict = {}

    if opt.shuffle:
        if opt.shuffle_h is not None:
            assert opt.shuffle_w is not None, "need to specify both shuffle_h and shuffle_w!"
            assert len(opt.shuffle_h) == len(opt.shuffle_w), "mismatch for shuffle h, w pairs"
            shuffle_list = list(zip(opt.shuffle_h, opt.shuffle_w))
        else:
            shuffle_list = opt.shuffle_size
        if isinstance(shuffle_list, int):
            shuffle_list = [shuffle_list, ]
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for shuffle_size in shuffle_list:
                opt.shuffle_size = shuffle_size
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                if isinstance(shuffle_size, tuple):
                    shuffle_size = shuffle_size[0] * shuffle_size[1]
                acc_dict[f"run_{rand_exp:03d}"][f"{shuffle_size}"] = acc
        if not opt.test_image:
            print(acc_dict)
            #json.dump(acc_dict, open(f"report/shuffle/{opt.model_name}.json", "w"), indent=4)

    elif opt.random_drop:
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            for drop_count in range(0, 10):
                if isinstance(opt.shuffle_size, list):
                    opt.drop_count = drop_count * opt.shuffle_size[0] * opt.shuffle_size[1] // 10
                else:
                    opt.drop_count = drop_count * 196 // 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"run_{rand_exp:03d}"][f"{drop_count}"] = acc
        if not opt.test_image:
            if isinstance(opt.shuffle_size, list):
                shuffle_name = f"_{opt.shuffle_size[0]}_{opt.shuffle_size[1]}"
            else:
                if opt.exp_name is None:
                    shuffle_name = ""
                else:
                    shuffle_name = f"_{opt.exp_name}"
            print(acc_dict)
            #json.dump(acc_dict, open(f"report/random/{opt.model_name}{shuffle_name}.json", "w"), indent=4)

    elif opt.dino:
        for drop_best in [True, False]:
            opt.drop_best = drop_best
            acc_dict[f"{'best' if opt.drop_best else 'worst'}"] = {}
            for drop_lambda in range(1, 11):
                opt.drop_lambda = drop_lambda / 10
                acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                acc_dict[f"{'best' if opt.drop_best else 'worst'}"][f"{drop_lambda}"] = acc
        if not opt.test_image:
            print(acc_dict)
            #json.dump(acc_dict, open(f"report/dino/{opt.model_name}.json", "w"), indent=4)

    elif opt.lesion:
        for rand_exp in range(opt.exp_count):
            acc_dict[f"run_{rand_exp:03d}"] = {}
            block_index_list = opt.block_index
            for cur_block_num in block_index_list:
                opt.block_index = cur_block_num
                acc_dict[f"run_{rand_exp:03d}"][f"{cur_block_num}"] = {}
                for drop_count in [0.25, 0.50, 0.75]:
                    opt.drop_count = drop_count
                    acc = main(args=opt, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    acc_dict[f"run_{rand_exp:03d}"][f"{cur_block_num}"][f"{drop_count}"] = acc
        if not opt.test_image:
            print(acc_dict)
            #json.dump(acc_dict, open(f"report/lesion/{opt.model_name}.json", "w"), indent=4)

    else:
        print("No arguments specified: finished running")