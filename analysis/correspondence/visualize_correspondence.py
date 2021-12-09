# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import utils
import torchvision.transforms.functional as TF
import models

from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler
from loader import ImageFolderInstance

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser("Visualize Correspondence")
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large'], help='Architecture.')
    parser.add_argument('--data_path', default='/path/to/imagenet/val/', type=str, help='Path of the images\' folder.')
    parser.add_argument("--pretrained_weights", type=str, default="", help="the pretraining models")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--output_path", type=str, default='pics/', help="patch for output images")
    parser.add_argument("--topk", type=int, default=12, help="top k matched pairs")
    parser.add_argument("--show_pics", type=utils.bool_flag, default=True)
    parser.add_argument("--show_with_attn", type=utils.bool_flag, default=False)
    parser.add_argument("--show_nums", type=int, default=100)
    parser.add_argument("--sample_type", type=str, default='class', choices=['class', 'instance'], 
        help="""sample pairs from the same class or from one instance (two views)""")
    parser.add_argument("--eval_radius", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # for attention
    attentioner = models.__dict__['vit_base'](patch_size=8, num_classes=0)
    print("We load the reference pretrained DINO weights to extract self-attention for fair comparison.")
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth")
    attentioner.load_state_dict(state_dict, strict=True)
    attentioner.eval()
    attentioner.to(device)

    network = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )
    network.eval()
    network.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")['state_dict']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = network.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))

    if args.sample_type == 'class':
        augmentation = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:

        class RandomResizedCrop(transforms.RandomResizedCrop):
            def forward(self, img):
                i, j, h, w = self.get_params(img, self.scale, self.ratio)
                W, H = TF._get_image_size(img)
                self.corner = np.array([i / H, j / W, (i + h) / H, (j + w) / W])
                return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
            def forward(self, img):
                self.flip = False
                if torch.rand(1) < self.p:
                    self.flip = True
                    return TF.hflip(img)
                return img

        class Augmentation(object):
            def __init__(self, img_size):
                color_jitter = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ])
                normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

                # first global crop
                self.global_transfo1 = transforms.Compose([
                    RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
                    RandomHorizontalFlip(p=0.5),
                    color_jitter,
                    normalize,
                ])
                # second global crop
                self.global_transfo2 = transforms.Compose([
                    RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=Image.BICUBIC),
                    RandomHorizontalFlip(p=0.5),
                    color_jitter,
                    normalize,
                ])

            def __call__(self, image):
                crops = []
                corners = []
                # im1
                crops.append(self.global_transfo1(image))
                corner1 = self.global_transfo1.transforms[0].corner
                if self.global_transfo1.transforms[1].flip:
                    corner1[1], corner1[3] = corner1[3], corner1[1]
                corners.append(corner1)
                # im2
                crops.append(self.global_transfo2(image))
                corner2 = self.global_transfo2.transforms[0].corner
                if self.global_transfo2.transforms[1].flip:
                    corner2[1], corner2[3] = corner2[3], corner2[1]
                corners.append(corner2)
                return crops + corners
        augmentation = Augmentation(args.img_size)

    if args.sample_type == 'class':

        class CustomSampler(Sampler):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                indices = []
                for n in range(self.data.num_classes):
                    index = torch.where(self.data.label == n)[0]
                    indices.append(index)
                indices = torch.cat(indices, dim=0)
                return iter(indices)

            def __len__(self):
                return len(self.data)

        class CustomBatchSampler:
            def __init__(self, sampler, batch_size, shuffle, drop_last):
                self.sampler = sampler
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.drop_last = drop_last

            def __iter__(self):
                batch = []
                i = 0
                sampler_list = list(self.sampler)
                all_list = []
                for idx in sampler_list:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        # yield batch
                        all_list.append(batch)
                        batch = []

                    if (
                        i < len(sampler_list) - 1
                        and self.sampler.data.label[idx]
                        != self.sampler.data.label[sampler_list[i + 1]]
                    ):
                        if len(batch) > 0 and not self.drop_last:
                            # yield batch
                            all_list.append(batch)
                            batch = []
                        else:
                            batch = []
                    i += 1
                if len(batch) > 0 and not self.drop_last:
                    # yield batch
                    all_list.append(batch)
                
                if self.shuffle:
                    random.shuffle(all_list)
                for bs in all_list:
                    yield bs

            def __len__(self):
                if self.drop_last:
                    return len(self.sampler) // self.batch_size
                else:
                    return (len(self.sampler) + self.batch_size - 1) // self.batch_size
                    
        train_dataset = ImageFolderInstance(args.data_path, transform=lambda x: torch.Tensor(0))
        train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)
        labels = np.array([], dtype=np.int)
        for data in tqdm(train_dataloader):
            _, cls, idx = data
            labels = np.concatenate([labels, cls])
        train_dataset.label = torch.Tensor(labels)
        train_dataset.num_classes = labels.max() - labels.min() + 1
        batch_sampler = CustomBatchSampler(CustomSampler(train_dataset), batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_dataset = ImageFolderInstance(args.data_path, transform=augmentation)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    else:
        train_dataset = ImageFolderInstance(args.data_path, transform=augmentation)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    cnt = 0
    acc = []
    for data in tqdm(train_dataloader):
        if  args.sample_type == 'class':
            img, cls, idx = data
            img1 = img.clone()
            img2 = torch.cat([img[1:].clone(), img[0:1].clone()], dim=0)
            ovlp = None
        elif args.sample_type == 'instance':
            (img1, img2, pos1, pos2), cls, idx = data
            unit = args.img_size // args.patch_size
            pos1, pos2 = pos1.cpu().numpy(), pos2.cpu().numpy()
            pos1 = torch.stack((
                torch.from_numpy(np.linspace(pos1[:, 0], pos1[:, 2], num=2*unit+1)[1::2]).unsqueeze(1).expand(-1, unit, -1),
                torch.from_numpy(np.linspace(pos1[:, 1], pos1[:, 3], num=2*unit+1)[1::2]).unsqueeze(0).expand(unit, -1, -1))
                ).flatten(1, 2).permute(2, 1, 0).cuda()
            pos2 = torch.stack((
                torch.from_numpy(np.linspace(pos2[:, 0], pos2[:, 2], num=2*unit+1)[1::2]).unsqueeze(1).expand(-1, unit, -1),
                torch.from_numpy(np.linspace(pos2[:, 1], pos2[:, 3], num=2*unit+1)[1::2]).unsqueeze(0).expand(unit, -1, -1))
                ).flatten(1, 2).permute(2, 1, 0).cuda()
            eps = args.patch_size // 2 / args.img_size
            ovlp = ((pos1 + eps) > pos2.min(dim=1, keepdim=True)[0]).all(dim=-1) & \
                ((pos1 - eps) < pos2.max(dim=1, keepdim=True)[0]).all(dim=-1)
            ovlp = ovlp.view(-1, 1, 14, 14)
            
        with torch.no_grad():             
            idx = idx.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)

            x1 = network(img1)[:, 1:]
            x2 = network(img2)[:, 1:]

            attentions = attentioner.get_last_selfattention(img1.to(device))
        
        attentions = attentions[:, :, 0, 1:].view(x1.size(0), -1, args.img_size // 8, args.img_size // 8)
        attentions1 = nn.functional.interpolate(attentions, scale_factor=0.5, mode="nearest")
        if ovlp is not None:
            attentions1 = attentions1 * ovlp.float()
            attentions = attentions * nn.functional.interpolate(ovlp.float(), scale_factor=2, mode="nearest")
        
        attentions = attentions.mean(1, keepdim=True)
        attentions1 = attentions1.mean(1, keepdim=True)

        sim_matrix = torch.bmm(x1, x2.permute(0, 2, 1))
        value, index = sim_matrix.max(-1)

        if args.sample_type == "instance":
            real_sim_matrix = torch.matmul(pos1, pos2.permute(0, 2, 1))
            real_dst_matrix = pos1.square().sum(2, keepdim=True) + \
                pos2.permute(0, 2, 1).square().sum(1, keepdim=True) - 2 * real_sim_matrix
            real_index = real_dst_matrix.min(dim=2)[1]

            index_h, index_w = index // unit, index % unit
            real_index_h, real_index_w = real_index // unit, real_index % unit
            accuracy = (((index_h - real_index_h).abs() <= args.eval_radius) & ((index_w - real_index_w).abs() <= args.eval_radius) & ovlp.flatten(1)).sum() / ovlp.sum()
            acc.append(accuracy)
        
        if args.show_pics:
            for cs, id, val, im1, im2, attention, attention1, in zip(cls, index, value, img1, img2, attentions, attentions1):

                for nh in range(attention.size(0)):
                    attn = attention[nh]
                    attn1 = attention1[nh].flatten()
                    
                    st = attn1.topk(args.topk)[1]
                    
                    point1 = st
                    point2 = id[st]

                    i1 = torchvision.utils.make_grid(im1, normalize=True, scale_each=True)
                    i1 = Image.fromarray(i1.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    i2 = torchvision.utils.make_grid(im2, normalize=True, scale_each=True)
                    i2 = Image.fromarray(i2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    if args.show_with_attn:
                        img = Image.new('RGB', (args.img_size * 3, args.img_size))
                    else:
                        img = Image.new('RGB', (args.img_size * 2, args.img_size))
                    draw = ImageDraw.Draw(img)
                    unit = args.img_size // args.patch_size
                    if args.show_with_attn:
                        attn = nn.functional.interpolate(attn.detach()[None, None], scale_factor=8, mode="nearest")[0, 0].cpu().numpy()
                        plt.imsave(fname=f'{args.output_path}/temp_attn.png', arr=attn, format='png')
                        img.paste(Image.open(f'{args.output_path}/temp_attn.png'), (0, 0))
                        img.paste(i1, (args.img_size, 0))
                        img.paste(i2, (args.img_size * 2, 0))
                    else:
                        img.paste(i1, (0, 0))
                        img.paste(i2, (args.img_size, 0))

                    for p1, p2 in zip(point1, point2):
                        p1y, p1x = p1 // unit + 0.5, p1 % unit + 0.5
                        p2y, p2x = p2 // unit + 0.5, p2 % unit + 0.5
                        draw.line((p1x * args.patch_size + args.img_size * (1 if args.show_with_attn else 0), 
                                p1y * args.patch_size, 
                                p2x * args.patch_size + args.img_size * (2 if args.show_with_attn else 1), 
                                p2y * args.patch_size), width=1, fill='red')
                    img.save(f'{args.output_path}/{args.sample_type}_corresp{cnt}_cls{cs}_nh{nh}.png')

                cnt += 1
                if cnt >= args.show_nums:
                    exit()

    if args.sample_type == "instance":
        print(sum(acc) / len(acc))

if __name__ == '__main__':
    main()