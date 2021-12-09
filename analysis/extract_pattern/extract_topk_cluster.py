# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.distributed as dist
import numpy as np
import models
import utils

from PIL import ImageFile
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from models.head import iBOTHead
from PIL import Image, ImageDraw
from loader import ImageFolderInstance

ImageFile.LOAD_TRUNCATED_IMAGES = True

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def main():
    parser = argparse.ArgumentParser("The first stage of BoostrapSelfSup")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed parallel')
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101'], help='Architecture.')
    parser.add_argument('--data_path', default='/path/to/imagenet/val/', type=str,
        help='Please specify path to the ImageNet validation data.')
    parser.add_argument("--pretrained_path", type=str, default="", help="the pretraining models")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help="""Key to use in the checkpoint 
        (Default: teacher)""")
    parser.add_argument("--save_path", type=str, default="", help="where to save the memory_bank")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_window", type=int, default=5, help="patch visualize window")
    parser.add_argument("--out_dim", type=int, default=8192, help="out_dim")
    parser.add_argument("--type", type=str, default='patch', choices=['cls', 'patch'], help="""wether to visualize
        patterns on patch level or cls level.""")
    parser.add_argument("--topk", type=int, default=196, help="topk")
    parser.add_argument("--show_pics", type=int, default=100, help="show pics of topk cluster with most items")
    parser.add_argument("--chunks", type=int, default=16, help="""Number of counting chunks. Set this larger (e.g., 128
        for DINO w/ 65536 out dim) when the model output dimension is large to avoid memory overflow.""")
    args = parser.parse_args()
    
    pretrained_path = os.path.expanduser(args.pretrained_path)
    save_path = os.path.expanduser(args.save_path)
    batch_size = args.batch_size

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)


    network = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )
    network = utils.MultiCropWrapper(network, iBOTHead(
        network.embed_dim,
        args.out_dim,
        patch_out_dim=args.out_dim,
        act='gelu',
        shared_head=True,
    ))
    network.cuda(args.local_rank)

    try:
        utils.restart_from_checkpoint(pretrained_path, **{args.checkpoint_key: network})
    except:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.local_rank])
        utils.restart_from_checkpoint(pretrained_path, **{args.checkpoint_key: network})

    cudnn.benchmark = True

    augmentation = transforms.Compose([
        transforms.Resize(args.img_size // 7 * 8),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = ImageFolderInstance(args.data_path, transform=augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, rank=args.local_rank)

    n_train_points = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=10)
    
    try:
        data = torch.load(os.path.join(args.save_path, f'memory_{args.type}.pth'))
        memory_bank = data['memory_bank']
        num_per_cluster = data.get('num_per_cluster', None)
    except:
        memory_bank = None
        num_per_cluster = None
        network.eval()
        train_sampler.set_epoch(0)
        for data in tqdm(train_dataloader):
            idx, img, _ = data
            idx = idx.cuda(args.local_rank, non_blocking=True)
            img = img.cuda(args.local_rank, non_blocking=True)
            feature = network(img)[1].contiguous() if args.type == \
                'patch' else network(img)[0].contiguous()
            feature = concat_all_gather(feature).detach().cpu()
            idx = concat_all_gather(idx)
            if memory_bank is None:
                print("Initializing memory_bank bank: {} points.".format(n_train_points))
                memory_bank = torch.zeros(n_train_points, feature.size(1), 2) if \
                    args.type == 'patch' else torch.zeros(n_train_points, 2)
                memory_bank = memory_bank.to("cpu").detach()

            with torch.no_grad():
                memory_bank[idx] = torch.stack(feature.max(-1), dim=-1)
        
        torch.save(
            {'memory_bank': memory_bank},
            os.path.join(args.save_path, f'memory_{args.type}.pth'),
        )
        
    if num_per_cluster is None and args.local_rank == 0:
        num_per_cluster = torch.Tensor([])
        all_dim = torch.arange(args.out_dim).chunk(args.chunks)
        for i in tqdm(all_dim):
            mask = memory_bank[..., 1, None] == i.view(1, 1, -1)
            num_per_cluster = torch.cat((num_per_cluster, mask.sum((0,1))))
        
        torch.save(
            {'memory_bank': memory_bank,
             'num_per_cluster': num_per_cluster},
            os.path.join(args.save_path, f'memory_{args.type}.pth'),
        )
    
    if args.local_rank == 0:
        patterns = {}
        for i in num_per_cluster.topk(args.show_pics)[1]:
            mask = memory_bank[..., 1] == i
            if args.type == 'patch':
                values, spatial_id = (memory_bank[..., 0] * mask).max(-1)
                values, instance_id = torch.topk(values, args.topk * 2)
                spatial_id = spatial_id[instance_id]
                npatch = args.img_size // args.patch_size
                height_id = spatial_id // npatch
                width_id = spatial_id % npatch
                indices = torch.stack((instance_id, height_id, width_id), dim=-1)
            else:
                values, indices = torch.topk((memory_bank[..., 0] * mask), args.topk)
            patterns[i.item()] = indices
            
        augmentation = transforms.Compose([
            transforms.Resize(args.img_size // 7 * 8),
            transforms.CenterCrop(args.img_size),
        ])
        
        train_dataset = ImageFolderInstance(args.data_path, transform=augmentation)
        for nrank, (cluster, idxs) in enumerate(patterns.items()):
            size = math.ceil(args.topk ** 0.5) # 6
            unit = args.patch_size if args.type == 'patch' else args.img_size # 16/224
            vis_unit = (unit * args.patch_window) if args.type == 'patch' else unit # 80/224
            img = Image.new('RGB', (size * vis_unit, size * vis_unit))
            
            i = 0
            for idx in idxs.numpy():
                if args.type == 'patch':
                    _, raw, _ = train_dataset[idx[0]]
                    data = raw.crop((
                        (idx[2] - args.patch_window // 2) * unit, 
                        (idx[1] - args.patch_window // 2) * unit, 
                        (idx[2] + args.patch_window // 2 + 1) * unit, 
                        (idx[1] + args.patch_window // 2 + 1) * unit))
                    
                    # filter too dark patch for visualization
                    hsv = np.array(data.convert('HSV'))
                    if hsv[..., -1].mean() <= 40:
                        continue
                    
                    # draw highlight region
                    if args.patch_window > 1:
                        draw = ImageDraw.Draw(data, "RGBA")
                        draw.rectangle((
                            args.patch_window // 2 * unit, 
                            args.patch_window // 2 * unit, 
                            (args.patch_window // 2 + 1) * unit, 
                            (args.patch_window // 2 + 1) * unit), 
                            fill=(200, 100, 0, 127))
                else:
                    _, data, _ = train_dataset[idx]
                
                img.paste(data, (i % size * vis_unit, i // size * vis_unit))
            
                i += 1
                if i >= args.topk:
                    break

            img.save(os.path.join(save_path, 'c{}_crank{}_cid{}_top{}.jpg'.format(args.type, nrank, cluster, args.topk)))

if __name__ == '__main__':
    main()