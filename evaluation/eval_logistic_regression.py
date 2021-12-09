# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO library and authors' responce in a github issue:
https://github.com/facebookresearch/dino
https://github.com/facebookresearch/dino/issues/121
"""

import os
import argparse
import copy
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import cyanure as ars
import utils
import models

from torch import nn
from torchvision import transforms as pth_transforms
from loader import ImageFolderInstance

def eval_logistic_regression(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        try:
            print("loading features...")
            train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
            test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
            train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
            test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
        except:
            train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        train_features = train_features.cpu().numpy()
        test_features = test_features.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        test_labels = test_labels.cpu().numpy()

        print("Features are ready!\nStart the logistic regression.")
        for lambd in args.lr_lambd:
            acc = logistic_regression(train_features, train_labels,
                test_features, test_labels, lambd, args)
            print(f"Logistic regression result: Acc: {acc}")

    dist.barrier()

def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")
    dataset_train = ImageFolderInstance(traindir, transform=transform)
    dataset_val = ImageFolderInstance(valdir, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if 'swin' in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](
            window_size=args.window_size,
            patch_size=args.patch_size,
            num_classes=0)
    else:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size, 
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens==1)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(model, data_loader_train, args.n_last_blocks, args.avgpool_patchtokens, args.use_cuda)
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(model, data_loader_val, args.n_last_blocks, args.avgpool_patchtokens, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        print("Dumping features ...")
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, n, avgpool, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    labels = None
    for samples, labs, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labs = labs.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        
        def forward_single(samples):
            intermediate_output = model.get_intermediate_layers(samples, n)
            if avgpool == 0:
                # norm(x[:, 0])
                output = [x[:, 0] for x in intermediate_output]
            elif avgpool == 1:
                # x[:, 1:].mean(1)
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            elif avgpool == 2:
                # norm(x[:, 0]) + norm(x[:, 1:]).mean(1)
                output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
            else:
                assert False, "Unkown avgpool type {}".format(avgpool)
            
            feats = torch.cat(output, dim=-1).clone()
            return feats
        
        if multiscale:
            v = None
            for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
                if s == 1:
                    inp = samples.clone()
                else:
                    inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
                feats = forward_single(inp)
                if v is None:
                    v = feats
                else:
                    v += feats
            v /= 3
            v /= v.norm()
            feats = v
        else:
            feats = forward_single(samples)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = feats.new_zeros(len(data_loader.dataset), feats.shape[-1])
            labels = labs.new_zeros(len(data_loader.dataset))
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing labels into tensor of shape {labels.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        
        labels_all = torch.empty(
            dist.get_world_size(),
            labs.size(0),
            dtype=labs.dtype,
            device=labs.device,
        )
        label_l = list(labels_all.unbind(0))
        label_all_reduce = torch.distributed.all_gather(label_l, labs, async_op=True)
        label_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                labels.index_copy_(0, index_all, torch.cat(label_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                labels.index_copy_(0, index_all.cpu(), torch.cat(label_l).cpu())
    return features, labels


def logistic_regression(train_features, train_labels, test_features, test_labels, lambd, args):
    classifier = ars.MultiClassifier(loss="multiclass-logistic", penalty="l2", fit_intercept=False)
    classifier.fit(
        train_features,
        train_labels,
        it0=args.it0,
        lambd=lambd,
        lambd2=lambd,
        tol=1e-3,
        solver="catalyst-miso",
        restart=False,
        seed=0,
        max_epochs=args.epochs)
    score = classifier.score(test_features, test_labels)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=1` all the time for k-NN evaluation.""")
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base-size models with [CLS] token when doing linear classification.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--lr_lambd', default=[0.01, 0.1, 1], nargs='+', type=float,
        help='Regularization term coefficient. 0.1 is usually working the best.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base', 
        'vit_large', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'resnet50', 'resnet101'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=".", help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
        help='Please specify path to the ImageNet data.')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs of logistic regression.')
    parser.add_argument('--it0', default=10, type=int)
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_logistic_regression(args_copy)

