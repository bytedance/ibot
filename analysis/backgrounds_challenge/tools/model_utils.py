# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from backgrounds_challenge library:
https://github.com/MadryLab/backgrounds_challenge
"""

import os
import timm
import torch as ch

from tqdm import tqdm as tqdm

class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized

class NormalizedModel(ch.nn.Module):
    """
    """
    def __init__(self, model, dataset):
        super(NormalizedModel, self).__init__()
        self.normalizer = InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, inp):
        """
        """
        normalized_inp = self.normalizer(inp)
        output = self.model(normalized_inp)
        return output

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False, use_normalization=True):
    """
    """
    if pytorch_pretrained:
        classifier_model = timm.create_model(arch, pretrained=pytorch_pretrained)
    else:
        classifier_model = dataset.get_model(arch, pytorch_pretrained) if \
                                isinstance(arch, str) else arch
    if use_normalization:
        # Normalize by dataset mean and std, as is standard.
        model = NormalizedModel(classifier_model, dataset)
    else:
        model = classifier_model

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path)
            
            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):] if ('module.' in k) else k: v for k, v in sd.items()}
            
            # To deal with some compatability issues
            # model_dict = model.state_dict()
            # sd = {k: v for k, v in sd.items() if k in model_dict}
            # model_dict.update(sd)
            if use_normalization:
                model.model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            
            if parallel:
                model = ch.nn.DataParallel(model)
            model = model.cuda()
            
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint

def eval_model(loader, model, map_to_in9, map_in_to_in9=True):
    """
    *Internal function*
    Args:
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model: model to evaluate
        map_in_to_in9: whether or not to map model outputs from
        ImageNet class labels to ImageNet9 class labels
    Returns:
        The average top1 accuracy across the epoch.
    """

    model = model.eval()

    iterator = tqdm(enumerate(loader), total=len(loader))
    correct = 0
    for i, (inp, target) in iterator:
        output = model(inp)
        
        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        if map_in_to_in9:
            if map_to_in9 is None:
                raise ValueError('Need to pass in mapping from IN to IN9')
            pred_list = list(pred.numpy())
            pred = ch.LongTensor([map_to_in9[str(x)] for x in pred_list])
        correct += (pred==target).sum().item()
    
    return correct/len(loader.dataset)

def adv_bgs_eval_model(bg_loader, model, im, mask, fg_class, batch_size, map_to_in9, map_in_to_in9=True):
    """
    *Internal function*
    Args:
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model: model to evaluate
        use_mapping: whether or not to map model outputs from
        ImageNet class labels to ImageNet9 class labels
    Returns:
        The average top1 accuracy across the epoch.
    """

    model = model.eval()

    big_im = im.repeat(batch_size, 1, 1, 1)
    big_mask = mask.repeat(batch_size, 1, 1, 1)
    
    # iterator = tqdm(enumerate(bg_loader), total=len(bg_loader))
    for i, (inp, target) in enumerate(bg_loader):
    # for i, (inp, target) in iterator:
        if inp.shape[0] != batch_size: # For handling the last batch
            big_im = im.repeat(inp.shape[0], 1, 1, 1)
            big_mask = mask.repeat(inp.shape[0], 1, 1, 1)
        combined = inp * (1 - big_mask) + big_mask * big_im
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # from torchvision import transforms
        # for_viz = transforms.ToPILImage()(combined[0])
        # plt.imshow(for_viz)
        # plt.savefig('bg_challenge.png')
        
        output = model(combined)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        if map_in_to_in9:
            pred_list = list(pred.numpy())
            pred = ch.LongTensor([map_to_in9[str(x)] for x in pred_list])
            
        has_adversarial = (pred != fg_class).any().item()
        if has_adversarial:
            return True
    return False

