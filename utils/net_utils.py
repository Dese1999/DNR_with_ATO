import os
import math
import torch
import shutil
import models
import pathlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from imgnet_models.gate_function import custom_STE, virtual_gate
from imgnet_models.mobilenetv3 import Hswish, Hsigmoid
import torchvision
from copy import deepcopy
from layers import bn_type, conv_type, linear_type
import torch.backends.cudnn as cudnn
from utils import hypernet

import matplotlib.pyplot as plt
from hypernet import HyperStructure

# ATo Utility Functions
DEFAULT_OPT_PARAMS = {
    "sgd": {"first_momentum": 0.0, "second_momentum": 0.0, "dampening": 0.0, "weight_decay": 0.0, "lmbda": 1e-3, "lmbda_amplify": 2, "hat_lmbda_coeff": 10},
    "adam": {"lr": 1e-3, "first_momentum": 0.9, "second_momentum": 0.999, "dampening": 0.0, "weight_decay": 0.0, "lmbda": 1e-2, "lmbda_amplify": 20, "hat_lmbda_coeff": 1e3},
    "adamw": {"lr": 1e-3, "first_momentum": 0.9, "second_momentum": 0.999, "dampening": 0.0, "weight_decay": 1e-2, "lmbda": 1e-2, "lmbda_amplify": 20, "hat_lmbda_coeff": 1e3}
}

def group_weight(module, weight_norm=True):
    """
    Group model parameters for optimization, separating decay and no-decay groups.
    Optimized for DNR with HyperStructure, soft_gate, and virtual_gate layers.
    """
    group_decay = []  # Parameters with weight decay (weights)
    group_no_decay = []  # Parameters without weight decay (biases, normalization layers)

    # Handle inputs if they exist (e.g., for HyperStructure)
    if hasattr(module, 'inputs'):
        group_no_decay.append(module.inputs)

    # Iterate over all modules in the model
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GRU):
            for k in range(m.num_layers):
                group_decay.append(getattr(m, f'weight_ih_l{k}'))
                group_decay.append(getattr(m, f'weight_hh_l{k}'))
                if getattr(m, f'bias_ih_l{k}') is not None:
                    group_no_decay.append(getattr(m, f'bias_ih_l{k}'))
                    group_no_decay.append(getattr(m, f'bias_hh_l{k}'))
                if m.bidirectional:
                    group_decay.append(getattr(m, f'weight_ih_l{k}_reverse'))
                    group_decay.append(getattr(m, f'weight_hh_l{k}_reverse'))
                    if getattr(m, f'bias_ih_l{k}_reverse') is not None:
                        group_no_decay.append(getattr(m, f'bias_ih_l{k}_reverse'))
                        group_no_decay.append(getattr(m, f'bias_hh_l{k}_reverse'))
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, hypernet.HyperStructure):
            # Group weights of Linear layers and GRU in HyperStructure
            for param in m.parameters():
                if param.ndim > 1:  # Weights (matrices/tensors)
                    group_decay.append(param)
                else:  # Biases or scalars
                    group_no_decay.append(param)
        elif isinstance(m, (hypernet.soft_gate, hypernet.virtual_gate)):
            # Group weights of soft_gate (e.g., self.weights)
            if isinstance(m, hypernet.soft_gate) and hasattr(m, 'weights'):
                group_decay.append(m.weights)
            # virtual_gate has no trainable parameters, skip

    # Verify the grouping
    total_params = sum(p.numel() for p in module.parameters())
    decay_params = sum(p.numel() for p in group_decay)
    no_decay_params = sum(p.numel() for p in group_no_decay)
    assert total_params == decay_params + no_decay_params, f"Parameter grouping mismatch! Total: {total_params}, Decay: {decay_params}, NoDecay: {no_decay_params}"

    # Return groups with weight decay settings aligned with ATO
    groups = [
        dict(params=group_decay, weight_decay=DEFAULT_OPT_PARAMS['adamw']['weight_decay']),  # Default to 1e-2
        dict(params=group_no_decay, weight_decay=0.0)
    ]
    return groups
class resource_constraint(nn.Module):
    def __init__(self, num_epoch, cut_off_epoch, p):
        super().__init__()
        self.num_epoch = num_epoch
        self.cut_off_epoch = cut_off_epoch
        self.p = p

    def forward(self, input, epoch):
        overall_length = sum(x.size(0) for x in input)
        cat_tensor = custom_STE.apply(input[0], False)
        for i in range(1, len(input)):
            cat_tensor = torch.cat([cat_tensor, custom_STE.apply(input[i], False)])
        return torch.abs(cat_tensor.mean() - self.p)

class Flops_constraint(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3):
        super().__init__()
        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.inc_1st = in_channel
        self.t_flops = self.init_total_flops()

    def init_total_flops(self):
        total_flops = sum(k * (ic / g) * oc * os + 3 * oc * os for k, ic, oc, os, g in zip(self.k_size, self.in_csize, self.out_csize, self.out_size, self.g_size))
        print(f'+ Number of FLOPs: {total_flops/1e9:.5f}G')
        return total_flops

    def forward(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        for i, tensor in enumerate(input):
            current_tensor = custom_STE.apply(tensor, False)
            if i > 0:
                c_in = custom_STE.apply(input[i-1], False).sum()
            c_out = current_tensor.sum()
            sum_flops += self.k_size[i] * (c_in / self.g_size[i]) * c_out * self.out_size[i] + 3 * c_out * self.out_size[i]
        return torch.log(torch.abs(sum_flops / self.t_flops - self.p) + 1) * 2

# DNR Utility Functions
def create_dense_mask_0(net, device, value):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    return net.to(device)

def get_model(args):
    args.logger.info(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch](args) if args.arch != 'resnet18' else models.__dict__[args.arch]()
    if args.set in ["CIFAR100", "CIFAR10"]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

def move_model_to_gpu(args, model):
    if not torch.cuda.is_available():
        raise ValueError("CPU-only experiments currently unsupported")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        return model.cuda(args.gpu)
    elif args.multigpu is None:
        return model
    else:
        args.logger.info(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        return torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(args.multigpu[0])

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=True):
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best and save:
        shutil.copyfile(filename, filename.parent / "model_best.pth")

def reparameterize_non_sparse(cfg, net, net_sparse_set):
    for (name, param), mask_param in zip(net.named_parameters(), net_sparse_set.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            re_init_param = re_init_weights(param.data.shape, cfg.device)
            param.data = param.data * mask_param.data
            re_init_param.data[mask_param.data == 1] = 0
            param.data = param.data + re_init_param.data
    return net

def re_init_weights(shape, device, reinint_method='kaiming'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
        return torch.squeeze(mask, 1)
    nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    return mask

