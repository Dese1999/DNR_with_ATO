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
import torchvision
from . import hypernet  #  
from .hypernet import HyperStructure, custom_STE, virtual_gate
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from packaging import version
import torch.utils.data
# ATO and DNR Utility Functions

def display_structure(all_parameters, p_flag=False):
    """Display the sparsity of the model's parameters."""
    layer_sparsity = []
    num_layers = 0
    for name, param in all_parameters.items():
        if "weight" in name and "bn" not in name and "downsample" not in name:
            current_parameter = param.cpu().data
            num_layers += 1
            if p_flag:
                sparsity = current_parameter.sum().item() / (current_parameter.size(0) * current_parameter[0].nelement())
            else:
                sparsity = current_parameter.sum().item() / current_parameter[0].nelement()
            layer_sparsity.append(sparsity)

    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f ' % (i + 1, layer_sparsity[i])
    print(print_string)

def display_structure_hyper(vectors):
    """Display the sparsity of each layer's mask vector in DNR."""
    num_layers = len(vectors)
    layer_sparsity = []
    for i in range(num_layers):
        current_parameter = vectors[i].cpu().data
        if i == 0:
            print(f"Layer 1 mask sample: {current_parameter[:5]}")
        sparsity = current_parameter.sum().item() / current_parameter.size(0)
        layer_sparsity.append(sparsity)
    
    print_string = ''
    return_string = ''
    for i in range(num_layers):
        print_string += f'l-{i+1} s-{layer_sparsity[i]:.3f} '
        return_string += f'{layer_sparsity[i]:.3f} '
    print(print_string.strip())
    return return_string.strip()
def group_weight(module, weight_norm=True):
    """Group model parameters for optimization in DNR framework."""
    group_decay = []
    group_no_decay = []
    grouped_params = set()  # 

    if hasattr(module, 'inputs'):
        group_no_decay.append(module.inputs)
        grouped_params.add(id(module.inputs))

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if id(m.weight) not in grouped_params:
                group_decay.append(m.weight)
                grouped_params.add(id(m.weight))
            if m.bias is not None and id(m.bias) not in grouped_params:
                group_no_decay.append(m.bias)
                grouped_params.add(id(m.bias))
        elif isinstance(m, nn.GRU):
            for k in range(m.num_layers):
                weight_ih = getattr(m, f'weight_ih_l{k}')
                weight_hh = getattr(m, f'weight_hh_l{k}')
                if id(weight_ih) not in grouped_params:
                    group_decay.append(weight_ih)
                    grouped_params.add(id(weight_ih))
                if id(weight_hh) not in grouped_params:
                    group_decay.append(weight_hh)
                    grouped_params.add(id(weight_hh))
                if getattr(m, f'bias_ih_l{k}') is not None:
                    bias_ih = getattr(m, f'bias_ih_l{k}')
                    bias_hh = getattr(m, f'bias_hh_l{k}')
                    if id(bias_ih) not in grouped_params:
                        group_no_decay.append(bias_ih)
                        grouped_params.add(id(bias_ih))
                    if id(bias_hh) not in grouped_params:
                        group_no_decay.append(bias_hh)
                        grouped_params.add(id(bias_hh))
                if m.bidirectional:
                    weight_ih_reverse = getattr(m, f'weight_ih_l{k}_reverse')
                    weight_hh_reverse = getattr(m, f'weight_hh_l{k}_reverse')
                    if id(weight_ih_reverse) not in grouped_params:
                        group_decay.append(weight_ih_reverse)
                        grouped_params.add(id(weight_ih_reverse))
                    if id(weight_hh_reverse) not in grouped_params:
                        group_decay.append(weight_hh_reverse)
                        grouped_params.add(id(weight_hh_reverse))
                    if getattr(m, f'bias_ih_l{k}_reverse') is not None:
                        bias_ih_reverse = getattr(m, f'bias_ih_l{k}_reverse')
                        bias_hh_reverse = getattr(m, f'bias_hh_l{k}_reverse')
                        if id(bias_ih_reverse) not in grouped_params:
                            group_no_decay.append(bias_ih_reverse)
                            grouped_params.add(id(bias_ih_reverse))
                        if id(bias_hh_reverse) not in grouped_params:
                            group_no_decay.append(bias_hh_reverse)
                            grouped_params.add(id(bias_hh_reverse))
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None and id(m.weight) not in grouped_params:
                group_no_decay.append(m.weight)
                grouped_params.add(id(m.weight))
            if m.bias is not None and id(m.bias) not in grouped_params:
                group_no_decay.append(m.bias)
                grouped_params.add(id(m.bias))
        elif isinstance(m, HyperStructure):
            for param in m.parameters():
                if id(param) not in grouped_params:
                    if param.ndim > 1:
                        group_decay.append(param)
                    else:
                        group_no_decay.append(param)
                    grouped_params.add(id(param))
        elif isinstance(m, hypernet.AC_layer):
            if id(m.fc.weight) not in grouped_params:
                group_decay.append(m.fc.weight)
                grouped_params.add(id(m.fc.weight))
            if m.fc.bias is not None and id(m.fc.bias) not in grouped_params:
                group_no_decay.append(m.fc.bias)
                grouped_params.add(id(m.fc.bias))

    total_params = sum(p.numel() for p in module.parameters())
    decay_params = sum(p.numel() for p in group_decay)
    no_decay_params = sum(p.numel() for p in group_no_decay)
    ## Debug to be sure
    print("Total params:", total_params)
    print("Decay params:", decay_params)
    print("No decay params:", no_decay_params)
    print("Sum of grouped params:", decay_params + no_decay_params)

    assert total_params == decay_params + no_decay_params, "Parameter grouping mismatch!"

    groups = [
        dict(params=group_decay, weight_decay=1e-2),
        dict(params=group_no_decay, weight_decay=0.0)
    ]
    return groups
def create_dense_mask_0(net, device, value):
    """Create a dense mask with all values set to the specified value."""
    for param in net.parameters():
        param.data[param.data == param.data] = value
    return net.to(device)

def get_model(args):
    """Create a model based on the architecture specified in args."""
    args.logger.info(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch](args) if args.arch != 'resnet18' else models.__dict__[args.arch]()
    if args.set in ["CIFAR100", "CIFAR10"]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model

def move_model_to_gpu(args, model):
    """Move the model to GPU based on args configuration."""
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
    """Save model checkpoint to the specified path."""
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best and save:
        shutil.copyfile(filename, filename.parent / "model_best.pth")

def reparameterize_non_sparse(cfg, net, net_sparse_set):
    """Re-parameterize the model by applying masks from HyperStructure in DNR."""
    device = cfg.device if hasattr(cfg, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for (name, param), mask_param in zip(net.named_parameters(), net_sparse_set.items()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            param.data = param.data * mask_param.to(device)
            re_init_param = torch.empty(param.data.shape, device=device)
            nn.init.kaiming_uniform_(re_init_param, a=math.sqrt(5))
            re_init_param.data[mask_param == 0] = 0
            re_init_param.data[mask_param == 1] = 0
            param.data = param.data + re_init_param.data
    return net

def re_init_weights(shape, device, reinint_method='kaiming'):
    """Re-initialize weights using the specified method."""
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
        return torch.squeeze(mask, 1)
    nn.init.kaiming_uniform_(mask, a=math.sqrt(5))
    return mask

class resource_constraint(nn.Module):
    """Resource constraint loss for ATO."""
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
    """FLOPs constraint loss for ATO."""
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

def TrainVal_split(dataset, validation_split, shuffle_dataset=True):
    """Split dataset into train and validation samplers."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

def binary_loss(cat_tensor):
    """Compute binary cross-entropy loss for mask vectors."""
    loss = cat_tensor * torch.log(cat_tensor + 1e-8) * (cat_tensor >= 0.5).detach().float() + \
           (1 - cat_tensor) * torch.log(1 - cat_tensor + 1e-8) * (cat_tensor < 0.5).detach().float()
    return loss.mean()

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    """Compute knowledge distillation loss."""
    labels.requires_grad = False
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs.detach()/T, dim=1)) * (alpha) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def loss_label_smoothing(outputs, labels, T, alpha):
    """Compute loss with label smoothing."""
    uniform = torch.Tensor(outputs.size()).fill_(1/outputs.size(1))
    if outputs.is_cuda:
        uniform = uniform.cuda()
    sm_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(uniform/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return sm_loss

def print_model_param_nums(model=None, multiply_adds=True):
    """Print the number of parameters in the model."""
    if model is None:
        model = torchvision.models.alexnet()
    total = sum(param.nelement() for param in model.parameters())
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

def print_model_param_flops(model=None, input_res=224, multiply_adds=False):
    """Print the number of FLOPs in the model."""
    if model is None:
        model = torchvision.models.alexnet()
    prods = {}
    list_conv = []
    list_linear = []
    list_bn = []
    list_relu = []
    list_pooling = []
    list_upsample = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        flops = (kernel_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)

    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(conv_hook)
            if isinstance(module, nn.Linear):
                module.register_forward_hook(linear_hook)
            if isinstance(module, nn.BatchNorm2d):
                module.register_forward_hook(bn_hook)
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(relu_hook)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                module.register_forward_hook(pooling_hook)
            if isinstance(module, nn.Upsample):
                module.register_forward_hook(upsample_hook)

    foo(model)
    input = torch.rand(1, 3, input_res, input_res).cuda()
    out = model(input)
    total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample)
    print('  + Number of FLOPs: %.5fG' % (total_flops / 1e9))
    return total_flops

def get_middle_Fsize(model, input_res=32):
    """Get intermediate sizes for FLOPs calculation."""
    size_out = []
    size_kernel = []
    group_size = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        group_size.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    return size_out, size_kernel, group_size, size_inchannel, size_outchannel

def get_middle_Fsize_resnet(model, input_res=224):
    """Get intermediate sizes for ResNet FLOPs calculation."""
    size_out = []
    size_kernel = []
    group_size = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        group_size.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                if layer_id - 3 >= 0:
                    modules[layer_id - 3].register_forward_hook(conv_hook)
                if layer_id + 1 < len(modules):
                    modules[layer_id + 1].register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    return size_out, size_kernel, group_size, size_inchannel, size_outchannel

def get_middle_Fsize_resnetbb(model, input_res=224, num_gates=2):
    """Get intermediate sizes for ResNetBB FLOPs calculation."""
    size_out = []
    size_kernel = []
    group_size = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        group_size.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        soft_gate_count = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                if num_gates == 2:
                    if layer_id - 2 >= 0:
                        modules[layer_id - 2].register_forward_hook(conv_hook)
                    if soft_gate_count % 2 == 1 and layer_id + 1 < len(modules):
                        modules[layer_id + 1].register_forward_hook(conv_hook)
                    soft_gate_count += 1
                else:
                    if layer_id - 4 >= 0:
                        modules[layer_id - 4].register_forward_hook(conv_hook)
                    if layer_id - 2 >= 0:
                        modules[layer_id - 2].register_forward_hook(conv_hook)
                    if layer_id + 1 < len(modules):
                        modules[layer_id + 1].register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    return size_out, size_kernel, group_size, size_inchannel, size_outchannel

def get_middle_Fsize_densenet(model, input_res=224):
    """Get intermediate sizes for DenseNet FLOPs calculation."""
    size_out = []
    size_kernel = []
    group_size = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        group_size.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear) or isinstance(m0, nn.ReLU) or isinstance(m0, virtual_gate):
                truncate_module.append(m0)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            if isinstance(m, virtual_gate):
                if layer_id - 1 >= 0:
                    truncate_module[layer_id - 1].register_forward_hook(conv_hook)
                if layer_id + 3 < len(truncate_module):
                    truncate_module[layer_id + 3].register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    return size_out, size_kernel, group_size, size_inchannel, size_outchannel

def get_middle_Fsize_mobnetv3(model, input_res=224):
    """Get intermediate sizes for MobileNetV3 FLOPs calculation."""
    all_dict = {
        'size_out': [],
        'size_kernel': [],
        'group_size': [],
        'size_inchannel': [],
        'size_outchannel': [],
        'se_list': [],
        'hswish_list': []
    }

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        all_dict['size_out'].append(output_height * output_width)
        all_dict['size_kernel'].append(self.kernel_size[0] * self.kernel_size[1])
        all_dict['group_size'].append(self.groups)
        all_dict['size_inchannel'].append(input_channels)
        all_dict['size_outchannel'].append(output_channels)

    def foo(net):
        modules = list(net.modules())
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, (nn.BatchNorm2d, nn.Conv2d, nn.Linear, nn.ReLU, nn.ReLU6, virtual_gate)):
                truncate_module.append(m0)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            if isinstance(m, virtual_gate) and layer_id + 4 < len(truncate_module) - 1:
                if isinstance(truncate_module[layer_id - 1], Hswish):
                    all_dict['hswish_list'].append(True)
                elif isinstance(truncate_module[layer_id - 1], nn.ReLU):
                    all_dict['hswish_list'].append(False)
                if layer_id - 3 >= 0:
                    truncate_module[layer_id - 3].register_forward_hook(conv_hook)
                if layer_id + 1 < len(truncate_module):
                    truncate_module[layer_id + 1].register_forward_hook(conv_hook)
                if layer_id + 4 < len(truncate_module) and isinstance(truncate_module[layer_id + 4], nn.Conv2d):
                    truncate_module[layer_id + 4].register_forward_hook(conv_hook)
                    all_dict['se_list'].append(False)
                elif layer_id + 7 < len(truncate_module):
                    truncate_module[layer_id + 7].register_forward_hook(conv_hook)
                    all_dict['se_list'].append(True)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    return all_dict

def transfer_weights(model, my_model):
    """Transfer weights from one model to another."""
    mymbnet_v2_ms = list(my_model.modules())
    mbnet_v2_ms = list(model.modules())
    for m in mymbnet_v2_ms:
        if isinstance(m, virtual_gate):
            mymbnet_v2_ms.remove(m)
    for layer_id in range(len(mymbnet_v2_ms)):
        m0, m1 = mbnet_v2_ms[layer_id], mymbnet_v2_ms[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Conv2d):
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

def moving_average(net1, net2, alpha=1):
    """Compute moving average of two models' parameters."""
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def check_bn(model):
    """Check if the model contains BatchNorm layers."""
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    """Reset BatchNorm running mean and variance."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _check_bn(module, flag):
    """Helper function to check BatchNorm layers."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def _get_momenta(module, momenta):
    """Helper function to get BatchNorm momenta."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    """Helper function to set BatchNorm momenta."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """Update BatchNorm buffers using the training dataset."""
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = input
        b = input_var.data.size(0)
        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum
        model(input_var)
        n += b
    model.apply(lambda module: _set_momenta(module, momenta))

def one_hot(y, num_classes, smoothing_eps=None):
    """Convert labels to one-hot encoding with optional label smoothing."""
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y

def cross_entropy_onehot_target(logit, target):
    """Compute cross-entropy loss with one-hot targets."""
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def mixup_func(input, target, alpha=0.2):
    """Apply mixup augmentation to inputs and targets."""
    gamma = np.random.beta(alpha, alpha)
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(perm_input, alpha=1 - gamma), target.mul_(gamma).add_(perm_target, alpha=1 - gamma)

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss implementation."""
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def load_pretrained(pretrained_path, gpu, model, cfg):
    """Load pretrained weights into the model."""
    if os.path.isfile(pretrained_path):
        logger.info(f"=> Loading pretrained checkpoint '{pretrained_path}'")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # remove 'module.' prefix
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"=> Loaded pretrained checkpoint '{pretrained_path}'")
    else:
        logger.warning(f"=> No pretrained checkpoint found at '{pretrained_path}'")


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler): The sampler to repeat.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """DataLoader that supports multiple epochs without reinitializing workers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
        # if version.parse(torch.__version__) > version.parse("1.1.0"):
        #     super(GradualWarmupScheduler, self).step()

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step()
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics)        
