from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.nn.utils import weight_norm

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, T, offset=0, device='cpu'):
    gumbel_sample = sample_gumbel(logits.size(), device=device)
    y = logits + gumbel_sample + offset
    return torch.sigmoid(y / T)

def hard_concrete(out, device='cpu'):
    out_hard = torch.zeros(out.size(), device=device)
    out_hard[out >= 0.5] = 1
    out_hard = (out_hard - out).detach() + out
    return out_hard

def truncate_normal_(size, a=-1, b=1, device='cpu'):
    values = truncnorm.rvs(a, b, size=size)
    return torch.from_numpy(values).float().to(device)

class custom_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_clone = input.clone()
        if input_clone.requires_grad:
            input_clone = prob_round_torch(input_clone)
        else:
            input_clone[input_clone >= 0.5] = 1
            input_clone[input_clone < 0.5] = 0
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > 1] = 0
        grad_input[input < 0] = 0
        return grad_input

class HyperStructure(nn.Module):
    def __init__(self, structure=None, T=0.4, base=3.0, args=None, training_mode=True):
        super(HyperStructure, self).__init__()
        self.bn1 = nn.LayerNorm([128 * 2])  # تطبیق با خروجی دوطرفه GRU
        self.T = T
        self.structure = structure
        self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)
        self.h0 = torch.zeros(2, 1, 128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure), 1, 64))
        nn.init.orthogonal_(self.inputs)
        self.linear_list = [nn.Linear(256, structure[i], bias=False) for i in range(len(structure))]
        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.base = base
        self.model_name = args.model_name if hasattr(args, 'model_name') else 'resnet'
        self.block_string = getattr(args, 'block_string', 'BasicBlock')
        self.se_list = getattr(args, 'se_list', [False] * len(structure))
        self.training_mode = training_mode

        # اعتبارسنجی
        if self.model_name not in ['resnet', 'mobnetv2', 'mobnetv3']:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if self.block_string not in ['BasicBlock', 'Bottleneck'] and self.model_name == 'resnet':
            raise ValueError(f"Unsupported block_string for resnet: {self.block_string}")

    def forward(self):
        device = self.bn1.weight.device
        self.inputs = self.inputs.to(device)
        self.h0 = self.h0.to(device)
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base, device=device)
        if not self.training_mode:
            out = hard_concrete(out, device=device)
        return out.squeeze()

    def transform_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end
        return arch_vector

    def resource_output(self):
        device = self.bn1.weight.device
        self.inputs = self.inputs.to(device)
        self.h0 = self.h0.to(device)
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base, device=device)
        out = hard_concrete(out, device=device)
        return out.squeeze()

    def vector2mask_resnet(self, inputs):
        vector = self.transform_output(inputs)
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_input)
            mask_list.append(item_list)
        return mask_list

    def vector2mask_resnetbb(self, inputs):
        vector = self.transform_output(inputs)
        mask_list = []
        length = len(vector)
        for i in range(0, length, 2):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mask_middle_output = vector[i + 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i + 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_middle_input)
            item_list.append(mask_middle_output)
            item_list.append(mask_input)
            mask_list.append(item_list)
        return mask_list

    def vector2mask_mobnetv2(self, inputs):
        vector = self.transform_output(inputs)
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_middle)
            item_list.append(mask_input)
            mask_list.append(item_list)
        return mask_list

    def vector2mask_mobnetv3(self, inputs):
        vector = self.transform_output(inputs)
        mask_list = []
        for i in range(len(vector)):
            item_list = []
            mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_middle = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_middle)
            item_list.append(mask_input)
            if self.se_list[i]:
                maskse_input = vector[i].unsqueeze(0)
                maskse_output = vector[i].unsqueeze(-1)
                item_list.append(maskse_input)
                item_list.append(maskse_output)
            mask_list.append(item_list)
        return mask_list

    def vector2mask(self, inputs):
        if self.model_name == 'resnet':
            if self.block_string == 'BasicBlock':
                return self.vector2mask_resnet(inputs)
            elif self.block_string == 'Bottleneck':
                return self.vector2mask_resnetbb(inputs)
        elif self.model_name == 'mobnetv2':
            return self.vector2mask_mobnetv2(inputs)
        elif self.model_name == 'mobnetv3':
            return self.vector2mask_mobnetv3(inputs)

class soft_gate(nn.Module):
    def __init__(self, width, base_width=-1, concrete_flag=False, margin=0):
        super(soft_gate, self).__init__()
        if base_width == -1:
            base_width = width
        self.weights = nn.Parameter(torch.ones(width))
        self.concrete_flag = concrete_flag
        self.g_w = torch.Tensor([float(base_width) / float(width)])
        self.margin = margin
        if concrete_flag:
            self.margin = 0

    def forward(self, input, cur_mask_vec=None):
        if not self.training:
            return input
        if cur_mask_vec is not None:
            device = input.device
            gate_f = cur_mask_vec.to(device)
        else:
            if self.concrete_flag:
                gate_f = custom_STE.apply(self.weights)
            else:
                gate_f = custom_STE.apply(self.weights)
            gate_f = gate_f.unsqueeze(0)
        if input.is_cuda:
            gate_f = gate_f.cuda()
        if len(input.size()) == 2:
            input = gate_f.expand_as(input) * input
        elif len(input.size()) == 4:
            gate_f = gate_f.unsqueeze(-1).unsqueeze(-1)
            input = gate_f.expand_as(input) * input
        return input

class virtual_gate(nn.Module):
    def __init__(self, width):
        super(virtual_gate, self).__init__()
        self.width = width

    def forward(self, input, cur_mask_vec):
        orignal_input = input.detach()
        device = input.device
        gate_f = cur_mask_vec.to(device)
        if len(input.size()) == 2:
            gate_f = gate_f.unsqueeze(0)
            input = gate_f.expand_as(input) * input
        elif len(input.size()) == 4:
            gate_f = gate_f.unsqueeze(-1).unsqueeze(-1)
            input = gate_f.expand_as(input) * input
        return input

    def collect_mse(self):
        return torch.tensor(0.0)  # Placeholder, not used in DNR

    def reset_value(self):
        pass  # No need to reset in DNR

class AC_layer(nn.Module):
    def __init__(self, num_class=10):
        super(AC_layer, self).__init__()
        self.fc = nn.Linear(num_class, num_class)
        self.num_class = num_class

    def forward(self, input):
        b_size, n_c, w, h = input.size()
        input = input.view(b_size, 1, -1)
        input = F.adaptive_avg_pool1d(input, self.num_class)
        out = self.fc(input.squeeze())
        return out

def prob_round_torch(x):
    if x.is_cuda:
        stochastic_round = torch.rand(x.size(0), device=x.device) < x
    else:
        stochastic_round = torch.rand(x.size(0)) < x
    return stochastic_round
    
