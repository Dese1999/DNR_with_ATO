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
        if not structure or len(structure) == 0:
            raise ValueError("The 'structure' argument must be a non-empty list. Received: {}".format(structure))
            
        self.bn1 = nn.LayerNorm([128 * 2])  
        self.T = T
        self.structure = structure
        print("Received structure:", structure)
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

        # validation
        if self.model_name not in ['resnet', 'mobnetv2', 'mobnetv3']:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if self.block_string not in ['BasicBlock', 'Bottleneck'] and self.model_name == 'resnet':
            raise ValueError(f"Unsupported block_string for resnet: {self.block_string}")

    def forward(self):
        device = self.bn1.weight.device
        self.inputs = self.inputs.to(device)
        self.h0 = self.h0.to(device)
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)  # outputs: [len(structure), 1, 256]
        outputs = [F.relu(self.bn1(outputs[i, 0, :])) for i in range(len(self.structure))]
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}, expected: [{self.structure[i]}]")
            if output.shape[0] != self.structure[i]:
                raise ValueError(f"Output {i} has shape {output.shape}, expected [{self.structure[i]}]")
        out = torch.cat(outputs, dim=0)  # [sum(structure)]
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base, device=device)
        if not self.training_mode:
            out = hard_concrete(out, device=device)
        #print(f"HyperStructure forward output shape: {out.shape}, expected length: {sum(self.structure)}")
        return out

    # def transform_output(self, inputs):
    #     arch_vector = []
    #     start = 0
    #     for i in range(len(self.structure)):
    #         end = start + self.structure[i]
    #         arch_vector.append(inputs[start:end])
    #         start = end
    #     return arch_vector
    def transform_output(self, inputs):
        if inputs.dim() == 1 and len(inputs) == sum(self.structure):
            return inputs  # بازگشت بردار پیوسته [3904]
        raise ValueError(f"Expected 1D vector of length {sum(self.structure)}, got shape {inputs.shape}")
    # def transform_output(self, inputs):
    #     if inputs.dim() == 1 and len(inputs) == sum(self.structure):
    #         return inputs  # 
    #     start = 0
    #     arch_vector = []
    #     for width in self.structure:
    #         end = start + width
    #         arch_vector.append(inputs[start:end])
    #         start = end
    #     return torch.cat(arch_vector)  
        
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

    # def vector2mask_resnet(self, inputs):
    #     vector = self.transform_output(inputs)
    #     mask_list = []
    #     for i in range(len(vector)):
    #         item_list = []
    #         mask_output = vector[i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         mask_input = vector[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #         item_list.append(mask_output)
    #         item_list.append(mask_input)
    #         mask_list.append(item_list)
    #     return mask_list
    def vector2mask_resnet(self, inputs):
        vector = self.transform_output(inputs)  # [sum(structure)]
        mask_list = []
        start = 0
        total_channels = sum(self.structure)  # مثلاً 3904
        if len(vector) != total_channels:
            print(f"Error: Vector length {len(vector)} is less than required {total_channels}")
            return []
        for i in range(len(self.structure)):
            item_list = []
            out_channels = self.structure[i]
            in_channels = 3 if i == 0 else self.structure[i-1]
            end = start + out_channels
            if end > len(vector):
                print(f"Error: Not enough values for layer {i}. Expected {end}, got {len(vector)}")
                return mask_list
            mask_output = vector[start:end].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if i == 0:
                mask_input = torch.ones(1, in_channels, 1, 1, device=vector.device)
            else:
                mask_input = vector[start-out_channels:start].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            item_list.append(mask_output)
            item_list.append(mask_input)
            mask_list.append(item_list)
            start = end
            #print(f"Layer {i}: mask_out shape {mask_output.shape}, mask_in shape {mask_input.shape}, out_channels {out_channels}, in_channels {in_channels}")
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
    def __init__(self, in_features, num_class=10):
        super(AC_layer, self).__init__()
        self.num_class = num_class
        self.fc_adapt = nn.Linear(in_features, num_class)  # 
        self.fc = nn.Linear(num_class, num_class)

    def forward(self, input):
        if len(input.size()) == 2:
            input = self.fc_adapt(input)  #num_class
            out = self.fc(input)
        else:
            b_size, n_c, w, h = input.size()
            input = input.view(b_size, 1, -1)
            input = F.adaptive_avg_pool1d(input, self.num_class)
            input = input.squeeze()
            input = self.fc_adapt(input)  # 
            out = self.fc(input)
        return out

def prob_round_torch(x):
    if x.is_cuda:
        stochastic_round = torch.rand(x.size(0), device=x.device) < x
    else:
        stochastic_round = torch.rand(x.size(0)) < x
    return stochastic_round

class SelectionBasedRegularization(nn.Module):
    def __init__(self, args, model=None):
        super().__init__()
        self.grad_mul = getattr(args, "grad_mul", 1.0)
        self.structure = getattr(args, "structure", [])
        self.lam = getattr(args, "gl_lam", 0.0001)
        self.model_name = getattr(args, "model_name", "resnet")
        self.block_string = getattr(args, "block_string", "BasicBlock")
        self.pruning_rate = getattr(args, "p", 0.5)  # Target pruning rate (ATO)
        self.use_fim = getattr(args, "use_fim", False)  # Use Fisher Information Matrix
        self.model = model  # Main model for FIM computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dynamically adjust lam based on pruning rate
        self.adjust_lam()

    def adjust_lam(self):
        """Dynamically adjust Group Lasso coefficient based on pruning rate."""
        if self.pruning_rate < 0.3:
            self.lam *= 0.5  # Reduce lam for lower pruning
        elif self.pruning_rate > 0.7:
            self.lam *= 2.0  # Increase lam for higher pruning

    def compute_fim(self, weights, masks):
        """Compute Fisher Information Matrix for identifying important weights."""
        if not self.use_fim or self.model is None:
            return torch.tensor(0.0).to(self.device)
        fim = []
        for (w_up, w_low), (m_out, m_in) in zip(weights, masks):
            grad_up = torch.autograd.grad(w_up.sum(), self.model.parameters(), create_graph=True, allow_unused=True)[0]
            grad_low = torch.autograd.grad(w_low.sum(), self.model.parameters(), create_graph=True, allow_unused=True)[0]
            fim_up = grad_up.pow(2).mean() if grad_up is not None else 0.0
            fim_low = grad_low.pow(2).mean() if grad_low is not None else 0.0
            fim.append(fim_up + fim_low)
        return torch.tensor(fim).mean().to(self.device)

    def forward(self, weights, masks):
        """Compute Group Lasso loss for pruning."""
        if self.model_name == "resnet":
            if self.block_string == "BasicBlock":
                return self.basic_forward(weights, masks)
            elif self.block_string == "Bottleneck":
                return self.bb_forward(weights, masks)
        elif self.model_name in ["mobnetv2", "mobnetv3"]:
            return self.mobilenet_forward(weights, masks)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def basic_forward(self, weights, masks):
        """Group Lasso for ResNet BasicBlock."""
        gl_list = []
        for i in range(len(self.structure)):
            w_up, w_low = weights[i]
            m_out, m_in = masks[i]
            m_out = custom_STE.apply(m_out)
            m_in = custom_STE.apply(m_in)
    
            # دیباگ
           # print(f"Layer {i}: w_up shape {w_up.shape}, m_out shape {m_out.shape}, w_low shape {w_low.shape}, m_in shape {m_in.shape}")
    
            expected_out_channels = w_up.shape[0]
            if m_out.shape[0] != expected_out_channels:
                print(f"Warning: m_out size {m_out.shape[0]} does not match output channels {expected_out_channels} for layer {i}")
                if m_out.shape[0] > expected_out_channels:
                    m_out = m_out[:expected_out_channels]
                else:
                    padding = torch.ones(expected_out_channels - m_out.shape[0], 1, 1, 1, device=m_out.device)
                    m_out = torch.cat([m_out, padding], dim=0)
    
            expected_in_channels = w_low.shape[1]
            if m_in.shape[1] != expected_in_channels:
                print(f"Warning: m_in size {m_in.shape[1]} does not match input channels {expected_in_channels} for layer {i}")
                if i == 0:  # برای conv1
                    m_in = torch.ones(1, expected_in_channels, 1, 1, device=m_in.device)  # 3 کانال برای RGB
                else:
                    if m_in.shape[1] > expected_in_channels:
                        m_in = m_in[:, :expected_in_channels]
                    else:
                        padding = torch.ones(1, expected_in_channels - m_in.shape[1], 1, 1, device=m_in.device)
                        m_in = torch.cat([m_in, padding], dim=1)
    
            #  Group Lasso Loss
            gl_loss = (w_up * (1 - m_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(0.5).sum() + \
                      (w_low * (1 - m_in)).pow(2).sum((0, 2, 3)).add(1e-8).pow(0.5).sum()
            gl_list.append(gl_loss)
    
        sum_loss = self.lam * sum(gl_list) / len(gl_list)
        if self.use_fim:
            fim_loss = self.compute_fim(weights, masks)
            sum_loss += 0.1 * fim_loss
        return sum_loss

    def bb_forward(self, weights, masks):
        """Group Lasso for ResNet Bottleneck."""
        gl_list = []
        for i in range(len(weights)):
            w_up, w_middle, w_low = weights[i]
            m_out, mm_in, mm_out, m_in = masks[i]
            m_out = custom_STE.apply(m_out)
            mm_in = custom_STE.apply(mm_in)
            mm_out = custom_STE.apply(mm_out)
            m_in = custom_STE.apply(m_in)
            gl_loss = (w_up * (1 - m_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(0.5).sum() + \
                      (w_middle * (1 - mm_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(0.5).sum() + \
                      (w_low * (1 - m_in)).pow(2).sum((0, 2, 3)).add(1e-8).pow(0.5).sum()
            gl_list.append(gl_loss)
        sum_loss = self.lam * sum(gl_list) / len(gl_list)
        if self.use_fim:
            fim_loss = self.compute_fim(weights, masks)
            sum_loss += 0.1 * fim_loss
        return sum_loss

    def mobilenet_forward(self, weights, masks):
        """Group Lasso for MobileNetV2/V3."""
        gl_list = []
        for i in range(len(weights)):
            w_up, w_middle, w_low = weights[i]
            m_out, m_middle, m_in = masks[i]
            m_out = custom_STE.apply(m_out)
            m_middle = custom_STE.apply(m_middle)
            m_in = custom_STE.apply(m_in)
            gl_loss = (w_up * (1 - m_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(0.5).sum() + \
                      (w_middle * (1 - m_middle)).pow(2).sum((1, 2, 3)).add(1e-8).pow(0.5).sum() + \
                      (w_low * (1 - m_in)).pow(2).sum((0, 2, 3)).add(1e-8).pow(0.5).sum()
            gl_list.append(gl_loss)
        sum_loss = self.lam * sum(gl_list) / len(gl_list)
        if self.use_fim:
            fim_loss = self.compute_fim(weights, masks)
            sum_loss += 0.1 * fim_loss
        return sum_loss    
