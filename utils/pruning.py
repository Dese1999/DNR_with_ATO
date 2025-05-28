import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
from utils import net_utils, path_utils,hypernet
import torch.nn.functional as F
from hypernet import custom_STE, virtual_gate

# Assuming utils is the updated module from previous files

class Pruner:
    def __init__(self, model, loader=None, device='cpu', silent=False):
        self.device = device
        self.loader = loader
        self.model = model
        
        self.weights = [layer for name, layer in model.named_parameters() if 'mask' not in name]
        self.indicators = [torch.ones_like(layer) for name, layer in model.named_parameters() if 'mask' not in name]
        self.mask_ = net_utils.create_dense_mask_0(deepcopy(model), self.device, value=1)
        self.pruned = [0 for _ in range(len(self.indicators))]
 
        if not silent:
            print("number of weights to prune:", [x.numel() for x in self.indicators])

    def indicate(self):
        """
        Apply indicators (masks) to the model weights using custom_STE for ATo compatibility.
        """
        with torch.no_grad():
            idx = 0
            for name, param in self.model.named_parameters():
                if 'mask' not in name:
                    # Apply mask using custom_STE
                    masked_weight = custom_STE.apply(param * self.indicators[idx], False)
                    param.data.copy_(masked_weight)
                    idx += 1
            
            # Update mask_ to reflect indicators
            idx = 0
            for name, param in self.mask_.named_parameters():
                if 'mask' not in name:
                    param.data.copy_(self.indicators[idx].data)
                    idx += 1

    def snip(self, sparsity, mini_batches=1, silent=False):
        """
        Apply SNIP pruning method with ATo compatibility.
        """
        mini_batches = min(mini_batches, len(self.loader) // 32)
        mini_batch = 0
        self.indicate()
        self.model.zero_grad()
        grads = [torch.zeros_like(w) for w in self.weights]
        
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            x = self.model.forward(x)
            L = torch.nn.CrossEntropyLoss()(x, y)
            grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                     for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
            
            mini_batch += 1
            if mini_batch >= mini_batches: 
                break

        with torch.no_grad():
            saliences = [(grad * weight).view(-1).abs().cpu() for weight, grad in zip(self.weights, grads)]
            saliences = torch.cat(saliences)
            
            thresh = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0])
            
            for j, layer in enumerate(self.indicators):
                layer[(grads[j] * self.weights[j]).abs() <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
        
        self.indicate()
        
        # Calculate current sparsity
        current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])
            print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
        
        return self.mask_

    def snip_it(self, sparsity, steps=5, mini_batches=1, silent=False):
        """
        Apply SNIP-it pruning (iterative, unstructured) with ATo compatibility.
        """
        start = 0.5
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        if not silent:
            print(f"prune_steps: {prune_steps}")
        
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNIP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            self.indicate()
            self.model.zero_grad()
            grads = [torch.zeros_like(w) for w in self.weights]
            loss = 0.0
            mini_batch = 0
            
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                x = self.model.forward(x)
                L = torch.nn.CrossEntropyLoss()(x, y)
                loss += L.item()
                grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                         for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
                
                mini_batch += 1
                if mini_batch >= mini_batches: 
                    break
            
            loss /= max(1, mini_batch)
            
            with torch.no_grad():
                saliences = [(grad * weight).view(-1).abs() / (loss + 1e-8) 
                             for grad, weight in zip(grads, self.weights)]
                saliences = torch.cat(saliences).cpu()
                thresh = float(saliences.kthvalue(int(target_sparsity * saliences.shape[0]))[0])
                
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    layer[(grad * weight).abs() / (loss + 1e-8) <= thresh] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
            
            self.indicate()
            
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            
            if not silent:
                print("Step weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
                print("Step sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                          for i, pruned in enumerate(self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
            
            if abs(current_sparsity - sparsity) < 1e-3:
                break
        
        return self.mask_

    def snap_it(self, sparsity, steps=5, start=0.5, mini_batches=1, silent=False):
        """
        Apply SNAP-it pruning (iterative, structured) with ATo compatibility.
        """
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        current_sparsity = 0.0
        remaining = 1.0
        
        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"SNAP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")
            
            prune_rate = (target_sparsity - current_sparsity) / (remaining + 1e-8)
            
            self.indicate()
            self.model.zero_grad()
            grads = [torch.zeros_like(w) for w in self.weights]
            loss = 0.0
            mini_batch = 0
            
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                x = self.model.forward(x)
                L = torch.nn.CrossEntropyLoss()(x, y)
                loss += L.item()
                grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g)) 
                         for g, ag in zip(grads, torch.autograd.grad(L, self.weights, allow_unused=True))]
                
                mini_batch += 1
                if mini_batch >= mini_batches: 
                    break
            
            loss /= max(1, mini_batch)
            
            with torch.no_grad():
                saliences = []
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    if len(weight.shape) == 4:  # Conv2d layer
                        importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        saliences.append(importance.view(-1))
                    else:
                        importance = (grad * weight).abs().view(-1) / (loss + 1e-8)
                        saliences.append(importance)
                
                saliences = torch.cat(saliences).cpu()
                thresh = float(saliences.kthvalue(int(prune_rate * saliences.shape[0]))[0])
                
                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    if len(weight.shape) == 4:  # Conv2d layer
                        importance = torch.sum((grad * weight).abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        node_mask = importance >= thresh
                        layer[node_mask == False, :, :, :] = 0
                    else:
                        layer[(grad * weight).abs() / (loss + 1e-8) <= thresh] = 0
                    self.pruned[j] = int(torch.sum(layer == 0))
            
            self.indicate()
            
            current_sparsity = sum(self.pruned) / sum([ind.numel() for ind in self.indicators])
            remaining = 1.0 - current_sparsity
            
            if not silent:
                print("Step weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
                print("Step sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                          for i, pruned in enumerate(self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")
            
            if abs(current_sparsity - sparsity) < 1e-3:
                break
        
        return self.mask_

    def cnip_it(self, sparsity, steps=5, start=0.5, mini_batches=1, silent=False):
        """
        Apply CNIP-it pruning (iterative, combined unstructured and structured) with ATo compatibility.
        """
        prune_steps = [sparsity - (sparsity - start) * (0.5 ** i) for i in range(steps)] + [sparsity]
        current_sparsity = 0.0
        remaining = 1.0

        for step, target_sparsity in enumerate(prune_steps):
            if not silent:
                print(f"CNIP-it step {step + 1}/{len(prune_steps)}: Targeting sparsity {target_sparsity:.4f}")

            prune_rate = (target_sparsity - current_sparsity) / (remaining + 1e-8)

            self.indicate()
            self.model.zero_grad()
            grads = [torch.zeros_like(w) for w in self.weights]
            loss = 0.0
            mini_batch = 0

            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model.forward(x)
                L = nn.CrossEntropyLoss()(output, y)
                loss += L.item()
                grad_outputs = torch.autograd.grad(L, self.weights, allow_unused=True)
                grads = [g.abs() + (ag.abs() if ag is not None else torch.zeros_like(g))
                         for g, ag in zip(grads, grad_outputs)]
                mini_batch += 1
                if mini_batch >= mini_batches:
                    break

            if mini_batch == 0:
                raise ValueError("No mini-batches processed. Check data loader.")
            loss /= mini_batch

            with torch.no_grad():
                weight_saliences = []
                node_saliences = []
                for j, (grad, weight) in enumerate(zip(grads, self.weights)):
                    weight_importance = (grad * weight).abs().view(-1) / (loss + 1e-8)
                    weight_saliences.append(weight_importance)

                    if len(weight.shape) == 4:  # Conv2d layer
                        node_importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        node_saliences.append(node_importance.view(-1))
                    else:
                        node_saliences.append(torch.zeros_like(weight_importance))

                all_saliences = torch.cat(weight_saliences + node_saliences).cpu()
                if all_saliences.numel() == 0:
                    raise ValueError("No saliencies computed. Check model weights or gradients.")
                thresh = float(all_saliences.kthvalue(int(prune_rate * all_saliences.shape[0]))[0])

                weight_threshold = thresh
                node_threshold = thresh
                percentage_weights = sum((ws < weight_threshold).sum().item() for ws in weight_saliences) / sum(ws.numel() for ws in weight_saliences) if weight_saliences else 0.0
                percentage_nodes = sum((ns < node_threshold).sum().item() for ns in node_saliences) / sum(ns.numel() for ns in node_saliences) if node_saliences else 0.0

                if not silent:
                    print(f"Fraction for pruning nodes: {percentage_nodes:.4f}, Fraction for pruning weights: {percentage_weights:.4f}")

                for j, (grad, weight, layer) in enumerate(zip(grads, self.weights, self.indicators)):
                    weight_mask = (grad * weight).abs() / (loss + 1e-8) >= weight_threshold
                    layer[weight_mask == False] = 0

                    if len(weight.shape) == 4:  # Conv2d layer
                        node_importance = torch.sum(grad.abs(), dim=(1, 2, 3)) / (loss + 1e-8)
                        node_mask = node_importance >= node_threshold
                        layer[node_mask == False, :, :, :] = 0

                    self.pruned[j] = int(torch.sum(layer == 0))

            self.indicate()

            total_params = sum(ind.numel() for ind in self.indicators)
            current_sparsity = sum(self.pruned) / total_params if total_params > 0 else 0.0
            remaining = 1.0 - current_sparsity

            if not silent:
                print("Step weights left: ", [ind.numel() - pruned for ind, pruned in zip(self.indicators, self.pruned)])
                print("Step sparsities: ", [round(100 * pruned / ind.numel(), 2)
                                          for ind, pruned in zip(self.indicators, self.pruned)])
                print(f"Current total sparsity: {current_sparsity*100:.2f}\n")

            if abs(current_sparsity - sparsity) < 1e-3:
                break

        if self.mask_ is None:
            raise ValueError("Mask not generated. Check pruning process.")
        return self.mask_

    def snipR(self, sparsity, silent=False):
        """
        Apply SNIP-R pruning (perturbation-based) with ATo compatibility.
        """
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y)

            for laynum, layer in enumerate(self.weights):
                if not silent:
                    print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0

                    z = self.model.forward(x)
                    L = torch.nn.CrossEntropyLoss()(z, y)
                    saliences[laynum].view(-1)[weight] = (L - L0).abs()    
                    layer.view(-1)[weight] = temp
                
            saliences_bag = torch.cat([s.view(-1) for s in saliences]).cpu()
            thresh = float(saliences_bag.kthvalue(int(sparsity * saliences_bag.numel()))[0])

            for j, layer in enumerate(self.indicators):
                layer[saliences[j] <= thresh] = 0
                self.pruned[j] = int(torch.sum(layer == 0))   
        
        self.indicate()
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel() - pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100 * pruned / self.indicators[i].numel(), 2) 
                                  for i, pruned in enumerate(self.pruned)])

        return self.mask_

    def cwi_importance(self, sparsity, device):
        """
        Compute importance based on weight and gradient magnitudes with ATo compatibility.
        """
        mask = net_utils.create_dense_mask_0(deepcopy(self.model), device, value=0)
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                param_grad = param.grad if param.grad is not None else torch.zeros_like(param)
                importance = abs(param.data) + abs(param_grad)
                param_mask.data = custom_STE.apply(importance, False)

        imp = [layer for name, layer in mask.named_parameters() if 'mask' not in name]
        imp = torch.cat([i.view(-1).cpu() for i in imp])
        percentile = np.percentile(imp.numpy(), sparsity * 100)
        above_threshold = [i > percentile for i in imp]
        
        idx = 0
        for name, param_mask in mask.named_parameters():
            if 'mask' not in name:
                threshold_tensor = above_threshold[idx].view(param_mask.shape).to(device)
                param_mask.data = custom_STE.apply(param_mask * threshold_tensor, False)
                idx += 1
        return mask

    def apply_reg(self, mask):
        """
        Apply regularization based on mask with ATo compatibility.
        """
        for (name, param), param_mask in zip(self.model.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                l2_grad = custom_STE.apply(param_mask.data * param.data, False)
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad += l2_grad

    def update_reg(self, mask, reg_decay, cfg):
        """
        Update regularization mask with ATo compatibility.
        """
        reg_mask = net_utils.create_dense_mask_0(deepcopy(mask), cfg.device, value=0)
        for (name, param), param_mask in zip(reg_mask.named_parameters(), mask.parameters()):
            if 'weight' in name and 'bn' not in name and 'downsample' not in name:
                mask_tensor = custom_STE.apply(param_mask.data, False)
                param.data[mask_tensor == 1] = 0
                if cfg.reg_type == 'x':
                    if reg_decay < 1:
                        param.data[mask_tensor == 0] += min(reg_decay, 1)
                elif cfg.reg_type == 'x^2':
                    if reg_decay < 1:
                        param.data[mask_tensor == 0] += min(reg_decay, 1)
                        param.data[mask_tensor == 0] = param.data[mask_tensor == 0] ** 2
                elif cfg.reg_type == 'x^3':
                    if reg_decay < 1:
                        param.data[mask_tensor == 0] += min(reg_decay, 1)
                        param.data[mask_tensor == 0] = param.data[mask_tensor == 0] ** 3
        reg_decay += cfg.reg_granularity_prune
        return reg_mask, reg_decay