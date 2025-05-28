import time
import torch
import numpy as np
import torch.nn as nn
from utils import net_utils
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
import matplotlib.pyplot as plt
# Import DNR-specific utilities
from utils.net_utils import display_structure, display_structure_hyper, one_hot, mixup_func, cross_entropy_onehot_target, LabelSmoothingLoss

__all__ = ["soft_train_ato", "train_ato_no_mask", "validate_ato_mask", "validate_ato_no_mask", "one_step_hypernet_ato", "one_step_net_ato"]

def set_bn_eval(m):
    """Set BatchNorm layers to evaluation mode."""
    if isinstance(m, nn.modules.batchnorm._BatchNorm):        
        m.eval()

def set_bn_train(m):
    """Set BatchNorm layers to training mode."""
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()

# Training function with ATO masks and DNR logic
def soft_train(train_loader, model, hyper_net, criterion, valid_loader, optimizer, optimizer_hyper, epoch, cur_maskVec, cfg):
    """
    Train the model with ATO pruning masks and DNR logic.
    Args:
        train_loader: DataLoader for training data.
        model: Main neural network model.
        hyper_net: Hypernetwork to generate pruning masks.
        criterion: Loss function for one-hot encoded targets.
        valid_loader: DataLoader for validation data (used for hyper_net updates).
        optimizer: Optimizer for the main model.
        optimizer_hyper: Optimizer for the hyper_net.
        epoch: Current epoch number.
        cur_maskVec: Current mask vector (optional, used in later epochs).
        cfg: Configuration object with training parameters.
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, average loss, return_vect).
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    alignments = AverageMeter('AlignmentLoss', ':.4e')
    hyper_losses = AverageMeter('HyperLoss', ':.4e')
    res_losses = AverageMeter('ResLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    h_top1 = AverageMeter('HAcc@1', ':6.2f')
    h_top5 = AverageMeter('HAcc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, alignments, top1, top5, hyper_losses, res_losses, h_top1, h_top5], cfg, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    lmdValue = 0

    # Select mask: dynamic (early epochs) or fixed (later epochs)
    if epoch < int((cfg.epochs - 5) / 2) + 5:
        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()
            return_vect = vector.clone()
            masks = hyper_net.vector2mask(vector)
    else:
        print(">>>>> Using fixed mask")
        return_vect = cur_maskVec.clone()
        vector = cur_maskVec
        masks = hyper_net.vector2mask(vector)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.long().squeeze().cuda()

        optimizer.zero_grad()

        # Apply DNR techniques: label smoothing via one_hot and mixup
        targets_one_hot = one_hot(target, num_classes=1000, smoothing_eps=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1)
        if cfg.mix_up:
            images, targets_one_hot = mixup_func(images, targets_one_hot)

        model.train()
        sel_loss = torch.tensor(0.0).cuda()
        outputs = model(images)

        # Compute loss with one-hot targets
        loss = cross_entropy_onehot_target(outputs, targets_one_hot)

        # Compute selection loss for ATO
        weights = model.get_weights()
        with torch.no_grad():
            sel_loss = cfg.selection_reg(weights, masks) if hasattr(cfg, 'selection_reg') else 0.0
        loss += sel_loss

        loss.backward()
        optimizer.step()

        # Apply projection (Group Lasso or OTO) for ATO
        if epoch >= cfg.start_epoch_gl:
            if cfg.lmd > 0:
                lmdValue = cfg.lmd
            elif cfg.lmd == 0:
                if epoch < int((cfg.epochs - 5) / 2):
                    lmdValue = 10
                else:
                    lmdValue = 1000

            with torch.no_grad():
                if cfg.project == 'gl':
                    if hasattr(model, 'module'):
                        model.module.project_wegit(hyper_net.transform_output(vector), lmdValue, model.lr)
                    else:
                        model.project_wegit(hyper_net.transform_output(vector), lmdValue, model.lr)
                elif cfg.project == 'oto':
                    model.oto(hyper_net.transform_output(vector))

        # Compute accuracy
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        alignments.update(sel_loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # Update hyper_net
        if epoch >= cfg.start_epoch_hyper and (epoch < int((cfg.epochs - 5) / 2) + 5):
            if (i + 1) % cfg.hyper_step == 0:
                val_inputs, val_targets = next(iter(valid_loader))
                val_inputs = val_inputs.cuda()
                val_targets = val_targets.long().squeeze().cuda()

                optimizer_hyper.zero_grad()

                # Compute hyper_net loss with resource constraint (DNR)
                hyper_net.train()
                vector = hyper_net()
                model.set_virtual_gate(vector)
                hyper_outputs = model(val_inputs)

                res_loss = 2 * cfg.resource_constraint(hyper_net.resource_output())
                h_loss = nn.CrossEntropyLoss()(hyper_outputs, val_targets) + res_loss
                h_loss.backward()
                optimizer_hyper.step()

                h_acc1, h_acc5 = accuracy(hyper_outputs, val_targets, topk=(1, 5))
                h_top1.update(h_acc1.item(), val_inputs.size(0))
                h_top5.update(h_acc5.item(), val_inputs.size(0))
                hyper_losses.update(h_loss.item(), val_inputs.size(0))
                res_losses.update(res_loss.item(), val_inputs.size(0))

                model.reset_gates()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.print(i)

    print("Project Lmd in this Epoch:", lmdValue)
    if epoch >= cfg.start_epoch:
        if epoch < int((cfg.epochs - 5) / 2) + 5:
            with torch.no_grad():
                hyper_net.eval()
                vector = hyper_net()
                print("Sparsity (display_structure):")
                display_structure(hyper_net.transform_output(vector))
                print("Sparsity (display_structure_hyper):")
                display_structure_hyper(vector)
        else:
            print("Sparsity (display_structure):")
            display_structure(hyper_net.transform_output(vector))
            print("Sparsity (display_structure_hyper):")
            display_structure_hyper(vector)

    return top1.avg, top5.avg, losses.avg, return_vect

# Training function without masks (simple training inspired by simple_train)
def train(train_loader, model, criterion, optimizer, epoch, cfg):
    """
    Train the model without ATO masks (simple training mode).
    Args:
        train_loader: DataLoader for training data.
        model: Main neural network model.
        criterion: Loss function (supports label smoothing if cfg.ls is True).
        optimizer: Optimizer for the model.
        epoch: Current epoch number.
        cfg: Configuration object with training parameters.
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, average loss).
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], cfg, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.long().squeeze().cuda()

        outputs = model(images)

        # Use LabelSmoothingLoss if cfg.ls is True
        if cfg.ls:
            smooth_loss = LabelSmoothingLoss(classes=1000)(outputs, target)
            loss = smooth_loss
        else:
            loss = criterion(outputs, target)

        # Compute accuracy and record loss
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # Compute gradient and perform optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.print(i)

    return top1.avg, top5.avg, losses.avg

# Validation function with ATO masks
def validate_mask(val_loader, model, criterion, cfg, epoch, cur_maskVec=None):
    """
    Validate the model with ATO pruning masks.
    Args:
        val_loader: DataLoader for validation data.
        model: Main neural network model.
        criterion: Loss function.
        cfg: Configuration object with training parameters.
        epoch: Current epoch number.
        cur_maskVec: Current mask vector (optional).
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, average loss).
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], cfg, prefix='Test: ')

    model.eval()

    # Apply mask if provided
    if cur_maskVec is not None:
        if hasattr(model, 'module'):
            model.module.set_virtual_gate(cur_maskVec)
        else:
            model.set_virtual_gate(cur_maskVec)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.long().squeeze().cuda()

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                progress.print(i)

        progress.print(len(val_loader))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    # Reset gates if mask was applied
    if cur_maskVec is not None:
        if hasattr(model, 'module'):
            model.module.reset_gates()
        else:
            model.reset_gates()

    return top1.avg, top5.avg, losses.avg

# Validation function without masks (simple validation inspired by simple_validate)
def validate(val_loader, model, criterion, cfg, epoch):
    """
    Validate the model without ATO pruning masks.
    Args:
        val_loader: DataLoader for validation data.
        model: Main neural network model.
        criterion: Loss function.
        cfg: Configuration object with training parameters.
        epoch: Current epoch number.
    Returns:
        Tuple of (top1 accuracy, top5 accuracy, average loss).
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], cfg, prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.long().squeeze().cuda()

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                progress.print(i)

        progress.print(len(val_loader))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

# Helper function for hypernetwork step in ATO
def one_step_hypernet(inputs, targets, net, hyper_net, cfg):
    """
    Perform one step of hypernetwork training with ATO and DNR logic.
    Args:
        inputs: Input tensor.
        targets: Target tensor.
        net: Main neural network model.
        hyper_net: Hypernetwork to generate pruning masks.
        cfg: Configuration object with training parameters.
    Returns:
        Tuple of (masks, hyper_loss, res_loss, outputs).
    """
    net.eval()
    hyper_net.train()

    vector = hyper_net()
    if hasattr(net, 'module'):
        net.module.set_virtual_gate(vector)
    else:
        net.set_virtual_gate(vector)

    outputs = net(inputs)

    res_loss = 2 * cfg.resource_constraint(hyper_net.resource_output())
    hyper_loss = nn.CrossEntropyLoss()(outputs, targets) + res_loss

    with torch.no_grad():
        hyper_net.eval()
        vector = hyper_net()
        masks = hyper_net.vector2mask(vector)

    return masks, hyper_loss, res_loss, outputs

# Helper function for network step in ATO
def one_step_net(inputs, targets, net, masks, cfg):
    """
    Perform one step of network training with ATO and DNR logic.
    Args:
        inputs: Input tensor.
        targets: Target tensor.
        net: Main neural network model.
        masks: Pruning masks generated by hyper_net.
        cfg: Configuration object with training parameters.
    Returns:
        Tuple of (sel_loss, loss, outputs).
    """
    targets_one_hot = one_hot(targets, num_classes=1000, smoothing_eps=cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1)

    if cfg.mix_up:
        inputs, targets_one_hot = mixup_func(inputs, targets_one_hot)

    net.train()
    sel_loss = torch.tensor(0.0).cuda()
    outputs = net(inputs)

    loss = cross_entropy_onehot_target(outputs, targets_one_hot)

    # Compute selection loss for ATO
    weights = net.get_weights() if not hasattr(net, 'module') else net.module.get_weights()
    with torch.no_grad():
        sel_loss = cfg.selection_reg(weights, masks) if hasattr(cfg, 'selection_reg') else 0.0
    loss += sel_loss

    return sel_loss, loss, outputs
