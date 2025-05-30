import torch
import numpy as np
import pandas as pd
import pathlib
import sys
import sys
sys.path.append('/content/DNR_with_ATO')
from copy import deepcopy
from torch import nn
from utils import net_utils, path_utils, hypernet#,plot_utils
from utils.logging import AverageMeter, ProgressMeter
from utils.eval_utils import accuracy
from layers.CS_KD import KDLoss
from configs.base_config import Config
import importlib
from torchvision import transforms, datasets
#from datasets import load_dataset  
from torch.utils.data import random_split
import wandb
import logging
import inspect
#from utils.plot_utils import plot_accuracy, plot_loss, plot_sparsity, plot_layer_sparsity, plot_mask_overlap  
from utils.hypernet import AC_layer,HyperStructure,SelectionBasedRegularization
from utils.net_utils import reparameterize_non_sparse,display_structure_hyper
from trainers.default_cls import soft_train, validate, validate_mask
from data.datasets import load_dataset  
from models.resnet_gate import (
    ResNet,
    resnet18,
    resnet34,
    my_resnet34,
    my_resnet50,
    resnet101,
    my_resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
def get_trainer(cfg):
    """Retrieve training and validation functions from the specified trainer module."""
    try:
        trainer_module = importlib.import_module(f"trainers.{cfg.trainer}")
        return trainer_module.train, trainer_module.validate
    except ImportError as e:
        logger.error(f"Failed to import trainer module: {e}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_checkpoint(state, cfg, epochs, is_best=False, filename=None, save=True):
    """Save model checkpoint to the specified path."""
    if not save:
        return
    if filename is None:
        run_base_dir, ckpt_base_dir, _ = path_utils.get_directories(cfg, state["generation"])
        filename = ckpt_base_dir / f"epoch_{epochs - 1}.state"
    torch.save(state, filename)
    if is_best:
        best_path = ckpt_base_dir / "model_best.pth"
        torch.save(state, best_path)
def train_dense(cfg, generation, model=None, hyper_net=None, cur_mask_vec=None):
    """Train the model for a single generation using DNR."""
    arch_mapping = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "my_resnet34": my_resnet34,
        "resnet50": my_resnet50,
        "resnet101": resnet101,
        "my_resnet101": my_resnet101,
        "resnet152": resnet152,
        "resnext50_32x4d": resnext50_32x4d,
        "resnext101_32x8d": resnext101_32x8d,
        "wide_resnet50_2": wide_resnet50_2,
        "wide_resnet101_2": wide_resnet101_2,
    }

    model_func = arch_mapping.get(cfg.arch.lower())
    if model_func is None:
        raise ValueError(f"Unsupported architecture: {cfg.arch}. Supported architectures: {list(arch_mapping.keys())}")

    if model is None:
        model = model_func(pretrained=hasattr(cfg, "pretrained_imagenet") and cfg.pretrained_imagenet, num_classes=cfg.num_cls)
        if cfg.use_ac_layer:
            model.fc = AC_layer(num_class=cfg.num_cls)
        if hasattr(cfg, "use_pretrain") and cfg.use_pretrain:
            net_utils.load_pretrained(cfg.init_path, cfg.gpu, model, cfg)

    if cfg.pretrained and cfg.pretrained != "imagenet":
        net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model, cfg)
        model = net_utils.move_model_to_gpu(cfg, model)

    model = net_utils.move_model_to_gpu(cfg, model)

    if hyper_net is None:
        width, structure = model.count_structure()
        print(f"Structure from count_structure: {structure}")  # خط دیباگ
        hyper_net = HyperStructure(structure=structure, T=cfg.hyper_t, base=cfg.hyper_base, args=cfg)
        hyper_net = hyper_net.cuda()

    if cfg.save_model:
        run_base_dir, ckpt_base_dir, _ = path_utils.get_directories(cfg, generation)
        net_utils.save_checkpoint({"epoch": 0, "arch": cfg.arch, "state_dict": model.state_dict(), "generation": generation},
                                 is_best=False, filename=ckpt_base_dir / "init_model.state", save=False)

    cfg.trainer = "default_cls"
    cfg.pretrained = None

    criterion = nn.CrossEntropyLoss()
    params_group = net_utils.group_weight(hyper_net)
    optimizer_hyper = torch.optim.AdamW(params_group, lr=1e-3, weight_decay=1e-2)
    scheduler_hyper = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyper, milestones=[int(0.98 * ((cfg.epochs - 5) / 2) + 5)], gamma=0.1)

    params = net_utils.group_weight(model) if hasattr(cfg, "bn_decay") and cfg.bn_decay else model.parameters()
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay) if cfg.optimizer.lower() == "sgd" else torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int((cfg.epochs - 5) / 2))
    scheduler = net_utils.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_scheduler)

    
    print(f"cfg.data: {cfg.data}")
    train_loader, val_loader = load_dataset(name=cfg.set, root=cfg.data, path=cfg.data, sample='default', batch_size=cfg.batch_size)
    # Create validation subset for HyperNet
    ratio = (len(train_loader.dataset) / cfg.hyper_step) / len(train_loader.dataset)
    _, val_gate_dataset = random_split(train_loader.dataset, [len(train_loader.dataset) - int(ratio * len(train_loader.dataset)), int(ratio * len(train_loader.dataset))])
    val_loader_gate = net_utils.MultiEpochsDataLoader(val_gate_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    ###
    cfg.selection_reg = SelectionBasedRegularization(args=cfg, model=model)
    print("Checking selection_reg:", hasattr(cfg, 'selection_reg'))
    
    epoch_metrics = {"train_acc1": [], "train_acc5": [], "train_loss": [], "test_acc1": [], "test_acc5": [], "test_loss": [], "avg_sparsity": [], "mask_update": []}
    print(f"masks type: {type(masks)}, len: {len(masks)}, sample: {masks[0] if masks else None}")
    for epoch in range(cfg.epochs):
        train_acc1, train_acc5, train_loss, cur_mask_vec = soft_train(
            train_loader, model, hyper_net, criterion, val_loader_gate, 
            optimizer, optimizer_hyper, epoch, cur_mask_vec, cfg, 
            scheduler=scheduler, scheduler_hyper=scheduler_hyper  
        )

        test_acc1, test_acc5, test_loss = validate(val_loader, model, criterion, cfg, epoch) if epoch == 0 or (epoch + 1) % 10 == 0 else (0, 0, 0)
        if epoch >= cfg.start_epoch_hyper:
            test_acc1, test_acc5, test_loss = validate_mask(val_loader, model, criterion, cfg, epoch, cur_mask_vec)

        with torch.no_grad():
            hyper_net.eval()
            sparsity_str = display_structure_hyper(cur_mask_vec)
            sparsity_values = [float(s) for s in sparsity_str.split() if s.replace(".", "").isdigit()]
            avg_sparsity = sum(sparsity_values) / len(sparsity_values) if sparsity_values else 0
            masks = hyper_net.vector2mask(cur_mask_vec)
            layer_sparsity = {}
            for idx, mask_sublist in enumerate(masks):
                for sub_idx, param in enumerate(mask_sublist):
                    name = f"layer_{idx}_mask_{sub_idx}"
                    sparsity = 100 * (1 - param.mean().item())
                    layer_sparsity[name] = sparsity
           
            if "layer_sparsity" not in epoch_metrics:
                epoch_metrics["layer_sparsity"] = {}
            for layer, sparsity in layer_sparsity.items():
                if layer not in epoch_metrics["layer_sparsity"]:
                    epoch_metrics["layer_sparsity"][layer] = []
                epoch_metrics["layer_sparsity"][layer].append(sparsity)

        epoch_metrics["train_acc1"].append(train_acc1)
        epoch_metrics["train_acc5"].append(train_acc5)
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["test_acc1"].append(test_acc1)
        epoch_metrics["test_acc5"].append(test_acc5)
        epoch_metrics["test_loss"].append(test_loss)
        epoch_metrics["avg_sparsity"].append(avg_sparsity)
        epoch_metrics["mask_update"].append((epoch + 1) % cfg.hyper_step == 0)

        if epoch == cfg.epochs - 1:
            with torch.no_grad():
                masks = hyper_net.vector2mask(cur_mask_vec)
                model = reparameterize_non_sparse(cfg, model, masks)

    if cfg.save_model:
        save_checkpoint(
            {
                "epoch": cfg.epochs,
                "arch": cfg.arch,
                "state_dict": model.state_dict(),
                "best_acc1": max(epoch_metrics["test_acc1"]),
                "optimizer": optimizer.state_dict(),
                "hyper_net": hyper_net.state_dict(),
                "generation": generation,
            },
            cfg,
            epochs=cfg.epochs,
        )

    return model, hyper_net, cur_mask_vec, epoch_metrics

def percentage_overlap(prev_mask, curr_mask, percent_flag=False):
    """Calculate the percentage overlap between previous and current masks."""
    total_percent = {}
    for (name, prev_param), curr_param in zip(prev_mask.named_parameters(), curr_mask.parameters()):
        if "weight" in name and "bn" not in name and "downsample" not in name:
            prev_param_np = prev_param.detach().cpu().numpy()
            curr_param_np = curr_param.detach().cpu().numpy()
            overlap = np.sum((prev_param_np == curr_param_np) * curr_param_np)
            n_params = prev_param_np.size
            percent = overlap / (np.sum(curr_param_np == 1) if percent_flag and np.sum(curr_param_np == 1) > 0 else n_params)
            total_percent[name] = percent * 100
    return total_percent

def start_KE(cfg):
    """Start the Knowledge Evolution training process."""
    base_dir = pathlib.Path(f"{path_utils.get_checkpoint_dir()}/{cfg.name}")
    base_dir.mkdir(parents=True, exist_ok=True)

    ckpt_queue = []
    model = None
    hyper_net = None
    cur_mask_vec = None

    weights_history = {
        "conv1": [],
        "layer1.0.conv1": [],
        "layer2.0.conv1": [],
        "layer3.0.conv1": [],
        "layer4.0.conv1": [],
        "fc": [],
    }
    mask_history = {}
    all_epoch_data = []

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0
        model, hyper_net, cur_mask_vec, epoch_metrics = train_dense(cfg, gen, model, hyper_net, cur_mask_vec)

        weights_history["conv1"].append(model.conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history["layer1.0.conv1"].append(model.layer1[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history["layer2.0.conv1"].append(model.layer2[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history["layer3.0.conv1"].append(model.layer3[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history["layer4.0.conv1"].append(model.layer4[0].conv1.weight.data.clone().cpu().numpy().flatten())
        weights_history["fc"].append(model.fc.weight.data.clone().cpu().numpy().flatten())

        mask_history[gen] = {}
        if cur_mask_vec is not None:
            masks = hyper_net.vector2mask(cur_mask_vec)
            mask_history[gen] = {}
            for idx, mask_sublist in enumerate(masks):
                for sub_idx, param in enumerate(mask_sublist):
                    name = f"layer_{idx}_mask_{sub_idx}"
                    mask_history[gen][name] = param.data.clone().cpu().numpy()
        try:
            expected_length = cfg.epochs
            for key in epoch_metrics:
                if len(epoch_metrics[key]) != expected_length:
                    epoch_metrics[key].extend([None] * (expected_length - len(epoch_metrics[key])))
            epoch_df = pd.DataFrame({
                "Epoch": range(cfg.epochs),
                "Generation": [gen] * cfg.epochs,
                "Train_Acc@1": epoch_metrics["train_acc1"],
                "Train_Acc@5": epoch_metrics["train_acc5"],
                "Train_Loss": epoch_metrics["train_loss"],
                "Test_Acc@1": epoch_metrics["test_acc1"],
                "Test_Acc@5": epoch_metrics["test_acc5"],
                "Test_Loss": epoch_metrics["test_loss"],
                "Avg_Sparsity": epoch_metrics["avg_sparsity"],
                "Mask_Update": epoch_metrics["mask_update"],
            })
            all_epoch_data.append(epoch_df)
        except Exception as e:
            logger.error(f"Failed to create DataFrame for generation {gen}: {e}")
            epoch_df = pd.DataFrame({
                "Epoch": range(cfg.epochs),
                "Generation": [gen] * cfg.epochs,
                "Train_Acc@1": [None] * cfg.epochs,
                "Train_Acc@5": [None] * cfg.epochs,
                "Train_Loss": [None] * cfg.epochs,
                "Test_Acc@1": [None] * cfg.epochs,
                "Test_Acc@5": [None] * cfg.epochs,
                "Test_Loss": [None] * cfg.epochs,
                "Avg_Sparsity": [None] * cfg.epochs,
                "Mask_Update": [False] * cfg.epochs,
            })
            all_epoch_data.append(epoch_df)

        if cfg.num_generations == 1:
            break

    try:
        df = pd.concat(all_epoch_data, ignore_index=True)
    except Exception as e:
        logger.error(f"Failed to concatenate DataFrames: {e}")
        df = pd.DataFrame()

    if not df.empty and mask_history:
        plot_accuracy(df, base_dir, cfg.set, cfg.arch)
        plot_loss(df, base_dir, cfg.set, cfg.arch)
        plot_sparsity(mask_history, base_dir, cfg.set, cfg.arch)
        plot_layer_sparsity(epoch_metrics, cfg, base_dir, cfg.set, cfg.arch)
        plot_mask_overlap(model, mask_history, base_dir, cfg.set, cfg.arch)

def clean_dir(ckpt_dir, num_epochs):
    """Clean up checkpoint directory by removing specified files."""
    if "0000" in str(ckpt_dir):
        return
    for fname in ["model_best.pth", f"epoch_{num_epochs - 1}.state", "initial.state"]:
        rm_path = ckpt_dir / fname
        if rm_path.exists():
            rm_path.unlink()

if __name__ == "__main__":
    #cfg = Config().parse([])  # sys.argv[1:] 
    cfg = Config().parse(sys.argv[1:])
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.conv_type = "SplitConv"
    cfg.logger = logger
    cfg.set = getattr(cfg, "set", "CIFAR10")
    cfg.arch = getattr(cfg, "arch", "resnet18")

    if not cfg.no_wandb:
        if cfg.group_vars:
            group_name = cfg.group_vars[0] + str(getattr(cfg, cfg.group_vars[0]))
            for var in cfg.group_vars[1:]:
                group_name += f"_{var}{str(getattr(cfg, var))}"
            wandb.init(project="llf_ke", group=cfg.group_name, name=group_name)
            for var in cfg.group_vars:
                wandb.config.update({var: getattr(cfg, var)})

    if cfg.seed is not None and cfg.fix_seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    start_KE(cfg)
