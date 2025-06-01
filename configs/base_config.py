import os
import sys
import yaml
import argparse
import os.path as osp
import logging.config
from utils import os_utils
from utils import log_utils
from utils import path_utils
import ast
from utils.hypernet import SelectionBasedRegularization
args = None

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Knowledge Evolution Training with DNR")
        parser.add_argument(
            "--label-smoothing", default=0.0, type=float, help="Label smoothing epsilon (default: 0.0, no smoothing)"
        )
        parser.add_argument(
            "--mix-up", action="store_true", default=False, help="Enable mixup augmentation"
        )
        parser.add_argument(
            '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
        # Parameters for SelectionBasedRegularization
        parser.add_argument(
            "--gl-lam", default=0.0001, type=float, help="Group Lasso coefficient for SelectionBasedRegularization"
        )
        parser.add_argument(
            "--p", default=0.8, type=float, help="Target pruning rate for ATO"
        )
        parser.add_argument(
            "--use-fim", action="store_true", default=False, help="Use Fisher Information Matrix for pruning"
        )
        parser.add_argument(
            "--structure", type=str, default="[]", help="Structure list for SelectionBasedRegularization (e.g., [64, 64, 128])"
        )
        parser.add_argument(
            "--grad-mul", default=10.0, type=float, help="Gradient multiplier for SelectionBasedRegularization"
        )
        parser.add_argument('--reg_w', default=4.0, type=float)  # 4.0 
        parser.add_argument('--start_epoch_hyper', default=20, type=int)

        # Core training arguments
        parser.add_argument(
            "--data", help="path to dataset base directory", default="/home/datasets"
        )
        parser.add_argument(
            "--set", type=str, default="CIFAR10",
            choices=['Flower102Pytorch', 'Flower102', 'CUB200', 'Aircrafts', 'Dog120', 'MIT67',
                     'CIFAR10', 'CIFAR10val', 'CIFAR100', 'CIFAR100val', 'tinyImagenet_full', 'tinyImagenet_val',
                     'CUB200_val', 'Dog120_val', 'MIT67_val', 'imagenet']
        )
        parser.add_argument(
            "--arch", type=str, default="resnet50", help="Model architecture (e.g., resnet18, resnet50, resnext50_32x4d)"
        )
        parser.add_argument(
            "--num-cls", type=int, default=10, help="Number of output classes"
        )
        parser.add_argument(
            "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
        )
        parser.add_argument(
            "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
        )
        parser.add_argument(
            "--lr", "--learning-rate", default=0.253, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay", dest="weight_decay"
        )
        parser.add_argument(
            "--optimizer", help="Which optimizer to use", default="sgd"
        )
        parser.add_argument(
            "--num-generations", default=11, type=int, help="Number of training generations"
        )
        parser.add_argument(
            "--seed", default=None, type=int, help="need to set fix_seed = True to take effect"
        )
        parser.add_argument(
            "--fix-seed", action="store_true", help="Fix random seed"
        )
        parser.add_argument(
            "--gpu", default='0', type=int, help="Which GPUs to use?"
        )
        parser.add_argument(
            "--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model"
        )
        parser.add_argument(
            "--init-path", default='/data/output-ai/', type=str, help="path to pre-trained model weights"
        )
        parser.add_argument(
            "--save-model", action="store_true", default=True, help="save model checkpoints"
        )
        parser.add_argument(
            "--config-file", help="Config file to use (see configs dir)", default=None
        )
        parser.add_argument(
            "--log-dir", help="Where to save the runs. If None use ./runs", default=None
        )
        parser.add_argument(
            "--name", default=None, type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--log-file", default='train_log.txt', type=str, help="Log file name"
        )
        parser.add_argument(
            "--hyper-step", default=20, type=int, help="hypernet update interval"
        )
        parser.add_argument(
            "--start-epoch-hyper", default=20, type=int, help="start epoch for hypernet updates"
        )
        
        parser.add_argument(
            '--start_epoch_gl', default=100, type=int)

        parser.add_argument(
            "--gates", default=2, type=int, help="number of gates in the model"
        )
        parser.add_argument(
            "--no-wandb", action="store_true", default=False, help="disable wandb logging"
        )
        parser.add_argument("--group-vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
        
        # HyperStructure parameters
        parser.add_argument(
            "--hyper-t", default=0.4, type=float, help="Temperature parameter T for HyperStructure"
        )
        parser.add_argument(
            "--hyper-base", default=3.0, type=float, help="Base parameter for HyperStructure"
        )
        parser.add_argument(
            '--hyper_step', default=20, type=int)

        # Additional parameters for HyperStructure and related classes
        parser.add_argument(
            "--model-name", type=str, default="resnet", help="Model type (e.g., resnet, mobnetv2, mobnetv3)"
        )
        parser.add_argument(
            "--block-string", type=str, default="BasicBlock", help="Block type (e.g., BasicBlock, Bottleneck)"
        )
        parser.add_argument(
            "--se-list", type=str, default="[False, False, False, False]", help="List of SE layer flags (e.g., [False, False])"
        )
        
        parser.add_argument(
            "--concrete-flag", action="store_true", default=False, help="Use concrete distribution for soft_gate"
        )
        parser.add_argument(
            "--margin", type=float, default=0.0, help="Margin for soft_gate (ignored if concrete_flag is True)"
        )
        parser.add_argument(
            "--use-ac-layer", action="store_true", default=False, help="Use AC_layer for classification"
        )

        self.parser = parser

    def parse(self, args):
        self.cfg = self.parser.parse_args(args)

        # Set number of classes based on dataset
        if self.cfg.set == 'Flower102' or self.cfg.set == 'Flower102Pytorch':
            self.cfg.num_cls = 102
        elif self.cfg.set == 'CUB200':
            self.cfg.num_cls = 200
        elif self.cfg.set == 'imagenet':
            self.cfg.num_cls = 1000
        elif self.cfg.set in ['tinyImagenet_full', 'tinyImagenet_val']:
            self.cfg.num_cls = 200
        elif self.cfg.set == 'Dog120':
            self.cfg.num_cls = 120
        elif self.cfg.set == 'MIT67':
            self.cfg.num_cls = 67
        elif self.cfg.set == 'Aircrafts':
            self.cfg.num_cls = 100
        elif self.cfg.set == 'CIFAR10' or self.cfg.set == 'CIFAR10val':
            self.cfg.num_cls = 10
        elif self.cfg.set == 'CIFAR100' or self.cfg.set == 'CIFAR100val':
            self.cfg.num_cls = 100
        else:
            raise NotImplementedError(f'Invalid dataset {self.cfg.set}')

        # Convert se_list to list
        self.cfg.se_list = ast.literal_eval(self.cfg.se_list)
#
        # Construct experiment directory and logging
        self.cfg.group_name = self.cfg.name
        self.cfg.name = f'SPLT_CLS_{self.cfg.set}_{self.cfg.arch}_G{self.cfg.num_generations}_e{self.cfg.epochs}_seed{self.cfg.seed}/'
        self.cfg.exp_dir = osp.join(path_utils.get_checkpoint_dir(), self.cfg.name)

        os_utils.touch_dir(self.cfg.exp_dir)
        log_file = os.path.join(self.cfg.exp_dir, self.cfg.log_file)
        logging.config.dictConfig(log_utils.get_logging_dict(log_file))
        self.cfg.logger = logging.getLogger('KE')

        return self.cfg

# Function to convert string to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
