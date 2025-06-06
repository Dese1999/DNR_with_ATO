import os
import pathlib
from . import constants
import random
import os.path as osp
from datetime import datetime


def get_checkpoint_dir():

    project_name = osp.basename(osp.abspath('./'))
    ckpt_dir = constants.checkpoints_dir
    assert osp.exists(ckpt_dir),('{} does not exists'.format(ckpt_dir))

    ckpt_dir = f'{ckpt_dir}/{project_name}'
    return ckpt_dir



def get_datasets_dir(dataset_name):
    datasets_dir = constants.datasets_dir

    assert osp.exists(datasets_dir),('{} does not exists'.format(datasets_dir))
    if dataset_name == 'CUB200' or dataset_name == 'CUB200_RET':
        dataset_dir = 'CUB_200_2011'
    elif dataset_name == 'CARS_RET':
        dataset_dir = 'stanford_cars'
    elif dataset_name == 'stanford':
        dataset_dir = 'Stanford_Online_Products'
    elif dataset_name == 'imagenet':
        dataset_dir = 'imagenet/ILSVRC/Data/CLS-LOC'
    elif dataset_name == 'market':
        dataset_dir = 'Market-1501-v15.09.15'
    elif dataset_name == 'Flower102' or dataset_name == 'Flower102Pytorch':
        dataset_dir = 'flower102'
    elif dataset_name == 'HAM':
        dataset_dir = 'HAM'
    elif dataset_name == 'FCAM':
        dataset_dir = 'FCAM'
    elif dataset_name == 'FCAMD':
        dataset_dir = 'FCAMD'
    elif dataset_name == 'Dog120':
        dataset_dir = 'stanford_dogs'
    elif dataset_name in ['MIT67','MINI_MIT67']:
        dataset_dir = 'mit67'
    elif dataset_name == 'Aircrafts':
        dataset_dir = 'aircraft/fgvc-aircraft-2013b/data'
    elif dataset_name == 'ImageNet':
        dataset_dir = 'imagenet/ILSVRC/Data/CLS-LOC'
    else:
        raise NotImplementedError('Invalid dataset name {}'.format(dataset_name))

    datasets_dir = '{}/{}'.format(datasets_dir, dataset_dir)

    return datasets_dir
####

def get_directories(args, generation):
    if args.config_file is None and args.name is None:
        raise ValueError("Must have name and config")

    config = args.name
    rno = random.randint(0, 1000000)
    if args.log_dir is None:
        run_base_dir = pathlib.Path(f"{get_checkpoint_dir()}/{args.name}/gen_{generation}/{rno}")
    else:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}/gen_{generation}/{rno}")

    def _run_dir_exists(run_base_dir):
        log_base_dir = run_base_dir / "logs"
        ckpt_base_dir = run_base_dir / "checkpoints"
        return log_base_dir.exists() or ckpt_base_dir.exists()

    rep_count = 0
    while _run_dir_exists(run_base_dir / f'{rep_count:04d}_g{args.gpu:01d}'):
        rep_count += 1

    run_base_dir = run_base_dir / f'{rep_count:04d}_g{args.gpu:01d}'
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    # Create directories
    run_base_dir.mkdir(parents=True, exist_ok=True)
    log_base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base_dir.mkdir(parents=True, exist_ok=True)

    (run_base_dir / "settings.txt").write_text(str(args))
    return run_base_dir, ckpt_base_dir, log_base_dir



###
# def get_directories(args,generation):
#     # if args.config_file is None or args.name is None:
#     if args.config_file is None and args.name is None:
#         raise ValueError("Must have name and config")

#     # config = pathlib.Path(args.config_file).stem
#     config = args.name
#     rno = random.randint(0, 1000000)
#     if args.log_dir is None:
#         run_base_dir = pathlib.Path(
#                 f"{get_checkpoint_dir()}/{args.name}/gen_{generation}/{rno}"
#             )
#     else:
#         run_base_dir = pathlib.Path(
#                 f"{args.log_dir}/{args.name}/gen_{generation}/{rno}"
#             )
    
        
    def _run_dir_exists(run_base_dir):
        log_base_dir = run_base_dir / "logs"
        ckpt_base_dir = run_base_dir / "checkpoints"

        return log_base_dir.exists() or ckpt_base_dir.exists()

   # if _run_dir_exists(run_base_dir):
    rep_count = 0
    while _run_dir_exists(run_base_dir / '{:04d}_g{:01d}'.format(rep_count,args.gpu)):
        rep_count += 1

    # date_time_int = int(datetime.now().strftime('%Y%m%d%H%M'))
    run_base_dir = run_base_dir / '{:04d}_g{:01d}'.format(rep_count,args.gpu)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir,exist_ok=True)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


if __name__ == '__main__':
    print(get_checkpoint_dir('test_exp'))
    print(get_datasets_dir('cub'))
