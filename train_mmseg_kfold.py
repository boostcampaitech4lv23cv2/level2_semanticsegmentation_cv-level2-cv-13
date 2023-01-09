# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
###########added################
import numpy as np
import random
################################
import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)
import wandb

def set_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    wandb.login()
    runs=wandb.Api().runs(path="boostcamp_cv13/Semantic_Segmentation",order="created_at")
    try:
        this_run_num=f"{int(runs[0].name[1:4])+1:03d}"
    except:
        this_run_num="000"
    wandb.init(
        entity="boostcamp_cv13",
        project="Semantic_Segmentation",
        config="/opt/ml/level2_semanticsegmentation_cv-level2-cv-13/config-defaults-mmseg.yaml"
        )
    this_run_name=f"[{this_run_num}]-{wandb.config.model_config}-{wandb.run.id}"
    wandb.run.name=this_run_name
    wandb.run.save() # type: ignore

    cfg = Config.fromfile(wandb.config.model_config)
    cfg.data.samples_per_gpu=wandb.config.batch_size

    cfg.work_dir = osp.join('/opt/ml/outputs',
                            wandb.run.id)
    cfg.optimizer.lr=wandb.config.learning_rate
    # create work_dir
    mmcv.mkdir_or_exist(cfg.work_dir)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(wandb.config.model_config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')



    # set random seeds
    cfg.device = get_device()
    cfg.seed = wandb.config.seed
    set_seed(wandb.config.seed)
    meta['exp_name'] = osp.basename(wandb.config.model_config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    best_pth='/opt/ml/outputs/13ddzwb9/best_mIoU_epoch_28.pth'
    checkpoint = load_checkpoint(model, best_pth, map_location='cuda')

    model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
