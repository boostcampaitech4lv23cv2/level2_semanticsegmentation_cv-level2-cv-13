# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
import pandas as pd
from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
import albumentations as A
import numpy as np

output_folder="/opt/ml/outputs/gaqoli29"
config_file=None
for file in os.listdir(output_folder):
    if file.endswith(".py"):
        config_file=os.path.join(output_folder, file)
        break
cfg = mmcv.Config.fromfile(config_file)

setup_multi_processes(cfg)

# if True:
#     # hard code index
#     cfg.data.test.pipeline[1].img_ratios = [
#         0.5, 0.75, 1.0, 1.25, 1.5, 1.75
#     ]
#     cfg.data.test.pipeline[1].flip = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
loader_cfg = dict(
    num_gpus=len(cfg.gpu_ids),
    dist=False,
    shuffle=False)
loader_cfg.update({
    k: v
    for k, v in cfg.data.items() if k not in [
        'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        'test_dataloader'
    ]
})
test_loader_cfg = {
    **loader_cfg,
    'samples_per_gpu': 1,
    'shuffle': False,  # Not shuffle by default
    **cfg.data.get('test_dataloader', {})
}
cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text,
            CLASSES=dataset.CLASSES,
            PALETTE=dataset.PALETTE)


# build the dataloader
data_loader = build_dataloader(dataset, **test_loader_cfg)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        
best_pth=None
for file in os.listdir(output_folder):
    if file.startswith("best"):
        best_pth=os.path.join(output_folder,file)
        break
checkpoint = load_checkpoint(model, best_pth, map_location='cuda')
'''if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = dataset.CLASSES
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = dataset.PALETTE'''

# clean gpu memory when starting a new evaluation.
torch.cuda.empty_cache()
cfg.device = get_device()
model = revert_sync_batchnorm(model)
model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
results = single_gpu_test(
    model,
    data_loader)
# transform = A.Compose([A.Resize(256,256)])
# i=0
# for data in data_loader:
#     results[i]=transform(image=np.zeros((512,512)),mask=results[i])['mask']
#     i+=1
    
submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

# test set에 대한 prediction
file_names=[]
names=open('/opt/ml/input/data/splits/test.txt', 'r').readlines()
for name in names:
    name=name.rstrip()
    file_names.append(name+".jpg")
# PredictionString 대입
for file_name, res in zip(file_names, results):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in res[::2,::2].flatten().tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(output_folder,"output.csv"), index=False)