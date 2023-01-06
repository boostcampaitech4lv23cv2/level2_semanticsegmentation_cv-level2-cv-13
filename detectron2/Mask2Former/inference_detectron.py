import os
import pickle
import sys
from pathlib import Path
import numpy as np
import argparse
import albumentations as A
import pandas as pd
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
# import Mask2Former project
from mask2former import add_maskformer2_config
sys.path.insert(0, "Mask2Former")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default = 'mask2former_result')
    parser.add_argument('--weight', type=str, default='model_final.pth')
    arg = parser.parse_args()
    return arg


def setup_cfg(args):
    config_dir = os.path.join(args.model_dir, 'config.yaml')
    weight_dir = os.path.join(args.model_dir, args.weight)
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg = get_cfg()
    cfg.merge_from_file(config_dir)
    cfg.MODEL.WEIGHTS = weight_dir
    cfg.freeze()
    return cfg


def main():
    args = get_parser()
    cfg = setup_cfg(args)
    with open('/opt/ml/input/data/test.json') as f:
        test_files = json.load(f)
    images = test_files['images']
    predictor = DefaultPredictor(cfg)
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    image_id = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        image_id.append(file_name)
        path = Path('/opt/ml/input/data') / file_name
        img = read_image(path, format="BGR")
        pred = predictor(img)
        output = pred['sem_seg'].argmax(dim=0).detach().cpu().numpy()
        temp_mask = []
        temp_img = np.zeros((3, 512, 512))
        transformed = transform(image=temp_img, mask=output)
        mask = transformed['mask']
        temp_mask.append(mask)

        out = np.array(temp_mask)
        out = out.reshape([out.shape[0], size * size]).astype(int)
        preds_array = np.vstack((preds_array, out))

    submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

    for file_name, string in zip(image_id, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
            ignore_index=True)

    submission.to_csv(f"{args.model_dir}/submission.csv", index=False)


if __name__ == "__main__":
    main()
