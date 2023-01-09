# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
import random
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

prog_description = '''K-Fold coco split.

To split coco data for semi-supervised object detection:
    python tools/misc/split_coco.py
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        help='The data root of coco dataset.',
        default='/opt/ml/input/data/train_all.json')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The output directory of coco semi-supervised annotations.',
        default='/opt/ml/level2_semanticsegmentation_cv-level2-cv-13/stratified_fold_dataset')
    parser.add_argument(
        '--fold',
        type=int,
        help='K-fold cross validation for semi-supervised object detection.',
        default=5)
    args = parser.parse_args()
    return args


def split_coco_stratified(data_root, out_dir, fold):
    """Split COCO data for Semi-supervised object detection.

    Args:
        data_root (str): The data root of coco dataset.
        out_dir (str): The output directory of coco semi-supervised
            annotations.
        percent (float): The percentage of labeled data in the training set.
        fold (int): The fold of dataset and set as random seed for data split.
    """



    def save_anns(name, images, annotations):
        sub_anns = dict()
        sub_anns['info'] = anns['info'] #데이터셋 정보
        sub_anns['licenses'] = anns['licenses'] #CC BY 4.0
        sub_anns['images'] = images #이미지 너비,높이, 파일명, 라이센스, url 2개, 캡처시기,id
        sub_anns['categories'] = anns['categories'] # class label 10개
        sub_anns['annotations'] = annotations #id, image_id, category_id, segmentation, area, bbox, iscrowd

        mmcv.mkdir_or_exist(out_dir)
        mmcv.dump(sub_anns, f'{out_dir}/{name}.json')

    # set random seed with the fold

    #나중에 진짜 TRAIN 할때 SEED 해제
    random.seed(42)
    anns = mmcv.load(data_root)
    var = [(ann['image_id'], ann['category_id']) for ann in anns['annotations']]
    X = np.ones((len(anns['annotations']),1)) #
    y = np.array([v[1] for v in var]) #category_id
    groups = np.array([v[0] for v in var]) #image_id
    cv = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=42)
    image_list = anns['images']
    k=0
    for train_idx, _ in cv.split(X, y, groups):
        labeled_inds=set(groups[train_idx])
        labeled_ids, labeled_images, validation_images = [], [], []
        for i in range(len(image_list)):
            if i in labeled_inds:
                labeled_images.append(image_list[i])
                labeled_ids.append(image_list[i]['id'])
            else:
                validation_images.append(image_list[i])

        # get all annotations of labeled images
        labeled_ids = set(labeled_ids)
        labeled_annotations, validation_annotation = [], []

        for ann in anns['annotations']:
            if ann['image_id'] in labeled_ids: #id 별 split 과정
                labeled_annotations.append(ann)
            else:
                validation_annotation.append(ann)

        # save train, validation file based on percentage
        labeled_name = f'train_fold_{k+1}_of_{args.fold}'
        validation_name = f'validation_fold_{k+1}_of_{args.fold}'

        save_anns(labeled_name, labeled_images, labeled_annotations)
        save_anns(validation_name, validation_images, validation_annotation)
        k+=1


if __name__=="__main__":
    args = parse_args()
    split_coco_stratified(args.data_root, args.out_dir,args.fold)