import argparse
import os
import json
import shutil
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

ROOT = '/opt/ml/input/data'
TRAINJSON = '/opt/ml/input/data/train.json'
VALJSON = '/opt/ml/input/data/val.json'
TESTJSON = '/opt/ml/input/data/test.json'

category_names = [
    'Backgroud',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing'
    ]

# 다음 구조에 맞춰 폴더 생성
'''
/opt/ml/input/data/mmseg
/opt/ml/input/data/mmseg/images/training
/opt/ml/input/data/mmseg/images/validation
/opt/ml/input/data/mmseg/annotations/training
/opt/ml/input/data/mmseg/annotations/validation
/opt/ml/input/data/mmseg/test
'''
dir_lists = [
    os.path.join(ROOT, 'mmseg', 'images', 'training'),
    os.path.join(ROOT, 'mmseg', 'images', 'validation'),
    os.path.join(ROOT, 'mmseg', 'annotations', 'training'),
    os.path.join(ROOT, 'mmseg', 'annotations', 'validation'),
    os.path.join(ROOT, 'mmseg','images', 'test')
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert custom dataset to mmsegmentation format')
    parser.add_argument('--root_path', default=ROOT, help='the path of root dir')
    parser.add_argument('--train_path', default=TRAINJSON, help='the path of train.json')
    parser.add_argument('--val_path', default=VALJSON, help='the path of val.json')
    parser.add_argument('--test_path', default=TESTJSON, help='the path of test.json')
    args = parser.parse_args()
    return args


'''
mmsegmentation format에 맞춰 폴더 생성
'''
def mkdir_or_exist(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    print(f'mkdir {dir_path} Success!')


'''
json_path에 있는 사진을 mmsegmentation format에 해당하는 폴더로 복사
flag = train, val, test를 옵션으로 가진다.
'''    
def copy_images(json_path, target_path):
    with open(json_path, "r", encoding="utf8") as f:
        json_data = json.load(f)
    
    images = json_data["images"]
    for image in images:
        shutil.copyfile(os.path.join(ROOT, image["file_name"]), os.path.join(target_path, f"{image['id']:04}.jpg"))
    print(f'image copy to {target_path} Success!')

'''
dataset에 있는 annotations을 이용한 masks 생성
'''
def create_mask(json_path, target_path):
    dataset = CustomDataLoader(data_dir=json_path, mode='train', transform=A.Compose([ToTensorV2()]))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              collate_fn=collate_fn
                                            )
    
    for _, masks, image_infos in data_loader:
        mask = masks[0].numpy()
        image_info = image_infos[0]        
        cv2.imwrite(os.path.join(target_path, f"{image_info['id']:04}.png"), mask)
    print(f'create mask to {target_path} Success!')

def collate_fn(batch):
    return tuple(zip(*batch))

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(ROOT, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

def main():
    args = parse_args()

    # make directories
    for dir in dir_lists:
        mkdir_or_exist(dir)

    # copy images
    copy_images(TRAINJSON, dir_lists[0])
    copy_images(VALJSON, dir_lists[1])
    copy_images(TESTJSON, dir_lists[-1])

    # create masks (annotations)
    create_mask(TRAINJSON, dir_lists[2])
    create_mask(VALJSON, dir_lists[3])
    print('Done!')

if __name__ == '__main__':
    main()