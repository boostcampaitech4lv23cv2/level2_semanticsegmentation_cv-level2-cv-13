import os.path as osp
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import json

def coco_to_mmsegmentation(
    annotations_file: str, output_annotations_file: str, output_masks_dir: str, mode: str
):
        path = '/opt/ml/input/data'
        if not osp.isdir(output_annotations_file):
            os.mkdir(osp.join(path, output_annotations_file))
        if not osp.isdir(output_masks_dir):
            os.mkdir(osp.join(path, output_masks_dir))
            os.mkdir(osp.join(path, output_masks_dir, 'batch_01_vt'))
            os.mkdir(osp.join(path, output_masks_dir, 'batch_02_vt'))
            os.mkdir(osp.join(path, output_masks_dir, 'batch_03'))


        print(f"Loading annotations form {annotations_file}")
        annotations = json.load(open(annotations_file))

        print(f"Saving annotations to {output_annotations_file}")
        with open(output_annotations_file + f"/{mode}.txt", "w") as f:
            for image in annotations["images"]:
                filename = image["file_name"][:-4]
                f.write(str(filename))
                f.write("\n")

        print(f"Saving masks to {output_masks_dir}")
        coco_annotations = COCO(annotations_file)
        for image_id, image_data in coco_annotations.imgs.items():

            filename = image_data["file_name"]

            anns_ids = coco_annotations.getAnnIds(imgIds=image_id)
            image_annotations = coco_annotations.loadAnns(anns_ids)

            print(f"Creating output mask for {filename}")

            output_mask = np.zeros(
                (image_data["height"], image_data["width"]), dtype=np.uint8
            )
            for image_annotation in image_annotations:
                category_id = image_annotation["category_id"]
                try:
                    category_mask = coco_annotations.annToMask(image_annotation)
                except Exception as e:
                    print(f"Skipping {image_annotation}")
                    continue
                category_mask *= category_id
                category_mask *= output_mask == 0
                output_mask += category_mask

            output_filename = osp.join(output_masks_dir, str(filename[:-4]) + ".png")
            # output_filename.parent.mkdir(parents=True, exist_ok=True)

            print(f"Writting mask to {output_filename}")
            cv2.imwrite(str(output_filename), output_mask)


if __name__ == "__main__":
    for mode in ['train','val', 'test']:
        path = '/opt/ml/input/data'
        json_path = path + f'/{mode}.json'
        labels_path = path + '/labels'
        splits_path = path + '/splits'
        coco_to_mmsegmentation(json_path, splits_path, labels_path,mode)