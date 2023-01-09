import json
import os
import os.path as osp

file_path = "/opt/ml/level2_semanticsegmentation_cv-level2-cv-13/stratified_fold_dataset"
new_file_path = ""

file_name_list = os.listdir(file_path)
#print(file_name_list)

for file_name in file_name_list:
    new_path = osp.join(file_path, file_name)
    # path = osp.join(file_path, "new", file_name)
    # print(new_path)
    with open(new_path, "r") as f:
        new_json = json.load(f)


    with open(new_path, "w") as nf:
        json.dump(new_json, nf, indent=4)#, sort_keys=True)
