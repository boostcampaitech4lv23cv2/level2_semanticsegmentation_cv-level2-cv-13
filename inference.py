import os
import torch
import wandb
import sweeps.model_collection as model_collection
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataLoader
import pandas as pd

def test(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def collate_fn(batch):
        return tuple(zip(*batch))

# best model 저장된 경로
model_folder_path = '/opt/ml/saved/[001]-FCN_ResNet50-CrossEntropyLoss-Adam-CosineAnnealingWarmRestarts-2hu2oor6'
model_name="0.3311562325088047_0.pth"

run_name=model_folder_path.split("/")[-1]
dataset_path  = '/opt/ml/input/data'
device = "cuda" if torch.cuda.is_available() else "cpu" 
test_path = dataset_path + '/test.json'

# best model 불러오기
api = wandb.Api()
# run = api.run("boostcamp_cv13/Semantic_Segmentation/"+run_name)
run = api.run("boostcamp_cv13/Semantic_Segmentation/"+model_folder_path.split("-")[-1])
state_dict = torch.load(os.path.join(model_folder_path,model_name), map_location=device)
model=getattr(model_collection,run.config['model'])()
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

test_transform = A.Compose([
                        ToTensorV2()
                        ])

test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform,dataset_path=dataset_path)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=run.config['batch_size'],
                                        num_workers=4,
                                        collate_fn=collate_fn)

# sample_submisson.csv 열기
submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)

# test set에 대한 prediction
file_names, preds = test(model, test_loader, device)

# PredictionString 대입
for file_name, string in zip(file_names, preds):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv(os.path.join(model_folder_path,model_name.split(".")[0])+"_"+model_folder_path.split("/")[-1][1:4]+"_"+".csv", index=False)