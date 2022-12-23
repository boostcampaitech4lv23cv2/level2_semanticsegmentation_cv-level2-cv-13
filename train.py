import json
import os
import torch
import numpy as np
import random
import model_collection
from dataset import CustomDataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import pandas as pd
from utils import label_accuracy_score, add_hist
import wandb
import loss_collection
import scheduler_collection
import optimizer_collection
from tqdm import tqdm

def set_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



def prepare_dataloader():

    cat_names = []
    for cat_it in categories:
        cat_names.append(cat_it['name'])

    cat_histogram = np.zeros(len(categories),dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1         
            
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)
    # category labeling 
    sorted_temp_df = df.sort_index()

    # background = 0 에 해당되는 label 추가 후 기존들을 모두 label + 1 로 설정
    sorted_df = pd.DataFrame(["Backgroud"], columns = ["Categories"])
    sorted_df = sorted_df.append(sorted_temp_df, ignore_index=True)

    category_names = list(sorted_df.Categories)


    def collate_fn(batch):
        return tuple(zip(*batch))


    train_transform = A.Compose([
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])



        
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform,dataset_path=dataset_path,category_names=category_names)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform,dataset_path=dataset_path,category_names=category_names)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(wandb.config.seed)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=wandb.config.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            worker_init_fn=seed_worker,
                                            generator=g
                                            )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=wandb.config.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn,
                                            worker_init_fn=seed_worker,
                                            generator=g
                                            )


    return train_loader,val_loader, sorted_df

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    
    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            log={"Train/mIoU": mIoU,"Train/acc": acc,"Train/acc_mean": acc_cls,"Train/fwavacc":fwavacc,"Train/avg_loss":mIoU}
            wandb.log(log)
        # scheduler.step()     
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            mIoU = validation(epoch , model, val_loader, criterion, device)
            save_model(model, saved_dir,mIoU,epoch)

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [("Val/IoU_"+classes, round(IoU,4)) for IoU, classes in zip(IoU , sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        log={"mIoU": mIoU,"Val/acc": acc,"Val/acc_mean": acc_cls,"Val/fwavacc":fwavacc,"Val/avg_loss":avrg_loss}
        for IoU, classes in zip(IoU , sorted_df['Categories']):
            log["Val/IoU_"+classes]=IoU
        wandb.log(log)
        
    return mIoU



def save_model(model, saved_dir,metric,epoch):
    output_path = os.path.join(saved_dir, this_run_name,)
    if not os.path.isdir(output_path):                                                           
        os.mkdir(output_path)
    torch.save(model.state_dict(), os.path.join(output_path,f"{metric}_{epoch}.pth"))
    files = os.listdir(output_path)
    if len(files)>wandb.config.save_top_k:
        files.sort()
        os.remove(os.path.join(output_path,files[0]))


if __name__=="__main__":
    wandb.login()
    runs=wandb.Api().runs(path="boostcamp_cv13/Semantic_Segmentation",order="created_at")
    try:
        this_run_num=f"{int(runs[0].name[1:4])+1:03d}"
    except:
        this_run_num="000"
    wandb.init(
        entity="boostcamp_cv13",
        project="Semantic_Segmentation",
        config="/opt/ml/level2_semanticsegmentation_cv-level2-cv-13/config-defaults.yaml"
        )
    this_run_name=f"[{this_run_num}]-{wandb.config.model}-{wandb.config.loss}-{wandb.config.optimizer}-{wandb.run.id}"
    wandb.run.name=this_run_name
    wandb.run.save()
    dataset_path  = '/opt/ml/input/data'
    anns_file_path = dataset_path + '/' + 'train_all.json'
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    saved_dir = '/opt/ml/saved'


    model=getattr(model_collection,wandb.config.model)()
    criterion = getattr(loss_collection,wandb.config.loss)()
    optimizer = getattr(optimizer_collection,wandb.config.optimizer)(model)
    # scheduler=getattr(scheduler_collection,wandb.config.scheduler)(optimizer)


    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
        
    categories = dataset['categories']
    anns = dataset['annotations']


    set_seed(wandb.config.seed)
    train_loader,val_loader,sorted_df = prepare_dataloader()
    train(wandb.config.num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, wandb.config.val_every, device)