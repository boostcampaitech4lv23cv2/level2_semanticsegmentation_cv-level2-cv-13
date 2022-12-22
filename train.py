import model
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

class MaskModel(pl.LightningModule):
    def preprocessing(self):

        transform_train=A.Compose([
            A.RandomCrop(320,256),
            A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.Normalize(),  
            ToTensorV2(),
        ])

        transform_val=A.Compose([
            A.CenterCrop(320,256),
            A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
            A.CLAHE(),
            A.Normalize(),
            ToTensorV2(),
        ])

        train_idx,val_idx=SplitByHumanDataset.split_train_val()
        train_dataset=SplitByHumanDataset(train_idx, transform_train,train=True)
        val_dataset=SplitByHumanDataset(val_idx, transform_val,train=False)


        self.train_loader = DataLoader(
            train_dataset,
            batch_size=HyperParameter.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            sampler=Sampler(train_dataset),
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=HyperParameter.BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def log_confusion_matrix(self,metric):
        x=metric.compute().cpu().numpy()
        fig = plt.figure(figsize = (20,10),dpi=100)
        sns.heatmap(x,annot=True,cmap="Blues",fmt='g')
        return fig
        
    def __init__(self, model_name,num_class,learning_rate,loss_funtion_name):
        super().__init__()
        self.model=getattr(model,model_name)()
        self.lr=learning_rate
        self.criterion=getattr(loss_functions,loss_funtion_name)()
        # setup metrics module

        self.save_hyperparameters()
        self.preprocessing()

    def training_step(self, batch, batch_idx):
        inputs,labels = batch
        labels = SplitByHumanDataset.multi_to_single(*labels)
        outs=self.model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss= self.criterion(outs,labels)
        self.train_loss(loss)
        self.train_acc(preds,labels)
        self.train_f1(preds,labels)
        self.train_conf_matrix(preds,labels)
        self.log("Train/loss", self.train_loss,on_step=True,on_epoch=True)
        self.log("Train/acc",self.train_acc,on_step=True,on_epoch=True)
        self.log("Train/f1",self.train_f1,on_step=True,on_epoch=True)
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        tensorboard=self.logger.experiment
        tensorboard.add_figure("Train/confusion",self.log_confusion_matrix(self.train_conf_matrix),self.current_epoch)
        self.train_conf_matrix.reset()
        
    
    def validation_step(self, batch, batch_idx):
        inputs,labels = batch
        labels = SplitByHumanDataset.multi_to_single(*labels)
        outs=self.model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss= self.criterion(outs,labels)
        self.val_loss(loss)
        self.val_acc(preds,labels)
        self.val_f1(preds,labels)
        self.val_conf_matrix(preds,labels)
        self.log("Validation/loss", self.val_loss,on_step=False,on_epoch=True)
        self.log("Validation/acc",self.val_acc,on_step=False,on_epoch=True)
        self.log("Validation/f1",self.val_f1,on_step=False,on_epoch=True)
    
    def validation_epoch_end(self, outputs) -> None:
        tensorboard=self.logger.experiment
        tensorboard.add_figure("Validation/confusion",self.log_confusion_matrix(self.val_conf_matrix),self.current_epoch)
        self.val_conf_matrix.reset()

    def configure_optimizers(self):
        optimizer =torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer