from torchvision import models
import torch.nn as nn
import segmentation_models_pytorch as smp
class FCN_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.base_model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    def forward(self, x):
        x=self.base_model(x)
        return x

def UNet():
    return smp.Unet(in_channels=3, classes= 11)

def UNetPP():
    return smp.UnetPlusPlus(in_channels=3, classes= 11)

def Deeplab():
    return smp.DeepLabV3(in_channels=3,classes= 11)

def DeeplabP():
    return smp.DeepLabV3Plus(in_channels=3,classes= 11)

def PSPNet():
    return smp.PSPNet(in_channels=3, classes= 11)

def Linknet():
    return smp.Linknet(in_channels=3,classes= 11)

def MAnet():
    return smp.MAnet(in_channels=3,classes= 11)

def FPN():
    return smp.FPN(in_channels= 3, classes= 11)