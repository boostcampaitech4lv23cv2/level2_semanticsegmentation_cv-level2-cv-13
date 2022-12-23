from torchvision import models
import torch.nn as nn

class FCN_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.base_model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    def forward(self, x):
        x=self.base_model(x)
        return x
    