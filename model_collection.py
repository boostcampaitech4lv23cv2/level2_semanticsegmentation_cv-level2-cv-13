from torchvision import models
import torch.nn as nn
import segmentation_models_pytorch as smp
from mmcv.utils import Config
from mmseg.models import build_segmentor
from torch.nn.parallel import DistributedDataParallel as DDP
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

##mmseg config
##backbone : ResNet, ResNext, HRNet, ResNeSt, MobileNet V2,V3, Vision Transformer, Swin Transformer, Twins, BEit, ConvNext, MAE, PoolFormer
##models : FCN, ERFNet, Unet, PSPNet, DeepLabV3, V3+, BiSeNetV1, PSANet, UPerNet, ICNet, NonLocalNet, EncNet, Semantic FPN,
#          DANet, APCNet, EMANet, CCNet, ANN, GCNet, FastFCN, Fast-SCNN, ISANet, OCRNet, DNLNet, PointRend, CGNet, BiSeNetV2, STDC, SETR, DPT,
#          Segmenter, SegFormer, K-Net                       
# class BEiT(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()     
#         cfg=Config.fromfile('/opt/ml/mmsegmentation/configs/beit/upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py')
#         cfg.model.decode_head.num_classes=11
#         cfg.model.auxiliary_head.num_classes=11
#         cfg.model.backbone.img_size=(512,512)
#         cfg.norm_cfg.type='BN'
#         self.model=build_segmentor(
#             cfg.model,
#             train_cfg=cfg.get('train_cfg'),
#             test_cfg=cfg.get('test_cfg'))
#     def forward(self,x):
#         img_meta = {
#             'ori_shape': (512, 512, 3),
#             'img_shape': (512, 512, 3),
#             'pad_shape': (512, 512, 3),
#             'scale_factor': [1., 1., 1., 1.],
#             'flip': False,
#             'img_norm_cfg': {
#                 'mean': [123.675, 116.28 , 103.53 ],
#                 'std': [58.395, 57.12 , 57.375],
#                 'to_rgb': True
#             },
#             'batch_input_shape': (512, 512)
#             }
#         return self.model.simple_test_logits(x,img_meta=[img_meta])