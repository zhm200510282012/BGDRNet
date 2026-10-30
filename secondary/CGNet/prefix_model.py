"""Shared ImageNet prefix plus the unmodified torchvision LRASPPHead."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.segmentation.lraspp import LRASPPHead
from prefix_s2_frozen import auxiliary_targets,boundary_band,soft_iou_loss,loss_bundle

VARIANTS={'LRASPP_prefix':{'complementary':False,'route':False,'soft':False}}

class TinyGlass(nn.Module):
    def __init__(self,variant,weights=None,training_heads=True):
        super().__init__();self.variant=variant;self.options=VARIANTS[variant];self.training_heads=training_heads
        backbone=mobilenet_v2(weights=None)
        if weights is not None:backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.encoder=backbone.features[:11]
        self.head=LRASPPHead(low_channels=32,high_channels=64,num_classes=2,inter_channels=128)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def train(self,mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,x,with_aux=False):
        z=(x-self.mean)/self.std;low=None
        for i,layer in enumerate(self.encoder):
            z=layer(z)
            if i==6:low=z
        y=self.head({'low':low,'high':z})
        y=F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=False)
        logits=y[:,1:2]-y[:,0:1]
        return {'logits':logits,'body':[],'edge':[]} if with_aux else logits
    def deployment_copy(self):return copy.deepcopy(self)

def count_parameters(model):return sum(p.numel() for p in model.parameters())
