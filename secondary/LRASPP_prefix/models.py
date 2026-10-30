"""Pinned author CGNet and the existing prefix adapter, with binary-logit interfaces."""
import copy
import torch
from torch import nn
from vendor.CGNet import Context_Guided_Network
from prefix_model import TinyGlass as Prefix

NAMES = ['CGNet', 'LRASPP_prefix']

class CGNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Context_Guided_Network(classes=2, M=3, N=21, dropout_flag=False)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self, x):
        scores = self.net((x-self.mean)/self.std)
        return scores[:,1:2]-scores[:,0:1]

def build(name, weights=None):
    if name == 'CGNet':
        assert weights is None
        return CGNetBinary()
    if name == 'LRASPP_prefix':
        return Prefix(name, weights, training_heads=False)
    raise ValueError(name)

def parameters(model):
    return sum(p.numel() for p in model.parameters())

def loss(model, x, y):
    from torch.nn import functional as F
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z=model(x)
        assert z.shape == y.shape
        bce=F.binary_cross_entropy_with_logits(z, y)
        p=z.float().sigmoid()
        inter=(p*y).sum((1,2,3)); union=(p+y-p*y).sum((1,2,3))
        iou=(1-(inter+1e-6)/(union+1e-6)).mean()
        total=bce+iou
    assert torch.isfinite(total)
    return total, {'bce':float(bce.detach()), 'iou_loss':float(iou.detach())}
