"""FR11 with one first-stage, two-projection product correction (FR12)."""
import torch
from torch import nn
from fr11_frozen import TinyGlass as FR11Base
from fr11_frozen import auxiliary_targets,boundary_band,soft_iou_loss,loss_bundle

VARIANTS={v:{'complementary':True,'route':True,'soft':False} for v in ['FR11','FR12']}

class ProductGate(nn.Module):
    def __init__(self):
        super().__init__();self.diagnostic_mean_gate=False
    def forward(self,x):
        value,gate=x.chunk(2,dim=1)
        if self.diagnostic_mean_gate:gate=gate.mean((2,3),keepdim=True)
        return value*gate

class TinyGlass(FR11Base):
    def __init__(self,variant,weights=None,training_heads=True):
        assert variant in VARIANTS
        super().__init__('FR11',weights,training_heads)
        self.variant=variant
        if variant=='FR12':
            old=self.fusers[0].correction
            # Preserve the first projection slice and output projection. The
            # normalized group size remains four channels in both branches.
            new=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.GroupNorm(16,64),ProductGate(),nn.Conv2d(32,32,1,bias=False))
            with torch.no_grad():
                new[0].weight[:32].copy_(old[0].weight)
                new[1].weight[:32].copy_(old[1].weight)
                new[1].bias[:32].copy_(old[1].bias)
                new[3].weight.copy_(old[3].weight)
            self.fusers[0].correction=new

    def forward(self,x,with_aux=False,capture=False,probe=None):
        if self.variant!='FR12':return super().forward(x,with_aux,capture,probe)
        gate=self.fusers[0].correction[2]
        if not (capture or probe=='mean_product_gate'):
            return super().forward(x,with_aux,capture,probe)
        old=gate.diagnostic_mean_gate;handle=None;saved={}
        gate.diagnostic_mean_gate=probe=='mean_product_gate'
        if capture:
            def hook(module,inputs,output):
                a,b=inputs[0].chunk(2,dim=1)
                saved.update(product_value=a,product_gate=b,product_output=output)
            handle=gate.register_forward_hook(hook)
        try:
            out=super().forward(x,with_aux,capture,probe)
            if capture:out['trace'].update(saved)
            return out
        finally:
            gate.diagnostic_mean_gate=old
            if handle:handle.remove()

    def deployment_copy(self):
        model=TinyGlass(self.variant,training_heads=False);state=self.state_dict()
        model.load_state_dict({k:state[k] for k in model.state_dict()},strict=True)
        return model

def count_parameters(model):return sum(p.numel() for p in model.parameters())
