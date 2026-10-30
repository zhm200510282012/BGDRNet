"""FR2 plus one low-resolution nonlinear fusion residual; no gate/loss change."""
import torch
from torch import nn
from flow_base_frozen import TinyGlass as InstrumentedFR2
from fr2_frozen import auxiliary_targets,boundary_band,soft_iou_loss,loss_bundle

VARIANTS={v:{'complementary':True,'route':True,'soft':False} for v in ['FR2','FR11']}


class FusionResidual(nn.Module):
    def __init__(self,linear):
        super().__init__()
        self.linear=linear
        self.correction=nn.Sequential(nn.Conv2d(64,32,1,bias=False),nn.GroupNorm(8,32),
                                      nn.GELU(),nn.Conv2d(32,32,1,bias=False))
        nn.init.zeros_(self.correction[-1].weight)
        self.diagnostic_bypass=False

    @property
    def weight(self):return self.linear.weight

    def forward(self,x):
        original=self.linear(x)
        return original if self.diagnostic_bypass else original+self.correction(x)


class TinyGlass(InstrumentedFR2):
    def __init__(self,variant,weights=None,training_heads=True):
        assert variant in VARIANTS
        super().__init__('FR2',weights=weights,training_heads=training_heads)
        self.variant=variant
        if variant=='FR11':self.fusers[0]=FusionResidual(self.fusers[0])

    def forward(self,x,with_aux=False,capture=False,probe=None):
        if self.variant=='FR2' or not (capture or probe=='no_nonlinear_fusion'):
            return super().forward(x,with_aux=with_aux,capture=capture,probe=probe)
        saved={};handle=None
        module=self.fusers[0]
        old=module.diagnostic_bypass
        module.diagnostic_bypass=probe=='no_nonlinear_fusion'
        if capture and not module.diagnostic_bypass:
            def hook(layer,inputs,output):saved.update(nonlinear_input=inputs[0],nonlinear_output=output)
            handle=module.correction.register_forward_hook(hook)
        try:
            out=super().forward(x,with_aux=with_aux,capture=capture,probe=probe)
            if capture:out['trace'].update(saved)
            return out
        finally:
            module.diagnostic_bypass=old
            if handle is not None:handle.remove()

    def deployment_copy(self):
        deployed=TinyGlass(self.variant,training_heads=False)
        state=self.state_dict()
        deployed.load_state_dict({key:state[key] for key in deployed.state_dict()},strict=True)
        return deployed


def count_parameters(model):return sum(p.numel() for p in model.parameters())
