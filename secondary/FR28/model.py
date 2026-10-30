"""FR28: first fusion keeps a normalized value path alongside its product."""
import torch
from torch import nn
from torch.nn import functional as F
from fr12_frozen import TinyGlass as FR12Base,ProductGate
from fr12_frozen import auxiliary_targets,boundary_band,soft_iou_loss,loss_bundle
VARIANTS={v:{'complementary':True,'route':True,'soft':False} for v in ['FR12','FR21','FR28']}
class ValueProduct(ProductGate):
    def forward(self,x):
        u,v=x.chunk(2,dim=1)
        if self.diagnostic_mean_gate:v=v.mean((2,3),keepdim=True)
        return (u+u*v)*(2**-.5)

class TinyGlass(FR12Base):
    def __init__(self,variant,weights=None,training_heads=True):
        assert variant in VARIANTS
        super().__init__('FR12',weights,training_heads);self.variant=variant
        if variant in ['FR21','FR28']:
            # Preserve the inherited initialization and the caller's RNG stream.
            # The zero output starts at FR12; the input receives task gradients
            # only after the output projection leaves zero.
            with torch.random.fork_rng(devices=[]):
                self.region_low_projection=nn.Sequential(
                    nn.Conv2d(32,4,1,bias=False),nn.Conv2d(4,32,1,bias=False))
                nn.init.zeros_(self.region_low_projection[1].weight)
        if variant=='FR28':self.fusers[0].correction[2]=ValueProduct()
    def _forward_core(self,x,with_aux=False,capture=False,probe=None):
        hw=x.shape[-2:];z=(x-self.mean)/self.std;features=[];trace={} if capture else None
        for i,layer in enumerate(self.encoder):
            z=layer(z)
            if i in [1,3,6,10]:features.append(self.projections[len(features)](z))
        pre=features[-1] if probe=='no_context' else self.context(features[-1])
        b,c,height,width=pre.shape;tokens=pre.flatten(2).transpose(1,2)
        attended=self.attention(tokens).transpose(1,2).reshape(b,c,height,width).contiguous()
        semantic=pre if probe=='no_attention' else attended+self.attention_delta.tanh()*(attended-pre)
        h=self.to_decoder(semantic)
        if capture:
            trace.update(pre=pre,attended=attended,semantic=semantic,adapted=h)
            trace.update({f'feature{i}':a for i,a in enumerate(features)})
        bodies=[];edges=[]
        for j,i in enumerate([2,1,0]):
            high=F.interpolate(h,size=features[i].shape[-2:],mode='nearest')
            region_input=high
            reference_high=high
            difference=features[i]-high
            region=self.cells[j].region(high);detail=self.cells[j].detail(difference)
            if self.variant in ['FR21','FR28'] and j==0:
                injection=(torch.zeros_like(region) if probe=='no_region_low'
                    else self.region_low_projection(features[i]).to(region.dtype))
                region=region+injection
                if capture:trace.update(region_injection=injection)
            if capture:trace[f'region_input{j}']=region_input
            edge=self.edge_heads[j](torch.cat([region,detail],1));g=edge.sigmoid().detach();chosen=g*detail
            used_region=torch.zeros_like(region) if probe==f'no_region{j}' else region
            used_detail=torch.zeros_like(chosen) if probe==f'no_detail{j}' else chosen
            fusion=self.fusers[j](torch.cat([used_region,used_detail],1))
            if probe==f'no_fuser{j}':fusion=torch.zeros_like(fusion)
            h=used_region+used_detail+fusion
            if capture:
                trace.update({f'high{j}':high,f'reference_high{j}':reference_high,f'difference{j}':difference,
                    f'region{j}':region,f'detail{j}':detail,f'gate{j}':g,f'chosen{j}':chosen,f'fusion{j}':fusion,f'decoded{j}':h})
            if with_aux and self.options['complementary']:
                bodies.append(self.body_heads[j](region));edges.append(edge)
        logits=F.interpolate(self.mask_head(h),size=hw,mode='nearest')
        if with_aux or capture:
            out={'logits':logits,'body':bodies,'edge':edges}
            if capture:out['trace']=trace
            return out
        return logits
    def forward(self,x,with_aux=False,capture=False,probe=None):
        if not(capture or probe in ['no_nonlinear_fusion','mean_product_gate']):
            return self._forward_core(x,with_aux,capture,probe)
        fuser=self.fusers[0];op=fuser.correction[2]
        old=(fuser.diagnostic_bypass,op.diagnostic_mean_gate);saved={};handles=[]
        fuser.diagnostic_bypass=probe=='no_nonlinear_fusion';op.diagnostic_mean_gate=probe=='mean_product_gate'
        if capture and not fuser.diagnostic_bypass:
            def correction_hook(module,inputs,output):saved.update(nonlinear_input=inputs[0],nonlinear_output=output)
            def product_hook(module,inputs,output):
                u,v=inputs[0].chunk(2,1);saved.update(product_value=u,product_gate=v,product_output=output)
            handles=[fuser.correction.register_forward_hook(correction_hook),op.register_forward_hook(product_hook)]
        try:
            out=self._forward_core(x,with_aux,capture,probe)
            if capture:out['trace'].update(saved)
            return out
        finally:
            fuser.diagnostic_bypass,op.diagnostic_mean_gate=old
            for handle in handles:handle.remove()
    def deployment_copy(self):
        m=TinyGlass(self.variant,training_heads=False);state=self.state_dict()
        m.load_state_dict({k:state[k] for k in m.state_dict()},strict=True);return m
def count_parameters(model):return sum(p.numel() for p in model.parameters())
