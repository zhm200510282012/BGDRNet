"""FR2 capacity validation with optional diagnostics, inactive in normal inference."""
import torch
from torch import nn
from torch.nn import functional as F
from fr2_frozen import TinyGlass as FR2Base
from s2_frozen import auxiliary_targets, boundary_band, soft_iou_loss, loss_bundle

VARIANTS={v:{'complementary':True,'route':True,'soft':False} for v in ['FR2','FR7','FR8','FR9']}


class Context(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.local=nn.Conv2d(c,c,3,padding=1,groups=c,bias=False)
        self.dilated=nn.Conv2d(c,c,3,padding=3,dilation=3,groups=c,bias=False)
        self.fuse=nn.Sequential(nn.Conv2d(3*c,c,1,bias=False),nn.GroupNorm(8,c),nn.ReLU(inplace=False))
    def forward(self,x):
        return x+self.fuse(torch.cat([self.local(x),self.dilated(x),F.adaptive_avg_pool2d(x,1).expand_as(x)],1))


class GlobalBlock(nn.Module):
    is_coarse_attention_block=True
    def __init__(self,c,f):
        super().__init__();self.heads=c//16
        self.norm1=nn.LayerNorm(c);self.qkv=nn.Linear(c,3*c);self.proj=nn.Linear(c,c)
        self.norm2=nn.LayerNorm(c);self.ffn=nn.Sequential(nn.Linear(c,f),nn.GELU(),nn.Linear(f,c))
    def forward(self,x,mask_extra=False):
        b,n,c=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,16).permute(2,0,3,1,4).unbind(0)
        a=((q@k.transpose(-2,-1))*16**-.5).float().softmax(-1).to(v.dtype)
        x=x+self.proj((a@v).transpose(1,2).reshape(b,n,c))
        hidden=self.ffn[1](self.ffn[0](self.norm2(x)))
        if mask_extra:
            hidden=torch.cat([hidden[...,:64],torch.zeros_like(hidden[...,64:])],dim=-1)
        return x+self.ffn[2](hidden)


class TinyGlass(FR2Base):
    def __init__(self,variant,weights=None,training_heads=True):
        assert variant in VARIANTS
        super().__init__('FR2',weights=weights,training_heads=training_heads)
        self.variant=variant;self.to_decoder=nn.Identity()
        if variant=='FR7':
            old=self.attention[0]
            block=GlobalBlock(32,256)
            # Existing projections/norms and the first64 FFN units retain fresh
            # FR2 initialization. New FFN output columns start at zero.
            block.norm1,block.qkv,block.proj=old.norm1,old.qkv,old.proj
            block.norm2=old.norm2
            with torch.no_grad():
                block.ffn[0].weight[:64].copy_(old.ffn[0].weight)
                block.ffn[0].bias[:64].copy_(old.ffn[0].bias)
                block.ffn[2].weight.zero_()
                block.ffn[2].weight[:,:64].copy_(old.ffn[2].weight)
                block.ffn[2].bias.copy_(old.ffn[2].bias)
            self.attention=nn.Sequential(block)
        elif variant=='FR8':
            self.projections[3]=nn.Sequential(nn.Conv2d(64,48,1,bias=False),nn.GroupNorm(8,48))
            self.context=Context(48)
            self.attention=nn.Sequential(GlobalBlock(48,96))
            self.attention_delta=nn.Parameter(torch.zeros(1,48,1,1))
            self.to_decoder=nn.Sequential(nn.Conv2d(48,32,1,bias=False),nn.GroupNorm(8,32))
        elif variant=='FR9':
            self.semantic_value=nn.Sequential(nn.Conv2d(32,64,1,bias=False),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,32,1,bias=False))
            self.semantic_gate=nn.Conv2d(32,32,1,bias=False)
            nn.init.zeros_(self.semantic_value[-1].weight)
            nn.init.zeros_(self.semantic_gate.weight)

    def forward(self,x,with_aux=False,capture=False,probe=None):
        hw=x.shape[-2:];z=(x-self.mean)/self.std;features=[]
        trace={} if capture else None
        for i,layer in enumerate(self.encoder):
            z=layer(z)
            if i in [1,3,6,10]:features.append(self.projections[len(features)](z))
        pre=features[-1] if probe=='no_context' else self.context(features[-1])
        b,c,height,width=pre.shape;tokens=pre.flatten(2).transpose(1,2)
        attended=(self.attention[0](tokens,mask_extra=True) if probe=='no_extra_ffn' and self.variant=='FR7' else self.attention(tokens))
        attended=attended.transpose(1,2).reshape(b,c,height,width).contiguous()
        semantic=pre if probe=='no_attention' else attended+self.attention_delta.tanh()*(attended-pre)
        if probe=='zero_coarse_tail16' and self.variant=='FR8':
            semantic=torch.cat([semantic[:,:32],torch.zeros_like(semantic[:,32:])],1)
        h=self.to_decoder(semantic)
        if capture:
            trace.update(pre=pre,attended=attended,semantic=semantic,adapted=h)
            trace.update({f'feature{i}':v for i,v in enumerate(features)})
        if self.variant=='FR9':
            value=self.semantic_value(semantic);gate=2*self.semantic_gate(semantic).sigmoid()
            if capture:trace.update(bridge_value=value,bridge_gate=gate)
        bodies=[];edges=[]
        for j,i in enumerate([2,1,0]):
            high=F.interpolate(h,size=features[i].shape[-2:],mode='nearest')
            region=self.cells[j].region(high);detail=self.cells[j].detail(features[i]-high)
            edge=self.edge_heads[j](torch.cat([region,detail],1))
            g=edge.sigmoid().detach();chosen=g*detail
            used_region=torch.zeros_like(region) if probe==f'no_region{j}' else region
            used_detail=torch.zeros_like(chosen) if probe==f'no_detail{j}' else chosen
            fusion=self.fusers[j](torch.cat([used_region,used_detail],1))
            if probe==f'no_fuser{j}':fusion=torch.zeros_like(fusion)
            h=used_region+used_detail+fusion
            if capture:
                trace.update({f'region{j}':region,f'detail{j}':detail,f'gate{j}':g,f'chosen{j}':chosen,f'fusion{j}':fusion,f'decoded{j}':h})
            if self.variant=='FR9' and j==1 and probe!='no_bridge':
                size=h.shape[-2:]
                upgate=F.interpolate(gate,size=size,mode='nearest')
                upvalue=F.interpolate(value,size=size,mode='nearest')
                if probe=='no_bridge_gate':upgate=torch.ones_like(upgate)
                if probe=='no_bridge_value':upvalue=torch.zeros_like(upvalue)
                h=h*upgate+upvalue
                if capture:trace['after_bridge']=h
            if with_aux and self.options['complementary']:
                bodies.append(self.body_heads[j](region));edges.append(edge)
        logits=F.interpolate(self.mask_head(h),size=hw,mode='nearest')
        if with_aux or capture:
            out={'logits':logits,'body':bodies,'edge':edges}
            if capture:out['trace']=trace
            return out
        return logits

    def deployment_copy(self):
        deployed=TinyGlass(self.variant,training_heads=False)
        current=self.state_dict()
        deployed.load_state_dict({k:current[k] for k in deployed.state_dict()},strict=True)
        return deployed


def count_parameters(model):return sum(p.numel() for p in model.parameters())
