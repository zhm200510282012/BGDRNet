"""Under-0.30M glass models for a staged, single-seed development screen."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2

VARIANTS = {
    'S0': {'complementary': False, 'route': False, 'soft': False},
    'S1': {'complementary': True, 'route': False, 'soft': False},
    'S2': {'complementary': True, 'route': True, 'soft': False},
    'S3': {'complementary': True, 'route': True, 'soft': True},
}


def dsconv(channels=32):
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                         nn.Conv2d(channels, channels, 1, bias=False),
                         nn.GroupNorm(8, channels), nn.ReLU(inplace=False))


class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(32,32,3,padding=1,groups=32,bias=False)
        self.dilated = nn.Conv2d(32,32,3,padding=3,dilation=3,groups=32,bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(96,32,1,bias=False),nn.GroupNorm(8,32),nn.ReLU(inplace=False))

    def forward(self,x):
        pooled=F.adaptive_avg_pool2d(x,1).expand_as(x)
        return x+self.fuse(torch.cat([self.local(x),self.dilated(x),pooled],dim=1))


class Cell(nn.Module):
    def __init__(self):
        super().__init__()
        self.region=dsconv()
        self.detail=dsconv()


class TinyGlass(nn.Module):
    def __init__(self,variant,weights=None,training_heads=True):
        super().__init__()
        self.variant=variant
        self.options=VARIANTS[variant]
        self.training_heads=training_heads
        backbone=mobilenet_v2(weights=None)
        if weights is not None:
            backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.encoder=backbone.features[:11]
        self.projections=nn.ModuleList([
            nn.Sequential(nn.Conv2d(c,32,1,bias=False),nn.GroupNorm(8,32)) for c in [16,24,32,64]])
        self.context=Context()
        self.cells=nn.ModuleList([Cell() for _ in range(3)])
        self.mask_head=nn.Conv2d(32,1,1)
        # Append all auxiliary/gate heads after common modules so seed-matched common
        # tensors have identical initialization even for S0, which has no such heads.
        use_edges=self.options['route'] or (training_heads and self.options['complementary'])
        self.edge_heads=nn.ModuleList([
            nn.Sequential(nn.Conv2d(64,8,1),nn.ReLU(inplace=False),nn.Conv2d(8,1,1))
            for _ in range(3)]) if use_edges else nn.ModuleList()
        self.body_heads=nn.ModuleList([nn.Conv2d(32,1,1) for _ in range(3)]) if training_heads and self.options['complementary'] else nn.ModuleList()
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        return self

    def forward(self,x,with_aux=False):
        hw=x.shape[-2:]
        z=(x-self.mean)/self.std
        features=[]
        for i,layer in enumerate(self.encoder):
            z=layer(z)
            if i in [1,3,6,10]: features.append(self.projections[len(features)](z))
        h=self.context(features[-1])
        bodies,edges=[],[]
        for j,i in enumerate([2,1,0]):
            high=F.interpolate(h,size=features[i].shape[-2:],mode='nearest')
            region=self.cells[j].region(high)
            source=features[i]-high if self.options['route'] else features[i]
            detail=self.cells[j].detail(source)
            edge=None
            if self.options['route'] or (with_aux and self.options['complementary']):
                edge=self.edge_heads[j](torch.cat([region,detail],dim=1))
            h=region+edge.sigmoid().detach()*detail if self.options['route'] else region+detail
            if with_aux and self.options['complementary']:
                edges.append(edge)
                bodies.append(self.body_heads[j](region))
        logits=F.interpolate(self.mask_head(h),size=hw,mode='nearest')
        return {'logits':logits,'body':bodies,'edge':edges} if with_aux else logits

    def deployment_copy(self):
        deployed=TinyGlass(self.variant,weights=None,training_heads=False)
        state=self.state_dict()
        expected=deployed.state_dict()
        deployed.load_state_dict({key:state[key] for key in expected},strict=True)
        return deployed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def boundary_band(mask,radius):
    kernel=2*radius+1
    dilation=F.max_pool2d(mask,kernel,stride=1,padding=radius)
    erosion=-F.max_pool2d(-mask,kernel,stride=1,padding=radius)
    return (dilation-erosion).clamp(0,1)


@torch.no_grad()
def auxiliary_targets(mask,shapes,soft,radius):
    full=boundary_band(mask,radius) if soft else None
    outputs=[]
    for hw in shapes:
        region=F.interpolate(mask.float(),size=hw,mode='area')
        if soft:
            edge=F.interpolate(full,size=hw,mode='area')
        else:
            scaled_radius=max(1,round(radius*hw[-1]/mask.shape[-1]))
            edge=boundary_band((region>=.5).float(),scaled_radius)
        outputs.append((region,edge))
    return outputs


def soft_iou_loss(logits,target):
    probability=logits.float().sigmoid()
    intersection=(probability*target).sum(dim=(1,2,3))
    union=(probability+target-probability*target).sum(dim=(1,2,3))
    return (1-(intersection+1e-6)/(union+1e-6)).mean()


def loss_bundle(model,x,mask,cfg):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(x,with_aux=True)
        logits=output['logits']
        bce=F.binary_cross_entropy_with_logits(logits,mask)
        iou=soft_iou_loss(logits,mask)
        region_loss=logits.float().new_zeros(())
        edge_loss=logits.float().new_zeros(())
        targets=[]
        if model.options['complementary']:
            targets=auxiliary_targets(mask,[e.shape[-2:] for e in output['edge']],model.options['soft'],cfg['boundary_radius'])
            region_terms=[]; edge_terms=[]
            for region_logits,edge_logits,(region,edge) in zip(output['body'],output['edge'],targets):
                weight=1-edge
                region_terms.append((F.binary_cross_entropy_with_logits(region_logits.float(),region,reduction='none')*weight).sum()/weight.sum().clamp_min(1))
                ep=edge_logits.float().sigmoid()
                dice=1-(2*(ep*edge).sum(dim=(1,2,3))+1e-6)/(ep.sum(dim=(1,2,3))+edge.sum(dim=(1,2,3))+1e-6)
                edge_terms.append(F.binary_cross_entropy_with_logits(edge_logits.float(),edge)+dice.mean())
            region_loss=torch.stack(region_terms).mean()
            edge_loss=torch.stack(edge_terms).mean()
        total=bce+iou+cfg['region_weight']*region_loss+cfg['edge_weight']*edge_loss
    if not torch.isfinite(total): raise RuntimeError('Nonfinite training loss')
    meta={'bce':float(bce.detach()),'iou_loss':float(iou.detach()),'region_loss':float(region_loss.detach()),'edge_loss':float(edge_loss.detach()),
          'edge_target_mean':float(torch.stack([t[1].mean() for t in targets]).mean()) if targets else None,
          'soft_fraction_count':sum(int(((t[1]>0)&(t[1]<1)).sum()) for t in targets) if targets else 0}
    return total,meta
