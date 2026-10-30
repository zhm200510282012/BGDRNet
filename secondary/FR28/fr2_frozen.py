"""FNA1 recall candidates: raw-detail fusion and bounded context balancing."""
import torch
from torch import nn
from torch.nn import functional as F
from fna_frozen import TinyGlass as FNABase
from s2_frozen import auxiliary_targets, boundary_band, soft_iou_loss, loss_bundle

VARIANTS = {v: {'complementary': True, 'route': True, 'soft': False}
            for v in ['FR1', 'FR2', 'FR3']}


class TinyGlass(FNABase):
    def __init__(self, variant, weights=None, training_heads=True):
        assert variant in VARIANTS
        super().__init__('FNA1', weights=weights, training_heads=training_heads)
        self.variant = variant
        self.raw_detail_fusion = variant in ['FR1', 'FR3']
        self.balance_context = variant in ['FR2', 'FR3']
        if self.balance_context:
            # Effective attention residual scale is 1+tanh(delta), in [0,2].
            # The zero correction preserves the original FNA1 AMP function.
            self.attention_delta = nn.Parameter(torch.zeros(1, 32, 1, 1))

    def forward(self, x, with_aux=False):
        hw = x.shape[-2:]
        z = (x - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.encoder):
            z = layer(z)
            if i in [1, 3, 6, 10]:
                features.append(self.projections[len(features)](z))
        pre = self.context(features[-1])
        b, c, h, w = pre.shape
        attended = self.attention(pre.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w).contiguous()
        h = attended
        if self.balance_context:
            h = attended + self.attention_delta.tanh() * (attended - pre)
        bodies, edges = [], []
        for j, i in enumerate([2, 1, 0]):
            high = F.interpolate(h, size=features[i].shape[-2:], mode='nearest')
            region = self.cells[j].region(high)
            detail = self.cells[j].detail(features[i] - high)
            edge = self.edge_heads[j](torch.cat([region, detail], dim=1))
            chosen = edge.sigmoid().detach() * detail
            fusion_detail = detail if self.raw_detail_fusion else chosen
            h = region + chosen + self.fusers[j](torch.cat([region, fusion_detail], dim=1))
            if with_aux and self.options['complementary']:
                bodies.append(self.body_heads[j](region))
                edges.append(edge)
        logits = F.interpolate(self.mask_head(h), size=hw, mode='nearest')
        return {'logits': logits, 'body': bodies, 'edge': edges} if with_aux else logits

    def deployment_copy(self):
        deployed = TinyGlass(self.variant, training_heads=False)
        current = self.state_dict()
        deployed.load_state_dict({k: current[k] for k in deployed.state_dict()}, strict=True)
        return deployed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
