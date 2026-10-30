"""R1 fusion combined with one/two standard S2A-style attention blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from pilot_frozen import TinyGlass as PilotBase, NarrowAttentionBlock
from s2_frozen import auxiliary_targets, boundary_band, soft_iou_loss, loss_bundle

VARIANTS = {v: {'complementary': True, 'route': True, 'soft': False}
            for v in ['FNA1', 'FNA2']}


class StandardBlock(nn.Module):
    is_coarse_attention_block = True

    def __init__(self, width=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(32)
        self.qkv = nn.Linear(32, 96)
        self.proj = nn.Linear(32, 32)
        self.norm2 = nn.LayerNorm(32)
        self.ffn = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, 32))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, 2, 16).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * 16 ** -.5).float().softmax(-1).to(v.dtype)
        x = x + self.proj((attention @ v).transpose(1, 2).reshape(b, n, c))
        return x + self.ffn(self.norm2(x))


class TinyGlass(PilotBase):
    def __init__(self, variant, weights=None, training_heads=True):
        assert variant in VARIANTS
        # Build S2A first to preserve the entire original base and first attention
        # projections. Only FNA2's first FFN width and second block are new.
        super().__init__('S2A', weights=weights, training_heads=training_heads)
        self.variant = variant
        first = self.attention
        if variant == 'FNA2':
            first.ffn = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))
            second = StandardBlock(128)
            self.attention = nn.Sequential(first, second)
        else:
            self.attention = nn.Sequential(first)
        self.fusers = nn.ModuleList([nn.Conv2d(64, 32, 1, bias=False) for _ in range(3)])
        for layer in self.fusers:
            nn.init.zeros_(layer.weight)

    def forward(self, x, with_aux=False, bypass_fusion=False):
        hw = x.shape[-2:]
        z = (x - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.encoder):
            z = layer(z)
            if i in [1, 3, 6, 10]:
                features.append(self.projections[len(features)](z))
        h = self.context(features[-1])
        b, c, height, width = h.shape
        h = self.attention(h.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, c, height, width).contiguous()
        bodies, edges = [], []
        for j, i in enumerate([2, 1, 0]):
            high = F.interpolate(h, size=features[i].shape[-2:], mode='nearest')
            region = self.cells[j].region(high)
            detail = self.cells[j].detail(features[i] - high)
            edge = self.edge_heads[j](torch.cat([region, detail], dim=1))
            chosen = edge.sigmoid().detach() * detail
            h = region + chosen
            if not bypass_fusion:
                h = h + self.fusers[j](torch.cat([region, chosen], dim=1))
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
