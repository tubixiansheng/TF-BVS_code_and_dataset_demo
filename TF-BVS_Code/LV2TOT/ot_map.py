class TemporalPool(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        assert mode in ['mean', 'max']
        self.mode = mode
    def forward(self, v):  # v: [B, T, C]
        if v.dim() == 3:  # Ensure v is [B, T, C]
            if self.mode == 'mean':
                return v.mean(dim=1, keepdim=True)   # [B, 1, C]
            else:
                return v.max(dim=1).values
        else:
            raise ValueError("Input v must have 3 dimensions [B, T, C]")
class OTMap(nn.Module):
    def __init__(self, in_dim, t_dim, num_classes=4, hidden=512, pool='mean'):
        super().__init__()
        self.pool = TemporalPool(mode=pool)
        self.label_embed = nn.Embedding(num_classes, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, t_dim)
        )
    def forward(self, v_seq, labels=None):  # v_seq: [B, T, C]; labels: [B]
        if v_seq.dim() == 5:
            B, C, H, W, T = v_seq.shape
            v_seq = v_seq.flatten(2, 3).permute(0, 4, 1, 2).flatten(2).mean(-1)  # [B, T, C]
        elif v_seq.dim() == 4:
            v_seq = v_seq.mean(dim=[2, 3]).unsqueeze(1)  # [B, 1, C]

        if v_seq.dim() == 3:
            if torch.isnan(v_seq).any() or torch.isinf(v_seq).any():
                print(f"NaN or Inf detected in v_seq before pooling! Shape: {v_seq.shape}")
                v_seq = torch.nan_to_num(v_seq, nan=0.0, posinf=1e10, neginf=-1e10)
            v = self.pool(v_seq)  # [B, 1, C]
        else:
            raise ValueError("v_seq should have 3 dimensions [B, T, C], found: {}".format(v_seq.shape))
        v = v.squeeze(1)  # [B, C]
        v = v / (v.abs().max() + 1e-8)
        label_emb = self.label_embed(labels)  # [B, C]
        v_cat = torch.cat([v, label_emb], dim=-1)  # [B, 2C]
        return self.mlp(v_cat)  # [B, t_dim]