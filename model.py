import torch
from torch import nn


class DG_MMT(nn.Module):
    def __init__(self, num_classes=9, d_model=64, nhead=4, num_layers=2,
                 *, dynamic_gate=True, conv_channels=(16, 32), ffn_dim=128,
                 encoder_dropout=0.0, classifier_dropout=0.3):
        super().__init__()
        if num_classes not in (3, 9):
            raise ValueError('The primary protocol supports 3 or 9 classes.')
        if (d_model, nhead, num_layers) != (64, 4, 2):
            raise ValueError('Manuscript architecture: d_model=64, heads=4, layers=2.')
        self.dynamic_gate = dynamic_gate
        if len(conv_channels) != 2 or any(type(c) is not int or c <= 0 for c in conv_channels):
            raise ValueError('conv_channels requires two positive integer widths.')
        if type(ffn_dim) is not int or ffn_dim <= 0:
            raise ValueError('ffn_dim must be a positive integer.')
        if not 0 <= encoder_dropout < 1 or classifier_dropout != 0.3:
            raise ValueError('Encoder dropout must be in [0,1); manuscript classifier dropout is 0.3.')
        c1, c2 = conv_channels

        def branch(kernel):
            return nn.Sequential(
                nn.Conv1d(4, c1, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c1), nn.ReLU(),
                nn.Conv1d(c1, c2, 3, padding=1),
                nn.BatchNorm1d(c2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

        self.semg_conv = branch(3)
        self.us_conv = branch(5)
        self.semg_proj = nn.Linear(c2, 64)
        self.us_proj = nn.Linear(c2, 64)
        self.pos_embed = nn.Parameter(torch.randn(1, 8, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=ffn_dim, dropout=encoder_dropout,
            activation='relu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(8 * 64, 64), nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes))

        if dynamic_gate:
            self.gating_network = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, semg_seq, us_seq, return_gating=False):
        for x, shape in ((semg_seq, (4, 4, 50)), (us_seq, (4, 4, 128))):
            if x.ndim != 4 or tuple(x.shape[1:]) != shape:
                raise ValueError(f'Expected [batch, {shape}], received {tuple(x.shape)}')
            if not x.is_floating_point() or not torch.isfinite(x).all():
                raise ValueError('Inputs must be finite floating-point tensors.')
        if semg_seq.shape[0] != us_seq.shape[0] or semg_seq.shape[0] == 0:
            raise ValueError('Inputs require equal, nonempty batches.')

        es = torch.stack([self.semg_proj(self.semg_conv(semg_seq[:, i]).squeeze(-1))
                          for i in range(4)], dim=1)
        ea = torch.stack([self.us_proj(self.us_conv(us_seq[:, i]).squeeze(-1))
                          for i in range(4)], dim=1)
        if self.dynamic_gate:
            g = self.gating_network(torch.cat((es.mean(1), ea.mean(1)), dim=-1))
            fused = torch.cat((g.unsqueeze(-1) * es, (1-g).unsqueeze(-1) * ea), dim=1)
        else:
            g = None
            fused = torch.cat((es, ea), dim=1)
        logits = self.classifier(self.transformer(fused + self.pos_embed))
        return (logits, g) if return_gating else logits

    def predict_proba(self, semg_seq, us_seq):
        if self.training:
            raise RuntimeError('Call model.eval() before inference.')
        with torch.no_grad():
            return self(semg_seq, us_seq).softmax(dim=-1)
