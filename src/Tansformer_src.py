import torch
import torch.nn as nn
import math

def masked_mse(pred, target, mask, eps=1e-8):
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom

@torch.no_grad()
def masked_pearson(pred, target, mask, eps=1e-8):
    pred = pred[mask.bool()]
    target = target[mask.bool()]
    if pred.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)

    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.std(unbiased=False) * target.std(unbiased=False)).clamp_min(eps)
    return (pred * target).mean() / denom


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) to broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class TransformerRSA(nn.Module):
    """
    Transformer encoder per-residue RSA regressor.
    Input:  emb (B, L, D) padded
            lengths (B,) true lengths
    Output: pred (B, L) raw
    """
    def __init__(
        self,
        d_in=640,
        d_model=96,
        n_heads=4,
        n_layers=1,
        dim_ff=256,
        dropout=0.1,
        mlp_hidden=64,
        use_input_layernorm=True,
        max_len=10000,
    ):
        super().__init__()

        self.use_input_layernorm = use_input_layernorm
        if use_input_layernorm:
            self.ln_in = nn.LayerNorm(d_in)

        # project embeddings to transformer model dim
        self.in_proj = nn.Linear(d_in, d_model)

        # positional encoding
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # tends to be stable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # per-residue regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb, lengths):
        """
        emb: (B, L, D)
        lengths: (B,)
        """
        if self.use_input_layernorm:
            emb = self.ln_in(emb)

        x = self.in_proj(emb)     # (B, L, d_model)
        x = self.pos(x)           # add pos enc + dropout

        # key padding mask: True where PAD (positions >= length)
        B, L, _ = x.shape
        device = x.device
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        key_padding_mask = arange >= lengths.unsqueeze(1)  # (B, L) bool

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        pred = self.head(x).squeeze(-1)  # (B, L)
        return pred


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        emb = batch["emb"].to(device).float()          # (B, L, D)
        y   = batch["y"].to(device).float()            # (B, L)
        m   = batch["mask"].to(device).float()         # (B, L)
        lengths = batch["lengths"].to(device)          # (B,)

        pred = model(emb, lengths)                     # raw (B, L)
        loss = masked_mse(pred, y, m)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, clamp_for_metrics=True):
    model.eval()
    total_loss = 0.0
    total_r = 0.0
    n_r = 0

    for batch in loader:
        emb = batch["emb"].to(device).float()
        y   = batch["y"].to(device).float()
        m   = batch["mask"].to(device).float()
        lengths = batch["lengths"].to(device)

        pred_raw = model(emb, lengths)                 # (B, L)
        pred = pred_raw.clamp(0.0, 1.0) if clamp_for_metrics else pred_raw

        loss = masked_mse(pred, y, m)
        r = masked_pearson(pred, y, m)

        total_loss += loss.item()
        if torch.isfinite(r):
            total_r += r.item()
            n_r += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_r = total_r / max(1, n_r) if n_r > 0 else float("nan")
    return avg_loss, avg_r
