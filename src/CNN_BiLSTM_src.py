import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNN_BiLSTM_Head(nn.Module):
    """
    Hybrid per-residue RSA regressor:
      emb (B, L, D) -> CNN -> BiLSTM (packed) -> MLP -> pred (B, L) raw
    """
    def __init__(
        self,
        d_in=640,
        cnn_hidden=256,
        cnn_layers=2,
        cnn_kernel=5,
        lstm_hidden=256,
        lstm_layers=2,
        mlp_hidden=128,
        dropout=0.2,
        use_layernorm=True,
    ):
        super().__init__()
        assert cnn_kernel % 2 == 1, "Use odd cnn_kernel for same-length padding."

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln_in = nn.LayerNorm(d_in)

        # CNN trunk (local mixing)
        self.cnn_in = nn.Conv1d(d_in, cnn_hidden, kernel_size=1)
        self.cnn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(cnn_layers)
        ])

        # BiLSTM (global context)
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # MLP head (per residue)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb, lengths):
        """
        emb: (B, L, D) padded
        lengths: (B,) true lengths
        """
        if self.use_layernorm:
            emb = self.ln_in(emb)

        # CNN expects (B, D, L)
        x = emb.transpose(1, 2)            # (B, D, L)
        x = self.cnn_in(x)                 # (B, cnn_hidden, L)

        # residual CNN blocks
        for block in self.cnn_blocks:
            r = x
            x = block(x)
            x = x + r

        # back to (B, L, C) for LSTM
        x = x.transpose(1, 2)              # (B, L, cnn_hidden)

        # pack -> BiLSTM -> unpack
        lengths_cpu = lengths.to("cpu")
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, Lmax, 2*lstm_hidden)

        pred = self.head(out).squeeze(-1)  # (B, Lmax) raw
        return pred

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

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        emb = batch["emb"].to(device).float()         # (B, L, D)
        y   = batch["y"].to(device).float()           # (B, L)
        m   = batch["mask"].to(device).float()        # (B, L)
        lengths = batch["lengths"].to(device)         # (B,) true lengths

        pred = model(emb, lengths)                    # (B, Lb)
        yb = y[:, :pred.size(1)]
        mb = m[:, :pred.size(1)]

        loss = masked_mse(pred, yb, mb)

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

        pred_raw = model(emb, lengths)
        yb = y[:, :pred_raw.size(1)]
        mb = m[:, :pred_raw.size(1)]

        pred = pred_raw.clamp(0.0, 1.0) if clamp_for_metrics else pred_raw

        loss = masked_mse(pred, yb, mb)
        r = masked_pearson(pred, yb, mb)

        total_loss += loss.item()
        if torch.isfinite(r):
            total_r += r.item()
            n_r += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_r = total_r / max(1, n_r) if n_r > 0 else float("nan")
    return avg_loss, avg_r

