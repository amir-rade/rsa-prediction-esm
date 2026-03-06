import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def masked_mse(pred, target, mask, eps=1e-8):
    # pred/target/mask: (B, L)
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom

@torch.no_grad()
def masked_pearson(pred, target, mask, eps=1e-8):
    # pred/target/mask: (B, L)
    # returns scalar pearson over all valid positions in batch
    pred = pred[mask.bool()]
    target = target[mask.bool()]
    if pred.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)

    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.std(unbiased=False) * target.std(unbiased=False)).clamp_min(eps)
    return (pred * target).mean() / denom

class BiLSTMHead(nn.Module):
    """
    Per-residue RSA regressor with BiLSTM.
    Input:  emb (B, L, D) padded
            lengths (B,) true sequence lengths (not mask.sum!)
    Output: pred (B, L) raw
    """
    def __init__(
        self,
        d_in=640,
        hidden=256,
        n_layers=2,
        dropout=0.2,
        mlp_hidden=128,
        use_layernorm=True,
    ):
        super().__init__()

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln_in = nn.LayerNorm(d_in)


        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb, lengths):
        """
        emb: (B, L, D)
        lengths: (B,) true lengths (<= L)
        """
        if self.use_layernorm:
            emb = self.ln_in(emb)

        # pack (requires CPU lengths)
        lengths_cpu = lengths.to("cpu")
        packed = pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, 2*hidden) where L = max(lengths)

        pred = self.head(out).squeeze(-1)  # (B, L)
        return pred

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        emb = batch["emb"].to(device).float()         # (B, L, D)
        y   = batch["y"].to(device).float()           # (B, L)
        m   = batch["mask"].to(device).float()        # (B, L)

        # IMPORTANT: lengths should be true sequence lengths
        lengths = batch["lengths"].to(device)         # (B,)

        pred = model(emb, lengths)                    # raw (B, L)

        # pred returned length dimension equals max(lengths) in this batch,
        # which should match y/m second dim if collate padded to that Lmax.
        loss = masked_mse(pred, y[:, :pred.size(1)], m[:, :pred.size(1)])

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
    n_batches = 0

    for batch in loader:
        emb = batch["emb"].to(device).float()
        y   = batch["y"].to(device).float()
        m   = batch["mask"].to(device).float()
        lengths = batch["lengths"].to(device)

        pred_raw = model(emb, lengths)  # (B, Lb)
        yb = y[:, :pred_raw.size(1)]
        mb = m[:, :pred_raw.size(1)]

        pred = pred_raw.clamp(0.0, 1.0) if clamp_for_metrics else pred_raw

        loss = masked_mse(pred, yb, mb)
        r = masked_pearson(pred, yb, mb)

        total_loss += loss.item()

        if torch.isfinite(r):
            total_r += r.item()
            n_batches += 1

    avg_loss = total_loss / max(1, len(loader))
    avg_r = total_r / max(1, n_batches) if n_batches > 0 else float("nan")
    return avg_loss, avg_r