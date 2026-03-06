import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class CNNHead(nn.Module):

    def __init__(
        self,
        d_in=640,
        hidden=256,
        mlp_hidden=128,
        n_layers=4,
        kernel_size=5,
        dropout=0.2,
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        self.input_proj = nn.Conv1d(d_in, hidden, kernel_size=1)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, emb):

        # emb: (B, L, D)

        x = emb.transpose(1, 2)       # (B, D, L)

        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual

        x = x.transpose(1, 2)         # (B, L, hidden)

        y = self.mlp(x).squeeze(-1)   # (B, L)

        return y

def masked_mse(pred, target, mask, eps=1e-8):
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom

def masked_pearson(pred, target, mask, eps=1e-8):
    pred = pred[mask.bool()]
    target = target[mask.bool()]
    if pred.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)

    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.std(unbiased=False) * target.std(unbiased=False)).clamp_min(eps)
    return (pred * target).mean() / denom

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0

    for batch in loader:
        emb  = batch["emb"].to(device).float()   # (B, L, D)
        y    = batch["y"].to(device).float()     # (B, L)
        mask = batch["mask"].to(device).float()  # (B, L)

        pred = model(emb)                        # raw (B, L)
        loss = masked_mse(pred, y, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / max(1, len(loader))

@torch.no_grad()
def eval_one_epoch(model, loader, device, clamp_for_metrics=True):
    model.eval()
    total_loss = 0.0
    total_r = 0.0
    n_batches = 0

    for batch in loader:
        emb  = batch["emb"].to(device).float()
        y    = batch["y"].to(device).float()
        mask = batch["mask"].to(device).float()

        pred_raw = model(emb)

        pred_for_loss = pred_raw
        pred_for_metrics = pred_raw.clamp(0.0, 1.0) if clamp_for_metrics else pred_raw

        loss = masked_mse(pred_for_loss, y, mask)
        r = masked_pearson(pred_for_metrics, y, mask)

        total_loss += loss.item()
        total_r += float(r.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")

    return total_loss / n_batches, total_r / n_batches