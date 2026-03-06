import pandas as pd
import numpy as np
import torch as torch
import transformers
import esm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path



class RSADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        pdb_id = str(row["pdb_id"])
        chain_id = str(row["chain_id"])
        uid = f"{pdb_id}_{chain_id}"

        seq_str = "".join(row["sequence"])

        rsa  = np.asarray(row["rsa"], dtype=np.float32)
        mask = np.asarray(row["mask_label"], dtype=np.float32)

        rsa = np.nan_to_num(rsa, nan=0.0, posinf=0.0, neginf=0.0)

        finite = np.isfinite(rsa)
        mask = mask * finite.astype(np.float32)

        y = torch.from_numpy(rsa)
        mask = torch.from_numpy(mask)

        return {
            "uid": uid,
            "seq": seq_str,
            "y": y,
            "mask": mask,
        }

def collate_rsa(batch):
    # batch is a list of dicts from __getitem__
    uids = [b["uid"] for b in batch]
    seqs = [b["seq"] for b in batch]

    ys = [b["y"] for b in batch]         # list of (L_i,)
    masks = [b["mask"] for b in batch]   # list of (L_i,)

    lengths = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)  # (B,)
    Lmax = int(lengths.max().item())

    B = len(batch)
    y_pad = torch.zeros((B, Lmax), dtype=torch.float32)
    mask_pad = torch.zeros((B, Lmax), dtype=torch.float32)

    for i, (y, m) in enumerate(zip(ys, masks)):
        L = y.shape[0]
        y_pad[i, :L] = y
        mask_pad[i, :L] = m  # padded positions remain 0

    return {
        "uid": uids,
        "seq": seqs,                 # still raw strings (later tokenizer uses these)
        "y": y_pad,                  # (B, Lmax)
        "mask": mask_pad,            # (B, Lmax)
        "lengths": lengths,          # (B,)
    }


def masked_mse(pred, target, mask, eps=1e-8):
    # pred/target/mask: (B, L)
    se = (pred - target) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(eps)
    return se.sum() / denom



@torch.no_grad()
def cache_esm_embeddings(
    loader,
    model,
    batch_converter,
    device,
    cache_dir,
    layer=30,
    dtype=torch.float16,     # saves lots of space
    overwrite=False,
):
    """
    Saves one file per uid:
      {cache_dir}/{uid}.pt  containing:
        - emb: (L, D) residue embeddings (NO special tokens, NO pad)
        - y:   (L,) labels (same L)
        - mask:(L,) mask   (same L)
        - uid, seq (optional), layer, D
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model.eval().to(device)

    for batch in tqdm(loader, desc=f"Caching ESM layer {layer}"):
        uids = batch["uid"]
        seqs = batch["seq"]

        # (B, Lmax) padded labels/masks from your collate
        y_pad = batch["y"]
        m_pad = batch["mask"]
        lengths = batch["lengths"]  # (B,)

        # ESM tokenization (handles padding + BOS/EOS)
        data = list(zip(uids, seqs))
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        out = model(tokens, repr_layers=[layer], return_contacts=False)
        rep = out["representations"][layer]   # (B, Ltok, D)
        rep = rep[:, 1:-1, :]                 # (B, Lmax, D) remove BOS/EOS

        # move once to CPU and optionally cast
        rep = rep.to("cpu")
        if dtype is not None:
            rep = rep.to(dtype)

        for i, uid in enumerate(uids):
            L = int(lengths[i].item())
            out_path = cache_dir / f"{uid}.pt"
            if out_path.exists() and not overwrite:
                continue

            # store only true length (no padding)
            item = {
                "uid": uid,
                "emb": rep[i, :L].contiguous(),           # (L, D)
                "y":   y_pad[i, :L].to("cpu").contiguous(),
                "mask":m_pad[i, :L].to("cpu").contiguous(),
                "layer": layer,
            }

            tmp = str(out_path) + ".tmp"
            torch.save(item, tmp)
            Path(tmp).replace(out_path)  # atomic-ish rename



