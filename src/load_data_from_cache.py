import pandas as pd
import numpy as np
import torch as torch
import transformers
import esm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import torch


class CachedEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, in_memory=True):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found in: {cache_dir}")

        self.in_memory = in_memory
        self._ram = {} if in_memory else None   # idx -> loaded dict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.in_memory:
            item = self._ram.get(idx)
            if item is None:
                item = torch.load(self.files[idx], map_location="cpu")
                self._ram[idx] = item
            return item
        else:
            return torch.load(self.files[idx], map_location="cpu")




def collate_cached(batch):
    uids = [b["uid"] for b in batch]
    embs = [b["emb"] for b in batch]     # (L_i, D)
    ys   = [b["y"] for b in batch]       # (L_i,)
    ms   = [b["mask"] for b in batch]    # (L_i,)

    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    Lmax = int(lengths.max().item())
    B = len(batch)
    D = embs[0].shape[1]

    emb_pad  = torch.zeros((B, Lmax, D), dtype=embs[0].dtype)
    y_pad    = torch.zeros((B, Lmax), dtype=torch.float32)
    mask_pad = torch.zeros((B, Lmax), dtype=torch.float32)

    for i, (e, y, m) in enumerate(zip(embs, ys, ms)):
        L = e.shape[0]
        emb_pad[i, :L] = e
        y_pad[i, :L] = y.float()
        mask_pad[i, :L] = m.float()

    return {"uid": uids, "emb": emb_pad, "y": y_pad, "mask": mask_pad, "lengths": lengths}

