"""ABMIL — Attention-Based Multiple Instance Learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import Tensor


class _Attention(nn.Module):
    """Standard attention: w^T tanh(V h)."""

    def __init__(self, M: int, L: int, K: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(M, L),
            nn.Tanh(),
            nn.Linear(L, K),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)  # (B, N, K)


class _GatedAttention(nn.Module):
    """Gated attention: w^T (tanh(V h) ⊙ sigmoid(U h))."""

    def __init__(self, M: int, L: int, K: int = 1):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.w = nn.Linear(L, K)

    def forward(self, h: Tensor) -> Tensor:
        return self.w(self.V(h) * self.U(h))  # (B, N, K)


class ABMIL(L.LightningModule):
    """Attention-Based MIL model.

    encode()  → slide embedding + attention weights  (used in aggregation)
    forward() → logits + attention                   (used in training / inference)

    Args:
        input_dim:   patch feature dimension (M in the paper)
        hidden_dim:  attention hidden dimension (L in the paper)
        gate:        use gated attention
        num_classes: number of output classes
        lr:          learning rate
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 384,
        gate: bool = True,
        num_classes: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        attn_cls = _GatedAttention if gate else _Attention
        self.attention = attn_cls(M=input_dim, L=hidden_dim, K=1)

        self.classifier = nn.Linear(in_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Compute slide embedding and attention weights from patch embeddings.

        Args:
            x:    (N, D) / (1, N, D) or (B, N, D)
            mask: (N,) / (1, N)  or  (B, N)  — 1=real patch, 0=padding

        Returns:
            slide_embedding: (D,) for single WSI  /  (B, D) for batch
            attention:       (N,) for single WSI  /  (B, N) for batch  softmax-normalized
        """
        squeeze = x.dim() < 3
        if squeeze:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        a = self.attention(x).transpose(-2, -1)  # (B, 1, N)

        if mask is not None:
            a = a + (1.0 - mask.unsqueeze(1)) * torch.finfo(a.dtype).min

        a = F.softmax(a, dim=-1)                 # (B, 1, N)
        slide_emb = torch.bmm(a, x).squeeze(1)   # (B, D)
        a = a.squeeze(1)                          # (B, N)

        if squeeze:
            return slide_emb.squeeze(0), a.squeeze(0)
        return slide_emb, a

    def forward(self, x: Tensor, mask: Tensor | None = None) -> dict:
        """Compute logits and attention from patch embeddings.

        Args:
            x:    (N, D) / (1, N, D) or (B, N, D)
            mask: (N,) / (1, N)  or  (B, N) — optional

        Returns:
            {"logits": (num_classes,) or (B, num_classes), "attention": (N,) or (B, N)}
        """
        slide_emb, attention = self.encode(x, mask)
        logits = self.classifier(slide_emb)
        return {"logits": logits, "attention": attention}

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        out = self(x, mask)
        loss = self.loss_fn(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        out = self(x, mask)
        loss = self.loss_fn(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
