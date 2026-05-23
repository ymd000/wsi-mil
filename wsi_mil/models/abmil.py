"""ABMIL — Attention-Based Multiple Instance Learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import lightning as L
from torch import Tensor


class ABMIL(L.LightningModule):
    """Attention-Based MIL model.

    encode()  → slide embedding + attention weights  (used in aggregation)
    forward() → logits + attention                   (used in training / inference)

    Args:
        input_dim:   patch embedding dimension
        hidden_dim:  attention network hidden dimension
        num_classes: number of output classes
        lr:          learning rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute slide embedding and attention weights from patch embeddings.

        Args:
            x: (N, D) or (1, N, D)

        Returns:
            slide_embedding: (D,)
            attention:       (N,)  softmax-normalized
        """
        if x.dim() == 3:
            x = x.squeeze(0)
        a = self.attention(x)
        a = torch.softmax(a, dim=0)
        slide_emb = (a * x).sum(0)
        return slide_emb, a.squeeze(1)

    def forward(self, x: Tensor) -> dict:
        """Compute logits and attention from patch embeddings.

        Args:
            x: (N, D) or (1, N, D)

        Returns:
            {"logits": (num_classes,), "attention": (N,)}
        """
        slide_emb, attention = self.encode(x)
        logits = self.classifier(slide_emb)
        return {"logits": logits, "attention": attention}

    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        out = self(x)
        logits = out["logits"].unsqueeze(0)  # (C,) → (1, C) to match y shape (1,)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        out = self(x)
        logits = out["logits"].unsqueeze(0)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
