"""LinearProbeModel — slide embedding → class label."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearProbeModel(L.LightningModule):
    """Linear classifier on top of slide embeddings.

    Can be trained with the same fold splits as ABMIL via TrainConfig.existing_fold_dir.

    Args:
        embedding_dim: input embedding dimension
        num_classes:   number of output classes
        lr:            learning rate
        dropout:       dropout rate (0.0 = disabled)
        weight_decay:  Adam weight decay
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embedding_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> dict:
        """Args:
            x: (B, D)
        Returns:
            {"logits": (B, C), "attention": None}
        """
        return {"logits": self.classifier(x), "attention": None}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
