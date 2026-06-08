"""TrainCommand — cross-validation training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from wsi_mil.utils import FoldManager, mil_collate_fn as _default_collate


@dataclass
class TrainConfig:
    """Training configuration."""
    output_dir: str = "./outputs"
    num_fold: int = 5
    max_epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-3
    devices: int = 1
    num_workers: int = 0
    shuffle: bool = True
    random_state: int = 42
    existing_fold_dir: str | None = None  # reuse fold splits from an existing version dir
    model_kwargs: dict = field(default_factory=dict)


class TrainCommand:
    """Cross-validation training command.

    Python:
        cmd = TrainCommand(model_class=ABMIL, config=TrainConfig(...))
        cmd(dataset)

    CLI:
        mil train --config train.yaml
    """

    def __init__(
        self,
        model_class,
        config: TrainConfig | None = None,
        collate_fn=_default_collate,
    ):
        self.model_class = model_class
        self.config = config or TrainConfig()
        self.collate_fn = collate_fn

    def __call__(self, dataset) -> list[dict]:
        """Run training and return per-fold results.

        Returns:
            [{"fold_idx": 0, "best_checkpoint": "...", "best_val_loss": 0.xx}, ...]
        """
        torch.set_float32_matmul_precision("high")
        cfg = self.config
        base_dir = Path(cfg.output_dir)
        version_dir = self._get_version_dir(base_dir)
        version_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {version_dir}")

        self._save_config(version_dir)

        fold_manager = FoldManager(version_dir)
        if cfg.existing_fold_dir is not None:
            src = FoldManager(Path(cfg.existing_fold_dir))
            src.load()
            fold_manager.folds = src.folds
            fold_manager.num_folds = src.num_folds
            fold_manager.save()
            print(f"Reusing fold splits from: {cfg.existing_fold_dir}")
        else:
            fold_manager.create_folds(dataset, cfg.num_fold, cfg.shuffle, cfg.random_state)
            fold_manager.save()

        results = []
        for fold_info in fold_manager.folds:
            print(f"\n=== Fold {fold_info.fold_idx + 1}/{fold_manager.num_folds} ===")
            results.append(self._run_one_fold(fold_info, dataset, fold_manager))
        return results

    def run_retrain_all(
        self,
        dataset,
        version_dir: str | Path | None = None,
    ) -> dict:
        """Train a single model on the full dataset.

        Saves to <version_dir>/retrain_all/. Intended to be run after CV training
        using the same config. No validation split — trains for cfg.max_epochs fixed.

        Args:
            dataset:     full dataset (all samples used for training)
            version_dir: existing version directory from a prior CV run;
                         if None, uses the latest version in cfg.output_dir

        Returns:
            {"checkpoint": "...", "version_dir": "..."}
        """
        torch.set_float32_matmul_precision("high")
        cfg = self.config

        if version_dir is None:
            base_dir = Path(cfg.output_dir)
            existing = sorted(
                [d for d in base_dir.glob("version_*") if d.is_dir()],
                key=lambda d: int(d.name.split("_")[1]),
            )
            if existing:
                version_dir = existing[-1]
                print(f"Using latest version: {version_dir}")
            else:
                version_dir = self._get_version_dir(base_dir)
                version_dir.mkdir(parents=True, exist_ok=True)
                self._save_config(version_dir)
                print(f"Created new version directory: {version_dir}")

        version_dir = Path(version_dir)
        retrain_dir = version_dir / "retrain_all"
        retrain_dir.mkdir(parents=True, exist_ok=True)
        print(f"Retrain-all output: {retrain_dir}")

        model = self.model_class(lr=cfg.lr * cfg.devices, **cfg.model_kwargs)

        train_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        print(f"Training on all {len(dataset)} samples for {cfg.max_epochs} epochs")

        checkpoint_cb = ModelCheckpoint(
            dirpath=retrain_dir / "checkpoints",
            filename="last",
            save_last=True,
            save_top_k=0,
            enable_version_counter=False,
        )
        logger = CSVLogger(save_dir=str(retrain_dir), name="logs", version="")

        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="auto",
            devices=cfg.devices,
            strategy="ddp" if cfg.devices > 1 else "auto",
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint_cb],
        )
        trainer.fit(model=model, train_dataloaders=train_loader)

        ckpt_path = str(checkpoint_cb.last_model_path)
        print(f"Saved checkpoint: {ckpt_path}")
        return {"checkpoint": ckpt_path, "version_dir": str(version_dir)}

    def _get_version_dir(self, base_dir: Path) -> Path:
        existing = [d for d in base_dir.glob("version_*") if d.is_dir()]
        next_version = max((int(d.name.split("_")[1]) for d in existing), default=-1) + 1
        return base_dir / f"version_{next_version}"

    def _save_config(self, version_dir: Path) -> None:
        import yaml
        from datetime import datetime
        cfg = self.config
        data = {
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_class":       self.model_class.__name__,
            "num_fold":          cfg.num_fold,
            "max_epochs":        cfg.max_epochs,
            "batch_size":        cfg.batch_size,
            "lr":                cfg.lr,
            "devices":           cfg.devices,
            "num_workers":       cfg.num_workers,
            "shuffle":           cfg.shuffle,
            "random_state":      cfg.random_state,
            "model_kwargs":      cfg.model_kwargs,
            "existing_fold_dir": str(cfg.existing_fold_dir) if cfg.existing_fold_dir else None,
        }
        with open(version_dir / "config.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _run_one_fold(self, fold_info, dataset, fold_manager: FoldManager) -> dict:
        cfg = self.config
        fold_idx = fold_info.fold_idx

        model = self.model_class(lr=cfg.lr * cfg.devices, **cfg.model_kwargs)

        train_loader = DataLoader(
            Subset(dataset, fold_info.train_indices),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            Subset(dataset, fold_info.val_indices),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=self.collate_fn,
        )
        print(f"Train WSIs: {len(fold_info.train_indices)}, Val WSIs: {len(fold_info.val_indices)}")

        fold_dir = fold_manager.get_fold_dir(fold_idx)

        checkpoint_cb = ModelCheckpoint(
            dirpath=fold_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        )
        logger = CSVLogger(save_dir=str(fold_dir), name="logs", version="")

        trainer = L.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="auto",
            devices=cfg.devices,
            strategy="ddp" if cfg.devices > 1 else "auto",
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint_cb],
        )
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        return {
            "fold_idx":        fold_idx,
            "best_checkpoint": checkpoint_cb.best_model_path,
            "best_val_loss":   checkpoint_cb.best_model_score,
        }
