"""Dataset classes."""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset


class WSIDataset(Dataset):
    """HDF5 patch embedding dataset for whole slide images.

    Args:
        data_dir:     directory containing HDF5 files
        encoder_name: encoder name matching the HDF5 group key (e.g. "conch15_768")
        csv_path:     CSV with case_id and label columns; if omitted all labels are -1
        use_cache:    cache features in memory
    """

    def __init__(
        self,
        data_dir: str,
        encoder_name: str,
        csv_path: str | None = None,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.encoder_name = encoder_name
        self.use_cache = use_cache
        self._cache: dict[int, torch.Tensor] = {}
        self._memory_warned = False

        label_dict: dict[str, int] = {}
        if csv_path is not None:
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    label_dict[row["case_id"]] = int(row["label"])

        self.h5_files: list[Path] = []
        self.labels: list[int] = []

        for file in sorted(self.data_dir.iterdir()):
            if file.suffix == ".h5":
                self.h5_files.append(file)
                self.labels.append(label_dict.get(file.stem, -1))

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        """Returns (embeddings: Tensor(N, D), label: int)."""
        return self._load_embeddings(idx), self.labels[idx]

    def _load_embeddings(self, idx: int) -> torch.Tensor:
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        with h5py.File(self.h5_files[idx], "r") as f:
            embeddings = torch.from_numpy(
                np.asarray(f[f"{self.encoder_name}/features"])
            ).float()

        if self.use_cache:
            self._cache[idx] = embeddings
            if not self._memory_warned:
                mem = psutil.virtual_memory()
                if mem.percent >= 90.0:
                    warnings.warn(
                        f"RAM usage is {mem.percent:.1f}% "
                        f"({mem.available / 1e9:.1f} GB available). "
                        "Consider disabling cache (use_cache=False).",
                        ResourceWarning,
                        stacklevel=2,
                    )
                    self._memory_warned = True

        return embeddings
