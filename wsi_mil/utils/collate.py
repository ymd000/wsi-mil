"""Collate functions for DataLoader."""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def mil_collate_fn(batch: list[tuple[torch.Tensor, int]]):
    """Pad variable-length WSI features into a batch.

    Args:
        batch: [(features: Tensor(N_i, D), label: int), ...]

    Returns:
        features: (B, max_N, D)
        mask:     (B, max_N)  1 = real patch / 0 = padding
        labels:   (B,)
    """
    features_list, labels = zip(*batch)
    padded = pad_sequence(features_list, batch_first=True)
    max_n = padded.size(1)
    mask = torch.zeros(len(features_list), max_n, dtype=torch.float32)
    for i, feat in enumerate(features_list):
        mask[i, :feat.size(0)] = 1.0
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels
