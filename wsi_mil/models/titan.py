"""TITAN slide encoder — thin wrapper around a HuggingFace AutoModel."""
from __future__ import annotations

import contextlib
import torch
import torch.nn.functional as F
from torch import Tensor


class TITAN:
    """TITAN slide encoder wrapper.

    Wraps a HuggingFace AutoModel and exposes a consistent encode() interface.

    Args:
        model: return value of AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)

    Usage:
        titan = TITAN(model)
        slide_emb = titan.encode(patch_embeddings, coords, patch_size_lv0=512)
    """

    def __init__(self, model):
        self.model = model

    def encode(
        self,
        patch_embeddings: Tensor,  # (N, D)
        coords: Tensor,            # (N, 2)  level-0 pixel coordinates
        patch_size_lv0: int,
        normalize: bool = False,
    ) -> Tensor:
        """Encode patch embeddings and coordinates into a slide embedding (D,)."""
        autocast = (
            torch.autocast("cuda", torch.float16)
            if patch_embeddings.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            slide_emb = self.model.encode_slide_from_patch_features(
                patch_embeddings,
                coords,
                patch_size_lv0,
            )
        if slide_emb.dim() > 1:
            slide_emb = slide_emb.squeeze(0)
        if normalize:
            slide_emb = F.normalize(slide_emb, dim=0)
        return slide_emb
