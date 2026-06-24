"""PreviewCommand — visualize per-patch attention scores as spatial heatmaps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PreviewConfig:
    scores: str = ""
    patch_size: int = 512
    output_dir: str = "./previews"
    colormap: str = "jet"
    size: int = 64
    font_size: int = 12
    jpeg_quality: int = 90


class PreviewCommand:
    """Visualize per-patch attention scores using wsi-toolbox BasePreviewCommand.

    Renders actual patch thumbnails with a score-colored border frame.
    Requires ``cache/{patch_size}/patches`` and ``cache/{patch_size}/coordinates``
    to exist in each HDF5 file (created by wsi-toolbox CacheCommand).

    Scores file (npz) format:
        {slide_id}        → (N,) float scores  (required)
        {slide_id}_ids    → (N,) int   patch indices into cache coords  (optional)
    """

    def __init__(self, config: PreviewConfig | None = None):
        self.config = config or PreviewConfig()

    def __call__(self, h5_paths: list[Path]) -> None:
        from wsi_toolbox.commands.preview import BasePreviewCommand
        from matplotlib import pyplot as plt
        from matplotlib import colors as mcolors
        from PIL import ImageFont
        from wsi_toolbox.utils import get_platform_font, create_frame

        cfg = self.config
        scores_data = np.load(cfg.scores, allow_pickle=False)
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            font = ImageFont.truetype(font=get_platform_font(), size=cfg.font_size)
        except OSError:
            font = ImageFont.load_default()

        size = cfg.size
        colormap = cfg.colormap

        class _AttentionRenderer(BasePreviewCommand):
            def _prepare(self_r, f, *, scores, patch_ids, cmap_name):
                s_min, s_max = scores.min(), scores.max()
                norm = (scores - s_min) / (s_max - s_min + 1e-8)
                if patch_ids is not None:
                    idx_to_score = {int(p): float(s) for p, s in zip(patch_ids, norm)}
                else:
                    idx_to_score = {i: float(s) for i, s in enumerate(norm)}
                return {
                    "idx_to_score": idx_to_score,
                    "cmap": plt.get_cmap(cmap_name),
                    "font": font,
                    "frame_cache": {},
                }

            def _get_frame(self_r, index, data, f):
                score = data["idx_to_score"].get(index)
                if score is None:
                    return None
                key = round(score, 3)
                cache = data["frame_cache"]
                if key not in cache:
                    color = mcolors.rgb2hex(data["cmap"](score)[:3])
                    cache[key] = create_frame(size, color, f"{key:.3f}", data["font"])
                return cache[key]

        renderer = _AttentionRenderer(
            size=cfg.size,
            font_size=cfg.font_size,
            patch_size=cfg.patch_size,
        )

        total = len(h5_paths)
        for idx, h5_path in enumerate(h5_paths):
            h5_path = Path(h5_path)
            slide_id = h5_path.stem

            if slide_id not in scores_data:
                print(f"[{idx + 1}/{total}] SKIP {slide_id} (not in scores file)")
                continue

            scores = scores_data[slide_id].astype(np.float32)
            ids_key = f"{slide_id}_ids"
            patch_ids = scores_data[ids_key].astype(int) if ids_key in scores_data else None

            out_path = output_dir / f"{slide_id}.jpg"
            image = renderer(
                str(h5_path),
                scores=scores,
                patch_ids=patch_ids,
                cmap_name=cfg.colormap,
            )
            image.save(str(out_path), "JPEG", quality=cfg.jpeg_quality)
            print(f"[{idx + 1}/{total}] {slide_id} → {out_path}")
