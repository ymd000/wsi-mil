"""UMAP visualization utility for slide embeddings."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    coords_2d: np.ndarray | None = None,
    class_names: dict | None = None,
    predictions: np.ndarray | None = None,
    case_names: list[str] | None = None,
    overlay: np.ndarray | None = None,
    overlay_name: str | None = None,
    overlay_labels: dict | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.01,
    random_state: int = 42,
    figsize: tuple = (16, 14),
    annotate: bool = True,
    title: str | None = None,
    show_misclassified: bool = True,
    point_size: int = 100,
    ring_size: int = 160,
    ring_linewidth: float = 2.0,
) -> np.ndarray:
    """Generate and save a UMAP plot of slide embeddings.

    Args:
        embeddings:         (N, D) slide embeddings
        labels:             (N,) integer class labels
        output_path:        save path
        coords_2d:          pre-computed (N, 2) UMAP coordinates; if provided, UMAP fit is skipped
        class_names:        label index → display name
        predictions:        (N,) predicted labels; enables misclassified markers
        case_names:         per-sample names for annotation
        overlay:            (N,) metadata array to render as hollow rings
        overlay_name:       column name shown in legend
        overlay_labels:     overlay value → display name
        n_neighbors:        UMAP n_neighbors
        min_dist:           UMAP min_dist
        random_state:       random seed
        figsize:            figure size
        annotate:           annotate each point with case name
        title:              plot title
        show_misclassified: mark misclassified samples with an X
        point_size:         filled dot size
        ring_size:          overlay ring size
        ring_linewidth:     overlay ring line width

    Returns:
        coords_2d: (N, 2) UMAP coordinates (reusable for subsequent overlay plots)
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required. Install with: uv add umap-learn")

    if coords_2d is None:
        umap_model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value .* overridden to 1 by setting random_state",
                category=UserWarning,
            )
            coords_2d = umap_model.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    # Layer 1: dots colored by label
    unique_labels = np.unique(labels)
    label_color_map = _make_color_map(unique_labels, cmap="tab10")
    names = class_names or {}

    for i in unique_labels:
        mask = labels == i
        plt.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            label=names.get(i, str(i)),
            color=label_color_map[i],
            alpha=0.7,
            s=point_size,
            zorder=2,
        )

    # Layer 2: hollow rings colored by overlay
    if overlay is not None:
        overlay = np.asarray(overlay, dtype=object)
        unique_vals = sorted({v for v in overlay if v is not None}, key=str)
        if len(unique_vals) >= 2:
            ov_color_map = _make_color_map(unique_vals, cmap="tab20")
            ov_names = overlay_labels or {}
            ov_title = overlay_name or "overlay"
            for val in unique_vals:
                mask = overlay == val
                plt.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    facecolors="none",
                    edgecolors=ov_color_map[val],
                    linewidths=ring_linewidth,
                    s=ring_size,
                    label=f"{ov_title}: {ov_names.get(val, str(val))}",
                    zorder=3,
                )

    # Misclassified marks
    if show_misclassified and predictions is not None:
        misclassified = predictions != labels
        if np.any(misclassified):
            plt.scatter(
                coords_2d[misclassified, 0],
                coords_2d[misclassified, 1],
                marker="x",
                c="red",
                s=50,
                linewidths=1.5,
                label="Misclassified",
                zorder=10,
            )

    # Annotations
    if annotate and case_names is not None:
        for j, (x, y) in enumerate(coords_2d):
            plt.annotate(
                case_names[j],
                (x, y),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    if title is None:
        title = (
            f"UMAP — label / {overlay_name}" if overlay is not None
            else "UMAP Visualization of Slide Embeddings"
        )
    plt.legend(loc="best", fontsize=9)
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved UMAP plot: {output_path}")

    return coords_2d


def _make_color_map(keys, cmap: str = "tab10") -> dict:
    keys = list(keys)
    cm = mcm.get_cmap(cmap)
    return {k: cm(i % cm.N) for i, k in enumerate(keys)}
