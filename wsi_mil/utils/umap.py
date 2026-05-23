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
    data: dict,
    output_path: str | Path,
    class_names: dict | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.01,
    random_state: int = 42,
    figsize: tuple = (16, 14),
    annotate: bool = True,
    title: str | None = None,
    show_misclassified: bool = True,
    subtypes: np.ndarray | None = None,
    subtype_names: dict | None = None,
    point_size: int = 100,
    ring_size: int = 160,
    ring_linewidth: float = 2.0,
) -> None:
    """Generate and save a UMAP plot of slide embeddings.

    When subtypes is provided:
      - Layer 1: dots colored by label
      - Layer 2: hollow rings colored by subtype

    Args:
        data:               dict with keys: embeddings, labels, predictions (optional), case_names (optional)
        output_path:        save path
        class_names:        label index → display name
        n_neighbors:        UMAP n_neighbors
        min_dist:           UMAP min_dist
        random_state:       random seed
        figsize:            figure size
        annotate:           annotate each point with case name
        title:              plot title (default if None)
        show_misclassified: mark misclassified samples with an X
        subtypes:           subtype array (str or int); if None only label coloring is used
        subtype_names:      subtype value → display name
        point_size:         filled dot size
        ring_size:          subtype ring size
        ring_linewidth:     subtype ring line width
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required. Install with: uv add umap-learn")

    embeddings = data["embeddings"]
    labels = data["labels"]
    predictions = data.get("predictions")
    case_names = data.get("case_names")

    if subtypes is None:
        subtypes = data.get("subtypes")
    if subtypes is not None:
        subtypes = np.asarray(subtypes)

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

    # Layer 2: hollow rings colored by subtype
    if subtypes is not None:
        unique_subtypes = sorted(np.unique(subtypes), key=str)
        if len(unique_subtypes) >= 2:
            st_color_map = _make_color_map(unique_subtypes, cmap="tab20")
            s_names = subtype_names or {}

            for st in unique_subtypes:
                mask = subtypes == st
                plt.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    facecolors="none",
                    edgecolors=st_color_map[st],
                    linewidths=ring_linewidth,
                    s=ring_size,
                    label=s_names.get(st, str(st)),
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

    default_title = (
        "UMAP — label fill / subtype ring" if subtypes is not None
        else "UMAP Visualization of Slide Embeddings"
    )
    plt.legend(loc="best", fontsize=9)
    plt.title(title or default_title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved UMAP plot: {output_path}")


def _make_color_map(keys, cmap: str = "tab10") -> dict:
    keys = list(keys)
    cm = mcm.get_cmap(cmap)
    return {k: cm(i % cm.N) for i, k in enumerate(keys)}
