"""EvaluateCommand — metrics, UMAP, and confusion matrix."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EvaluateConfig:
    """Evaluation configuration."""
    output_dir: str = "./eval_outputs"
    positive_class: int = 1
    average: str = "macro"
    class_names: dict[int, str] | None = None
    plot_umap: bool = True
    plot_confusion_matrix: bool = True
    umap_figsize: list = field(default_factory=lambda: [12, 10])
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.01
    umap_annotate: bool = True
    confusion_matrix_filename: str = "confusion_matrix.png"
    umap_filename: str = "umap.png"
    umap_random_state: int = 42
    umap_point_size: int = 100
    umap_ring_size: int = 160
    umap_ring_linewidth: float = 2.0
    show_misclassified: bool = True
    overlay_csv: str | None = None
    overlay_cols: list[str] = field(default_factory=list)


class EvaluateCommand:
    """Evaluation and visualization command.

    Python:
        cmd = EvaluateCommand(config=EvaluateConfig(class_names={0: "A", 1: "B"}))
        metrics = cmd(results)   # results is the return value of load_embeddings()

    CLI:
        mil evaluate --config evaluate.yaml
    """

    @staticmethod
    def load_embeddings(
        data_dir: str | Path,
        method_name: str,
        csv_path: str | Path,
        encoder_name: str,
    ) -> dict:
        """Load slide embeddings for the entire dataset from HDF5.

        Returns:
            {
                "embeddings", "labels", "predictions", "probabilities",
                "attentions", "selected_indices", "indices",
                "h5_paths", "case_names"
            }
        """
        import csv as _csv
        from pathlib import Path as _Path

        import h5py

        data_dir = _Path(data_dir)
        label_dict: dict[str, int] = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                label_dict[row["case_id"]] = int(row["label"])

        embeddings, labels, predictions, probabilities = [], [], [], []
        attentions, selected_indices, indices, h5_paths, case_names = [], [], [], [], []
        has_pred = has_attn = has_selected = False

        group_path = f"{encoder_name}/slide_embedding/{method_name}"
        for idx, h5_path in enumerate(sorted(data_dir.glob("*.h5"))):
            case_id = h5_path.stem
            if case_id not in label_dict:
                continue
            with h5py.File(h5_path, "r") as f:
                if group_path not in f:
                    raise KeyError(f"{h5_path}: '{group_path}' not found.")
                grp = f[group_path]
                emb         = grp["embedding"][:]
                attention   = grp["attention"][:]     if "attention"      in grp      else None
                probs       = grp["probabilities"][:] if "probabilities"  in grp      else None
                prediction  = int(grp.attrs["prediction"])     if "prediction"     in grp.attrs else None
                selected    = int(grp.attrs["selected_index"]) if "selected_index" in grp.attrs else None

            embeddings.append(emb)
            labels.append(label_dict[case_id])
            predictions.append(prediction)
            probabilities.append(probs)
            attentions.append(attention)
            selected_indices.append(selected)
            indices.append(idx)
            h5_paths.append(h5_path)
            case_names.append(case_id)
            if prediction is not None: has_pred     = True
            if attention  is not None: has_attn     = True
            if selected   is not None: has_selected = True

        return {
            "embeddings":       np.stack(embeddings),
            "labels":           np.array(labels),
            "predictions":      np.array(predictions) if has_pred else None,
            "probabilities":    probabilities if any(p is not None for p in probabilities) else None,
            "attentions":       attentions if has_attn else None,
            "selected_indices": selected_indices if has_selected else None,
            "indices":          indices,
            "h5_paths":         h5_paths,
            "case_names":       case_names,
        }

    def __init__(self, config: EvaluateConfig | None = None):
        self.config = config or EvaluateConfig()

    def __call__(self, results: dict) -> dict:
        """Compute metrics and save visualizations.

        Args:
            results: return value of load_embeddings()

        Returns:
            metrics dict
        """
        metrics = self.compute_metrics(results)
        self.print_metrics(metrics)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.plot_confusion_matrix and results.get("predictions") is not None:
            self.save_confusion_matrix(results, output_dir)

        if self.config.plot_umap:
            self.save_umap(results, output_dir)

        return metrics

    def compute_metrics(self, results: dict) -> dict:
        from wsi_mil.utils.metrics import compute_metrics_from_results

        cfg = self.config
        if results.get("predictions") is None:
            return {}
        return compute_metrics_from_results(
            results,
            positive_class=cfg.positive_class,
            average=cfg.average,
            class_names=cfg.class_names,
        )

    def print_metrics(self, metrics: dict) -> None:
        if not metrics:
            return
        from wsi_mil.utils.metrics import print_metrics as _print
        _print(metrics)

    @staticmethod
    def infer_linear_probe(
        checkpoint_path: str | Path,
        data_dir: str | Path,
        encoder_name: str,
        method_name: str,
        csv_path: str | Path,
        device: str = "cpu",
    ) -> dict:
        """Load slide embeddings from HDF5, run LinearProbeModel, return results dict.

        The returned dict is compatible with __call__().

        Args:
            checkpoint_path: LinearProbeModel checkpoint (.ckpt)
            data_dir:        directory containing HDF5 files
            encoder_name:    encoder name (e.g. "conch15_768")
            method_name:     aggregate method name (e.g. "titan")
            csv_path:        label CSV with columns case_id, label
            device:          torch device string
        """
        import csv as _csv
        import h5py
        import torch
        from wsi_mil.models import LinearProbeModel

        data_dir = Path(data_dir)
        _device = torch.device(device)

        label_dict: dict[str, int] = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                label_dict[row["case_id"]] = int(row["label"])

        model = LinearProbeModel.load_from_checkpoint(checkpoint_path, map_location=_device)
        model.to(_device).eval()

        key = f"{encoder_name}/slide_embedding/{method_name}/embedding"

        embeddings, labels, predictions, probabilities = [], [], [], []
        h5_paths, case_names, indices = [], [], []

        for idx, h5_path in enumerate(sorted(data_dir.glob("*.h5"))):
            case_id = h5_path.stem
            if case_id not in label_dict:
                continue
            with h5py.File(h5_path, "r") as f:
                if key not in f:
                    raise KeyError(f"{h5_path}: '{key}' not found.")
                emb = torch.from_numpy(f[key][:]).float().to(_device)

            with torch.no_grad():
                out = model(emb.unsqueeze(0))
                logits = out["logits"].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                pred = int(probs.argmax().item())

            embeddings.append(emb.cpu().numpy())
            labels.append(label_dict[case_id])
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy())
            h5_paths.append(h5_path)
            case_names.append(case_id)
            indices.append(idx)

        return {
            "embeddings":       np.stack(embeddings),
            "labels":           np.array(labels),
            "predictions":      np.array(predictions),
            "probabilities":    probabilities,
            "attentions":       None,
            "selected_indices": None,
            "indices":          indices,
            "h5_paths":         h5_paths,
            "case_names":       case_names,
        }

    def save_confusion_matrix(self, results: dict, output_dir: Path) -> None:
        from wsi_mil.utils.metrics import compute_confusion_matrix, plot_confusion_matrix

        cm = compute_confusion_matrix(results["labels"], results["predictions"])
        fname = Path(self.config.confusion_matrix_filename)
        stem, suffix = fname.stem, fname.suffix
        plot_confusion_matrix(
            cm,
            output_path=output_dir / fname,
            class_names=self.config.class_names,
        )
        plot_confusion_matrix(
            cm,
            output_path=output_dir / f"{stem}_normalized{suffix}",
            class_names=self.config.class_names,
            normalize=True,
        )

    def save_umap(self, results: dict, output_dir: Path) -> None:
        from wsi_mil.utils.umap import plot_umap

        cfg = self.config
        common = dict(
            embeddings=results["embeddings"],
            labels=results["labels"],
            class_names=cfg.class_names,
            predictions=results.get("predictions"),
            case_names=results.get("case_names"),
            figsize=tuple(cfg.umap_figsize),
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            random_state=cfg.umap_random_state,
            annotate=cfg.umap_annotate,
            show_misclassified=cfg.show_misclassified,
            point_size=cfg.umap_point_size,
            ring_size=cfg.umap_ring_size,
            ring_linewidth=cfg.umap_ring_linewidth,
        )

        # Base plot (labels only) — also computes and returns coords_2d
        coords_2d = plot_umap(
            output_path=output_dir / cfg.umap_filename,
            **common,
        )

        # Per-column overlay plots — reuse coords_2d, no recomputation
        if cfg.overlay_csv and cfg.overlay_cols:
            stem = Path(cfg.umap_filename).stem
            suffix = Path(cfg.umap_filename).suffix
            for col in cfg.overlay_cols:
                overlay = self._load_overlay(results["case_names"], cfg.overlay_csv, col)
                if overlay is None:
                    print(f"Warning: no data found for overlay column '{col}', skipping.")
                    continue
                plot_umap(
                    output_path=output_dir / f"{stem}_{col}{suffix}",
                    coords_2d=coords_2d,
                    overlay=overlay,
                    overlay_name=col,
                    **common,
                )

    @staticmethod
    def _load_overlay(
        case_names: list[str],
        csv_path: str,
        col: str,
    ) -> np.ndarray | None:
        import csv as _csv

        lookup: dict[str, str] = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                val = row.get(col, "")
                if val:
                    lookup[row["case_id"]] = val

        if not lookup:
            return None

        vals = [lookup.get(c) for c in case_names]
        if all(v is None for v in vals):
            return None

        return np.array(vals, dtype=object)
