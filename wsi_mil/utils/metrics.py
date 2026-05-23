"""Classification metrics: computation, display, and confusion matrix plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute confusion matrix. cm[i, j] = count of true label i predicted as j."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1,
    average: str = "macro",
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute metrics for binary or multiclass classification."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = int(max(y_true.max(), y_pred.max())) + 1

    if num_classes == 2:
        return _binary(y_true, y_pred, positive_class, class_names)
    return _multiclass(y_true, y_pred, average, class_names)


def compute_metrics_from_results(
    results: dict,
    positive_class: int = 1,
    average: str = "macro",
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute metrics directly from an EvaluateCommand.load_embeddings results dict."""
    return compute_metrics(
        y_true=results["labels"],
        y_pred=results["predictions"],
        positive_class=positive_class,
        average=average,
        class_names=class_names,
    )


def print_metrics(metrics: dict, title: str = "Classification Metrics") -> None:
    print(_format_metrics(metrics, title))


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    class_names: Optional[dict[int, str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    cmap: str = "Blues",
    figsize: tuple = (8, 6),
) -> None:
    """Save a confusion matrix heatmap as a PNG."""
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = {i: str(i) for i in range(num_classes)}
    labels = [class_names.get(i, str(i)) for i in range(num_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        plot_data = np.where(row_sums > 0, cm / row_sums, 0.0)
        colorbar_label = "Proportion"
    else:
        plot_data = cm.astype(float)
        colorbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_data, interpolation="nearest", cmap=cmap, vmin=0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=11)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    thresh = plot_data.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            v = plot_data[i, j]
            text = f"{v:.2f}\n({cm[i, j]})" if normalize else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="white" if v > thresh else "black")

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title or "Confusion Matrix", fontsize=13)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _binary(y_true, y_pred, positive_class, class_names):
    y_tb = (y_true == positive_class).astype(int)
    y_pb = (y_pred == positive_class).astype(int)
    tp = int(np.sum((y_tb == 1) & (y_pb == 1)))
    tn = int(np.sum((y_tb == 0) & (y_pb == 0)))
    fp = int(np.sum((y_tb == 0) & (y_pb == 1)))
    fn = int(np.sum((y_tb == 1) & (y_pb == 0)))
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ba   = (sens + spec) / 2
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    unique = np.unique(np.concatenate([y_true, y_pred]))
    neg = [c for c in unique if c != positive_class]
    negative_class = int(neg[0]) if neg else (1 - positive_class)
    r = dict(tp=tp, tn=tn, fp=fp, fn=fn,
             accuracy=acc, sensitivity=sens, specificity=spec, precision=prec,
             balanced_accuracy=ba, f1_score=f1,
             positive_class=positive_class, negative_class=negative_class)
    if class_names:
        r["class_names"] = class_names
    return r


def _multiclass(y_true, y_pred, average, class_names):
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    acc = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    sens_list, spec_list, prec_list, sup_list = [], [], [], []
    for c in range(num_classes):
        tp = cm[c, c]; fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        sup = cm[c].sum()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        prec_list.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        sup_list.append(sup)
    s = np.array(sens_list); sp = np.array(spec_list)
    pr = np.array(prec_list); su = np.array(sup_list)
    if average == "macro":
        sens, spec, prec = s.mean(), sp.mean(), pr.mean()
    elif average == "weighted":
        w = su / su.sum() if su.sum() > 0 else np.ones(num_classes) / num_classes
        sens, spec, prec = (s * w).sum(), (sp * w).sum(), (pr * w).sum()
    else:
        raise ValueError(f"Unknown average: {average!r}")
    ba = (sens + spec) / 2
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    r = dict(accuracy=float(acc), sensitivity=float(sens), specificity=float(spec),
             precision=float(prec), balanced_accuracy=float(ba), f1_score=float(f1),
             per_class_sensitivity=s.tolist(), per_class_specificity=sp.tolist(),
             per_class_precision=pr.tolist(), confusion_matrix=cm.tolist())
    if class_names:
        r["class_names"] = class_names
    return r


def _format_metrics(metrics: dict, title: str) -> str:
    lines = ["=" * 50, f" {title}", "=" * 50]
    if "positive_class" in metrics:
        cn = metrics.get("class_names", {})
        pc, nc = metrics["positive_class"], metrics["negative_class"]
        lines += ["", f"  Positive: {pc} ({cn.get(pc, pc)})",
                      f"  Negative: {nc} ({cn.get(nc, nc)})"]
    keys = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"), ("precision", "Precision"),
            ("balanced_accuracy", "Balanced Accuracy"), ("f1_score", "F1-Score")]
    w = max(len(l) for _, l in keys)
    lines.append("")
    for k, l in keys:
        if k in metrics:
            lines.append(f"  {l:<{w}} : {metrics[k]:.4f}")
    if "tp" in metrics:
        cn = metrics.get("class_names", {}); pc = metrics["positive_class"]
        nc = metrics["negative_class"]
        lines += ["", "-" * 50, " Confusion Matrix",
                  f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}"]
    lines += ["", "=" * 50]
    return "\n".join(lines)
