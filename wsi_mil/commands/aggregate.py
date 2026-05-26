"""AggregateCommand — patch embeddings → slide embeddings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from wsi_mil.utils import FoldManager, WSIDataset


METHOD_MEAN_POOLING      = "mean_pooling"
METHOD_NEAREST_COSINE    = "nearest_cosine"
METHOD_NEAREST_EUCLIDEAN = "nearest_euclidean"
METHOD_ABMIL             = "abmil"
METHOD_ABMIL_TOP         = "abmil_top"

SIMPLE_METHODS = {METHOD_MEAN_POOLING, METHOD_NEAREST_COSINE, METHOD_NEAREST_EUCLIDEAN}


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class SimpleAggregateConfig:
    """Configuration for SimpleAggregateCommand."""
    method: str = METHOD_MEAN_POOLING
    normalize: bool = False
    device: str = "auto"


@dataclass
class ABMILCVAggregateConfig:
    """Configuration for ABMILAggregateCommand — CV fold models (internal data)."""
    method: str = METHOD_ABMIL
    output_dir: str = "./outputs"
    version: int | str = "latest"
    checkpoint_name: str = "best"
    use_val_fold: bool = True
    normalize: bool = False
    device: str = "auto"
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class ABMILCheckpointAggregateConfig:
    """Configuration for ABMILAggregateCommand — single checkpoint (external data)."""
    method: str = METHOD_ABMIL
    checkpoint_path: str = ""
    normalize: bool = False
    device: str = "auto"
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class TITANConfig:
    """Configuration for TITANAggregateCommand."""
    patch_size_lv0: int = 512
    device: str = "auto"
    method_name: str = "titan"


# ------------------------------------------------------------------
# SimpleAggregateCommand
# ------------------------------------------------------------------

class SimpleAggregateCommand:
    """Model-free slide embedding aggregation.

    Supported methods: mean_pooling / nearest_cosine / nearest_euclidean

    Python:
        cmd = SimpleAggregateCommand(config=SimpleAggregateConfig(method="mean_pooling"))
        results = cmd(dataset)

    CLI:
        mil aggregate --config aggregate_mean_pooling.yaml
    """

    def __init__(self, config: SimpleAggregateConfig | None = None):
        self.config = config or SimpleAggregateConfig()
        if self.config.method not in SIMPLE_METHODS:
            raise ValueError(
                f"SimpleAggregateCommand: unsupported method {self.config.method!r}. "
                f"Choose from {SIMPLE_METHODS}."
            )

    def __call__(self, dataset, overwrite: bool = True) -> dict:
        device = self._resolve_device()
        encoder = dataset.encoder_name
        h5_files = dataset.h5_files
        labels = dataset.labels
        case_names = [Path(p).stem for p in h5_files]

        all_embeddings, all_labels, all_h5_paths, all_case_names = [], [], [], []

        for idx, h5_path in enumerate(h5_files):
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                key = f"{encoder}/features"
                if key not in f:
                    raise KeyError(f"{h5_path}: '{key}' not found.")
                x = torch.from_numpy(np.asarray(f[key])).float().to(device)

            result = self._compute_one(x)
            slide_emb = result["slide_embedding"].cpu().numpy()
            self._save_to_hdf5(h5_path, encoder, slide_emb, result["selected_index"], overwrite)

            all_embeddings.append(slide_emb)
            all_labels.append(labels[idx])
            all_h5_paths.append(h5_path)
            all_case_names.append(case_names[idx])
            print(f"[{idx + 1}/{len(h5_files)}] {h5_path.name}")

        return {
            "embeddings":       np.stack(all_embeddings),
            "labels":           np.array(all_labels),
            "predictions":      None,
            "probabilities":    None,
            "selected_indices": None,
            "attentions":       None,
            "indices":          list(range(len(h5_files))),
            "h5_paths":         all_h5_paths,
            "case_names":       all_case_names,
        }

    def _compute_one(self, x: torch.Tensor) -> dict:
        mn = self.config.method
        if mn == METHOD_MEAN_POOLING:
            return self._mean_pooling(x)
        elif mn == METHOD_NEAREST_COSINE:
            return self._nearest_cosine(x)
        elif mn == METHOD_NEAREST_EUCLIDEAN:
            return self._nearest_euclidean(x)
        raise ValueError(f"Unknown method: {mn!r}")

    def _mean_pooling(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.squeeze(0)
        slide_emb = x.mean(dim=0)
        if self.config.normalize:
            slide_emb = F.normalize(slide_emb, dim=0)
        return {"slide_embedding": slide_emb, "selected_index": None}

    def _nearest_cosine(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.squeeze(0)
        mean = x.mean(dim=0, keepdim=True)
        sims = F.cosine_similarity(x, mean.expand_as(x), dim=1)
        idx = int(sims.argmax().item())
        slide_emb = x[idx]
        if self.config.normalize:
            slide_emb = F.normalize(slide_emb, dim=0)
        return {"slide_embedding": slide_emb, "selected_index": idx}

    def _nearest_euclidean(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.squeeze(0)
        mean = x.mean(dim=0, keepdim=True)
        dists = torch.cdist(x, mean).squeeze(1)
        idx = int(dists.argmin().item())
        slide_emb = x[idx]
        if self.config.normalize:
            slide_emb = F.normalize(slide_emb, dim=0)
        return {"slide_embedding": slide_emb, "selected_index": idx}

    def _save_to_hdf5(
        self,
        h5_path: Path,
        encoder: str,
        slide_embedding: np.ndarray,
        selected_index: int | None,
        overwrite: bool,
    ) -> None:
        key = f"{encoder}/slide_embedding/{self.config.method}/embedding"
        with h5py.File(h5_path, "a") as f:
            if key in f:
                if not overwrite:
                    return
                del f[key]
            ds = f.create_dataset(key, data=slide_embedding)
            if selected_index is not None:
                ds.parent.attrs["selected_index"] = selected_index

    def _resolve_device(self) -> torch.device:
        d = self.config.device
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)


# ------------------------------------------------------------------
# ABMILAggregateCommand helpers
# ------------------------------------------------------------------

def _namespace_from_checkpoint(checkpoint_path: str) -> str:
    """Derive HDF5 namespace from a checkpoint path.

    Examples:
        outputs/version_2/fold_3/checkpoints/best.ckpt  → version_2_fold_3
        outputs/version_13/retrain_all/checkpoints/last.ckpt → version_13_all
    """
    parts = Path(checkpoint_path).parts
    for i, part in enumerate(parts):
        if part.startswith("version_") and part.split("_")[1].isdigit():
            if i + 1 < len(parts):
                nxt = parts[i + 1]
                if nxt == "retrain_all":
                    return f"{part}_all"
                if nxt != "checkpoints":
                    return f"{part}_{nxt}"
            return part
    raise ValueError(f"Cannot derive namespace from checkpoint path: {checkpoint_path!r}")


# ------------------------------------------------------------------
# ABMILAggregateCommand
# ------------------------------------------------------------------

class ABMILAggregateCommand:
    """ABMIL-based slide embedding aggregation.

    Accepts either ABMILCVAggregateConfig (K fold models, internal data)
    or ABMILCheckpointAggregateConfig (single checkpoint, external data).

    The HDF5 key is auto-derived as ``{method}_{namespace}``:
        abmil_version_2            ← CV aggregate, version 2
        abmil_version_2_fold_3     ← single fold checkpoint
        abmil_version_13_all       ← retrain_all checkpoint
    """

    def __init__(
        self,
        model_class,
        config: ABMILCVAggregateConfig | ABMILCheckpointAggregateConfig,
    ):
        self.model_class = model_class
        self.config = config
        self._models: list = []
        self._fold_manager: FoldManager | None = None

    def __call__(self, dataset, overwrite: bool = True) -> dict:
        cfg = self.config
        device = self._resolve_device()
        encoder = dataset.encoder_name

        if isinstance(cfg, ABMILCVAggregateConfig):
            version_dir = self._resolve_version_dir()
            self._load_cv_models(version_dir, device)
            namespace = version_dir.name
            val_fold_map: dict[int, int] = {
                idx: fold_info.fold_idx
                for fold_info in self._fold_manager.folds
                for idx in fold_info.val_indices
            } if cfg.use_val_fold else {}
        else:
            self._load_checkpoint(cfg.checkpoint_path, device)
            namespace = _namespace_from_checkpoint(cfg.checkpoint_path)
            val_fold_map = {}

        method_name = f"{cfg.method}_{namespace}"
        print(f"Slide embedding key: {encoder}/slide_embedding/{method_name}/")

        all_embeddings, all_labels, all_h5_paths, all_case_names = [], [], [], []
        all_predictions, all_probabilities, all_attentions, all_selected = [], [], [], []
        has_pred = has_attn = has_selected = False

        for idx, h5_path in enumerate(dataset.h5_files):
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                key = f"{encoder}/features"
                if key not in f:
                    raise KeyError(f"{h5_path}: '{key}' not found.")
                x = torch.from_numpy(np.asarray(f[key])).float().to(device)

            fold_idx = val_fold_map.get(idx, 0)
            result = self._compute_one(x, fold_idx, device)

            slide_emb = result["slide_embedding"].cpu().numpy()
            att      = result["attention"].cpu().numpy() if result["attention"] is not None else None
            pred     = result.get("pred_class")
            probs    = result["probs"].cpu().numpy()    if result.get("probs")  is not None else None
            selected = result.get("selected_index")

            self._save_to_hdf5(h5_path, encoder, method_name, slide_emb, att, pred, probs, selected, overwrite)

            all_embeddings.append(slide_emb)
            all_labels.append(dataset.labels[idx])
            all_h5_paths.append(h5_path)
            all_case_names.append(h5_path.stem)
            all_predictions.append(pred)
            all_probabilities.append(probs)
            all_attentions.append(att)
            all_selected.append(selected)
            if pred     is not None: has_pred     = True
            if att      is not None: has_attn     = True
            if selected is not None: has_selected = True
            print(f"[{idx + 1}/{len(dataset.h5_files)}] {h5_path.name}")

        return {
            "embeddings":       np.stack(all_embeddings),
            "labels":           np.array(all_labels),
            "predictions":      np.array(all_predictions) if has_pred else None,
            "probabilities":    all_probabilities if any(p is not None for p in all_probabilities) else None,
            "selected_indices": all_selected if has_selected else None,
            "attentions":       all_attentions if has_attn else None,
            "indices":          list(range(len(dataset.h5_files))),
            "h5_paths":         all_h5_paths,
            "case_names":       all_case_names,
        }

    def _compute_one(self, x: torch.Tensor, fold_idx: int, device: torch.device) -> dict:
        cfg = self.config
        if x.dim() == 2:
            x = x.unsqueeze(0)
        model = self._models[fold_idx]
        with torch.no_grad():
            outputs = model(x)
            logits = outputs["logits"].squeeze()
            probs  = torch.softmax(logits, dim=-1)
            pred_class = int(probs.argmax().item())
            attention = outputs.get("attention")
            if attention is not None:
                attention = attention.squeeze()

        x_2d = x.squeeze(0)
        if cfg.method == METHOD_ABMIL:
            slide_emb = (torch.matmul(attention, x_2d) if attention is not None
                         else x_2d.mean(0)).cpu()
            result = {"slide_embedding": slide_emb, "attention": attention, "probs": probs, "pred_class": pred_class}
        elif cfg.method == METHOD_ABMIL_TOP:
            top_idx = int(attention.argmax().item()) if attention is not None else 0
            slide_emb = x_2d[top_idx].cpu()
            result = {"slide_embedding": slide_emb, "attention": attention, "selected_index": top_idx}
        else:
            raise ValueError(f"ABMILAggregateCommand: unsupported method {cfg.method!r}")

        if cfg.normalize:
            result["slide_embedding"] = F.normalize(result["slide_embedding"], dim=0)
        return result

    def _resolve_version_dir(self) -> Path:
        cfg = self.config
        base = Path(cfg.output_dir)
        v = cfg.version
        if v == "latest":
            existing = sorted(
                [d for d in base.glob("version_*") if d.is_dir()],
                key=lambda d: int(d.name.split("_")[1]),
            )
            if not existing:
                raise FileNotFoundError(f"No version directories found in {base}")
            vdir = existing[-1]
            print(f"Using latest version: {vdir}")
            return vdir
        return base / f"version_{v}"

    def _load_cv_models(self, version_dir: Path, device: torch.device) -> None:
        fm = FoldManager(version_dir)
        fm.load()
        self._fold_manager = fm
        self._models = []
        for fold_idx in range(fm.num_folds):
            ckpt = fm.get_checkpoint_path(fold_idx, self.config.checkpoint_name)
            model = self.model_class.load_from_checkpoint(str(ckpt), **self.config.model_kwargs)
            model.to(device).eval()
            self._models.append(model)
            print(f"Loaded fold {fold_idx}: {ckpt}")

    def _load_checkpoint(self, checkpoint_path: str, device: torch.device) -> None:
        model = self.model_class.load_from_checkpoint(
            checkpoint_path, **self.config.model_kwargs
        )
        model.to(device).eval()
        self._models = [model]
        print(f"Loaded checkpoint: {checkpoint_path}")

    def _save_to_hdf5(
        self,
        h5_path: Path,
        encoder: str,
        method_name: str,
        slide_embedding: np.ndarray,
        attention: np.ndarray | None,
        prediction: int | None,
        probabilities: np.ndarray | None,
        selected_index: int | None,
        overwrite: bool,
    ) -> None:
        group_path = f"{encoder}/slide_embedding/{method_name}"
        with h5py.File(h5_path, "a") as f:
            if group_path in f:
                if not overwrite:
                    return
                del f[group_path]
            grp = f.create_group(group_path)
            grp.create_dataset("embedding", data=slide_embedding)
            if attention is not None:
                grp.create_dataset("attention", data=attention)
            if prediction is not None:
                grp.attrs["prediction"] = prediction
            if probabilities is not None:
                grp.create_dataset("probabilities", data=probabilities)
            if selected_index is not None:
                grp.attrs["selected_index"] = selected_index

    def _resolve_device(self) -> torch.device:
        d = self.config.device
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)


# ------------------------------------------------------------------
# TITANAggregateCommand
# ------------------------------------------------------------------


class TITANAggregateCommand:
    """TITAN-based slide embedding aggregation.

    Only conch15_768 features are supported (CONCH v1.5, 768-dim).

    Python:
        cmd = TITANAggregateCommand(
            model_name_or_path="MahmoodLab/TITAN",
            config=TITANConfig(patch_size_lv0=512),
        )
        results = cmd(dataset)
    """

    _COMPATIBLE_ENCODERS = {"conch15_768"}

    def __init__(self, model_name_or_path: str, config: TITANConfig | None = None):
        self.model_name_or_path = model_name_or_path
        self.config = config or TITANConfig()
        self.model = self._load_model()

    def _load_model(self):
        from wsi_mil.models.titan import TITAN
        from transformers import AutoModel
        raw = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        device = self._resolve_device()
        raw.to(device).eval()
        print(f"TITAN loaded on {device}")
        return TITAN(raw)

    def __call__(self, dataset, overwrite: bool = True) -> dict:
        cfg = self.config
        device = self._resolve_device()
        encoder = dataset.encoder_name
        if encoder not in self._COMPATIBLE_ENCODERS:
            raise ValueError(
                f"TITAN requires CONCH v1.5 features "
                f"(encoder: {sorted(self._COMPATIBLE_ENCODERS)}), got: {encoder!r}"
            )
        h5_files = dataset.h5_files
        labels = dataset.labels
        case_names = [Path(p).stem for p in h5_files]

        all_embeddings, all_labels, all_h5_paths, all_case_names = [], [], [], []

        for idx, h5_path in enumerate(h5_files):
            h5_path = Path(h5_path)
            with h5py.File(h5_path, "r") as f:
                if f"{encoder}/coordinates" not in f:
                    raise KeyError(
                        f"{h5_path}: '{encoder}/coordinates' not found. "
                        "TITAN requires coordinate data."
                    )
                patch_embs = torch.from_numpy(np.asarray(f[f"{encoder}/features"])).float().to(device)
                coords     = torch.from_numpy(np.asarray(f[f"{encoder}/coordinates"])).long().to(device)

            slide_emb = self.model.encode(
                patch_embs, coords, patch_size_lv0=cfg.patch_size_lv0
            ).cpu().numpy()

            self._save_to_hdf5(h5_path, encoder, slide_emb, overwrite)
            all_embeddings.append(slide_emb)
            all_labels.append(labels[idx])
            all_h5_paths.append(h5_path)
            all_case_names.append(case_names[idx])
            print(f"[{idx + 1}/{len(h5_files)}] {h5_path.name}")

        return {
            "embeddings":       np.stack(all_embeddings),
            "labels":           np.array(all_labels),
            "predictions":      None,
            "probabilities":    None,
            "selected_indices": None,
            "attentions":       None,
            "indices":          list(range(len(h5_files))),
            "h5_paths":         all_h5_paths,
            "case_names":       all_case_names,
        }

    def _save_to_hdf5(self, h5_path: Path, encoder: str, slide_embedding: np.ndarray, overwrite: bool) -> None:
        key = f"{encoder}/slide_embedding/{self.config.method_name}/embedding"
        with h5py.File(h5_path, "a") as f:
            if key in f:
                if not overwrite:
                    return
                del f[key]
            f.create_dataset(key, data=slide_embedding)

    def _resolve_device(self) -> torch.device:
        d = self.config.device
        if d == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(d)
