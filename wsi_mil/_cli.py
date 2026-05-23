"""CLI entry point.

Usage (config file):
    mil aggregate --config aggregate.yaml

Usage (CLI args):
    mil aggregate --method titan --data-dir /path/to/hdf5 --encoder conch15_768
    mil aggregate --method mean_pooling --data-dir /path/to/hdf5 --encoder conch15_768

CLI args override config file values (priority: CLI > YAML > defaults).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _override(cfg: dict, args, mapping: dict[str, str]) -> None:
    """Overwrite cfg with non-None values from args. mapping is {cfg_key: args_attr}."""
    for cfg_key, attr in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            cfg[cfg_key] = val


# ------------------------------------------------------------------
# Subcommand: aggregate
# ------------------------------------------------------------------

def cmd_aggregate(args):
    from wsi_mil.utils import WSIDataset

    cfg_dict = _load_yaml(args.config) if args.config else {}

    # flatten legacy titan subsection
    titan_sub = cfg_dict.pop("titan", {})
    for k, v in titan_sub.items():
        cfg_dict.setdefault(k, v)

    _override(cfg_dict, args, {
        "method":             "method",
        "model_name_or_path": "model_name_or_path",
        "device":             "device",
        "normalize":          "normalize",
        "patch_size_lv0":     "patch_size_lv0",
        "output_dir":         "output_dir",
        "version":            "version",
        "checkpoint_name":    "checkpoint_name",

    })

    dataset_cfg = cfg_dict.pop("dataset", {})
    _override(dataset_cfg, args, {
        "data_dir": "data_dir",
        "encoder":  "encoder",
    })

    for key in ("data_dir", "encoder"):
        if not dataset_cfg.get(key):
            sys.exit(
                f"Error: dataset.{key} is required. "
                f"Specify via --{key.replace('_', '-')} or in a config file."
            )

    method = cfg_dict.get("method", "mean_pooling")

    if method == "titan":
        from wsi_mil.commands import TITANAggregateCommand, TITANConfig
        titan_kwargs = {k: v for k, v in cfg_dict.items()
                       if k in TITANConfig.__dataclass_fields__}
        config = TITANConfig(**titan_kwargs)
        model_name_or_path = cfg_dict.get("model_name_or_path", "MahmoodLab/TITAN")
        cmd = TITANAggregateCommand(model_name_or_path=model_name_or_path, config=config)

    elif method in ("abmil", "abmil_top"):
        from wsi_mil.commands import ABMILAggregateCommand, ABMILAggregateConfig
        from wsi_mil.models import ABMIL
        abmil_kwargs = {k: v for k, v in cfg_dict.items()
                       if k in ABMILAggregateConfig.__dataclass_fields__}
        config = ABMILAggregateConfig(**abmil_kwargs)
        cmd = ABMILAggregateCommand(model_class=ABMIL, config=config)

    else:
        from wsi_mil.commands import SimpleAggregateCommand, SimpleAggregateConfig
        simple_kwargs = {k: v for k, v in cfg_dict.items()
                        if k in SimpleAggregateConfig.__dataclass_fields__}
        cmd = SimpleAggregateCommand(config=SimpleAggregateConfig(**simple_kwargs))

    dataset = WSIDataset(
        data_dir=dataset_cfg["data_dir"],
        encoder_name=dataset_cfg["encoder"],
    )
    return cmd(dataset)


# ------------------------------------------------------------------
# Subcommand: evaluate
# ------------------------------------------------------------------

def cmd_evaluate(args):
    from wsi_mil.commands import EvaluateCommand, EvaluateConfig

    cfg_dict = _load_yaml(args.config)
    dataset_cfg = cfg_dict.pop("dataset", {})

    config = EvaluateConfig(**{k: v for k, v in cfg_dict.items()
                               if k != "method_name"})
    method_name = cfg_dict.get("method_name", "abmil")

    results = EvaluateCommand.load_embeddings(
        data_dir=dataset_cfg["data_dir"],
        method_name=method_name,
        csv_path=dataset_cfg["csv_path"],
        encoder_name=dataset_cfg["encoder"],
        subtype_csv_path=dataset_cfg.get("subtype_csv_path"),
        subtype_col=dataset_cfg.get("subtype_col", "subtype"),
    )

    cmd = EvaluateCommand(config=config)
    cmd(results)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(prog="mil")
    sub = parser.add_subparsers(dest="command")

    p_agg = sub.add_parser("aggregate", help="compute slide embeddings and save to HDF5")
    p_agg.add_argument("--config", help="YAML config file (optional; CLI args take precedence)")

    # dataset
    p_agg.add_argument("--data-dir",  dest="data_dir",           metavar="DIR",  help="HDF5 directory")
    p_agg.add_argument("--encoder",                              metavar="NAME", help="encoder name (e.g. conch15_768)")
    p_agg.add_argument("--csv",       dest="csv_path",           metavar="FILE", help="label CSV path")

    # common
    p_agg.add_argument("--method",    choices=["mean_pooling", "nearest_cosine", "nearest_euclidean", "titan", "abmil", "abmil_top"],
                       help="aggregation method")
    p_agg.add_argument("--device",    help="device (auto / cuda / cpu)")

    # simple methods
    p_agg.add_argument("--normalize", action="store_true", default=None, help="L2-normalize slide embedding")

    # titan
    p_agg.add_argument("--model",      dest="model_name_or_path", metavar="PATH", help="TITAN model path or HuggingFace ID (default: MahmoodLab/TITAN)")
    p_agg.add_argument("--patch-size", dest="patch_size_lv0", type=int,  metavar="PX", help="level-0 patch size in pixels (default: 512)")

    # abmil
    p_agg.add_argument("--output-dir",      dest="output_dir",       metavar="DIR",  help="ABMIL output directory")
    p_agg.add_argument("--version",                                  metavar="VER",  help="ABMIL version")
    p_agg.add_argument("--checkpoint",      dest="checkpoint_name",  metavar="NAME", help="ABMIL checkpoint name")

    p_eval = sub.add_parser("evaluate", help="output metrics / UMAP / confusion matrix")
    p_eval.add_argument("--config", required=True, help="YAML config file")

    args = parser.parse_args()

    if args.command == "aggregate":
        cmd_aggregate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
