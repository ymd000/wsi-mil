"""WSI-MIL — Multiple Instance Learning toolkit for Whole Slide Images."""

from .models import ABMIL, LinearProbeModel
from .commands import (
    TrainCommand, TrainConfig,
    SimpleAggregateCommand, SimpleAggregateConfig,
    TITANAggregateCommand, TITANConfig,
    ABMILAggregateCommand, ABMILCVAggregateConfig, ABMILCheckpointAggregateConfig,
    EvaluateCommand, EvaluateConfig,
)
from .utils import (
    FoldManager, FoldInfo,
    WSIDataset,
    mil_collate_fn,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "ABMIL",
    "LinearProbeModel",
    # Commands
    "TrainCommand", "TrainConfig",
    "SimpleAggregateCommand", "SimpleAggregateConfig",
    "TITANAggregateCommand", "TITANConfig",
    "ABMILAggregateCommand", "ABMILCVAggregateConfig", "ABMILCheckpointAggregateConfig",
    "EvaluateCommand", "EvaluateConfig",
    # Utils
    "FoldManager", "FoldInfo",
    "WSIDataset",
    "mil_collate_fn",
]
