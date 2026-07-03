"""
Contrastive Learning Module for YOLO11-UDA

Provides shared components for contrastive pretraining
using SimCLR or MoCo methods with YOLO11 backbone.
"""

from .model import IdentityDetect, ProjectionHead, ContrastiveYOLO, get_yolo_backbone_neck
from .dataset import CombinedDataset, RatioBatchSampler
from .freeze import (
    _set_bn_eval,
    freeze_backbone_upto_index,
    freeze_by_name_regex,
    build_optimizer_layerwise,
    print_trainable_report,
    YOLO11_BACKBONE_END,
    YOLO11_NECK_START,
    YOLO11_DETECT_IDX,
)
from .loss import info_nce_loss, moco_loss, dequeue_and_enqueue
from .utils import setup_environment, save_loss_curve, _csv_ints, _csv_list
from .ddp_utils import (
    init_distributed_mode,
    concat_all_gather,
    batch_shuffle_ddp,
    batch_unshuffle_ddp,
)

__all__ = [
    "IdentityDetect",
    "ProjectionHead",
    "ContrastiveYOLO",
    "get_yolo_backbone_neck",
    "CombinedDataset",
    "RatioBatchSampler",
    "freeze_backbone_upto_index",
    "freeze_by_name_regex",
    "build_optimizer_layerwise",
    "print_trainable_report",
    "info_nce_loss",
    "moco_loss",
    "dequeue_and_enqueue",
    "setup_environment",
    "save_loss_curve",
    "init_distributed_mode",
    "concat_all_gather",
    "batch_shuffle_ddp",
    "batch_unshuffle_ddp",
    "_csv_ints",
    "_csv_list",
    "YOLO11_BACKBONE_END",
    "YOLO11_NECK_START",
    "YOLO11_DETECT_IDX",
]
