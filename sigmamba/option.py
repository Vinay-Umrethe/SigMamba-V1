# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Configuration parameters for SigMamba training and evaluation.
"""

import argparse
import os

_shared = argparse.ArgumentParser(add_help=False)
_shared.add_argument("--features_path", type=str, default="features/", help="Path to pre-extracted video features")
_shared.add_argument("--test_anno", default="test_list.txt", help="Test annotation file")
_shared.add_argument("--output_dir", default="results/", help="Output directory for results and logs")
_shared.add_argument("--feature_size", type=int, default=1024, help="Input feature dimension")
_shared.add_argument("--seg_num", type=int, default=32, help="Number of temporal segments")
_shared.add_argument("--model_dim", type=int, default=512, help="Internal Mamba hidden dimension")
_shared.add_argument("--model_depth", type=int, default=3, help="Number of stacked Mamba layers")
_shared.add_argument("--workers", default=os.cpu_count(), type=int, help="DataLoader worker processes")


test_parser = argparse.ArgumentParser(description="SigMamba: Evaluation", parents=[_shared])
test_parser.add_argument("--detection_model", default=None, help="Path to the trained Mamba checkpoint weights")


train_parser = argparse.ArgumentParser(description="SigMamba: Training", parents=[_shared])
train_parser.add_argument("--train_anno", default="train_list.txt", help="Training annotation file")
train_parser.add_argument("--model_name", default="SigMamba", help="Model identifier used in logs")
train_parser.add_argument("--save_models", default="models/", help="Directory for saved checkpoints")

train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Adam learning rate")
train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
train_parser.add_argument("--num_steps", type=int, default=2000, help="Total training steps")
train_parser.add_argument(
    "--metric", type=str, choices=["AP", "AUC", "F1"], default="AUC", help="Metric used to select the best checkpoint"
)
train_parser.add_argument("--lambda_sparse", type=float, default=8e-3, help="Sparsity regularisation weight")
train_parser.add_argument("--lambda_smooth", type=float, default=8e-4, help="Temporal smoothness regularisation weight")

train_parser.add_argument(
    "--vision_model",
    type=str,
    default="google/siglip2-large-patch16-384",
    help="Vision encoder used for feature extraction",
)
train_parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint interval in steps")
train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
