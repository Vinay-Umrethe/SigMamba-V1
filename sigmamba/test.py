# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Evaluation for SigMamba on test list.
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

import sigmamba.option as option
from sigmamba.console import console, make_test_progress, print_val_results
from sigmamba.dataset import Dataset
from sigmamba.model import Model


def _load_weights(model: Model, path: str) -> Model:
    """Loads model weights."""
    if path.endswith(".safetensors"):
        state = load_file(path)
    else:
        state = torch.load(path, weights_only=True)
    model.load_state_dict(state)
    return model


def get_ground_truth(start_end_couples: list, num_frames: int, device: torch.device) -> torch.Tensor:
    """
    Builds a binary ground-truth tensor of length num_frames.
    Anomalous frame ranges are marked 1; normal frames are 0.
    """
    ground_truth = torch.zeros(num_frames, device=device)
    for start, end in zip(start_end_couples[::2], start_end_couples[1::2], strict=False):
        if start.item() != -1 and end.item() != -1:
            ground_truth[start.item() : end.item()] = 1.0
    return ground_truth


def test(dataloader: DataLoader, model: Model, device: torch.device, verbose: bool = True):
    """
    Runs inference on the test list and computes frame-level AUC, AP, and F1.

    Args:
        verbose: When False the per-video progress bar is suppressed.
                 Pass False when calling from the training loop to avoid
                 nesting a progress bar inside the training progress bar.

    Returns:
        per_video_results: Dict with keys video and AUC.
        overall_auc:       Frame-level ROC-AUC over the full test set.
        ap:                Average Precision.
        f1:                Optimal F1 score across all decision thresholds.
    """
    per_video_results = {"video": [], "AUC": []}

    model.to(device).eval()

    all_preds_list = []
    all_gts_list = []

    total = len(dataloader)
    progress = make_test_progress()

    with torch.no_grad():
        with progress if verbose else nullcontext():
            task = progress.add_task("Evaluating", total=total) if verbose else None

            for features, _label, start_end_couples, num_frames, file in dataloader:
                features = features.to(device)
                _, _, _, _, scores = model(features)

                sig = scores.squeeze()
                segment_len = num_frames.item() // sig.size(0)
                sig = sig.repeat_interleave(segment_len)
                if len(sig) < num_frames.item():
                    sig = torch.cat([sig, sig[-1].expand(num_frames.item() - len(sig))])

                cur_gt = get_ground_truth(start_end_couples, num_frames.item(), device)

                all_preds_list.append(sig.cpu())
                all_gts_list.append(cur_gt.cpu())

                sig_np = sig.cpu().numpy()
                gt_np = cur_gt.cpu().numpy()

                if len(np.unique(gt_np)) < 2:
                    video_auc = 0.5
                else:
                    try:
                        fpr, tpr, _ = roc_curve(gt_np, sig_np)
                        video_auc = auc(fpr, tpr)
                    except Exception:
                        video_auc = 0.5

                per_video_results["video"].append(file)
                per_video_results["AUC"].append(video_auc)

                if verbose:
                    progress.advance(task)

    pred_np = torch.cat(all_preds_list).numpy()
    gt_np = torch.cat(all_gts_list).numpy()

    try:
        if len(np.unique(gt_np)) < 2:
            raise ValueError("Test list contains only one class.")
        fpr, tpr, _ = roc_curve(gt_np, pred_np)
        overall_auc = auc(fpr, tpr)
        ap = average_precision_score(gt_np, pred_np)
        precision, recall, _ = precision_recall_curve(gt_np, pred_np)
        f1 = float(np.max(2 * precision * recall / (precision + recall + 1e-8)))
    except Exception:
        overall_auc, ap, f1 = 0.5, 0.0, 0.0

    return per_video_results, overall_auc, ap, f1


def main():
    args = option.test_parser.parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = Model(
        feature_dim=args.feature_size,
        batch_size=1,
        seg_num=args.seg_num,
        d_model=args.model_dim,
        depth=args.model_depth,
    )

    # Prepare for device/optimization
    model, test_loader = accelerator.prepare(model, test_loader)

    if args.detection_model:
        model = _load_weights(model, args.detection_model)
    else:
        accelerator.print("[yellow]Warning: No detection_model provided. Evaluating with random weights.[/yellow]")

    per_video, overall_auc, ap, f1 = test(dataloader=test_loader, model=model, device=device, verbose=True)

    if accelerator.is_main_process:
        print_val_results(step=0, auc=overall_auc, ap=ap, f1=f1, best=overall_auc, metric=args.metric)

        video_sub_dir = os.path.basename(os.path.dirname(per_video["video"][0][0]))
        results_path = os.path.join(args.output_dir, "AUC", video_sub_dir, "results.txt")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w") as f:
            for video, single_auc in zip(per_video["video"], per_video["AUC"], strict=False):
                f.write(f"Video: {video}, AUC: {single_auc:.4f}\n")
            f.write(f"Overall AUC: {overall_auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}\n")

        console.log(f"[green]Results saved to[/green] {results_path}")


if __name__ == "__main__":
    main()
