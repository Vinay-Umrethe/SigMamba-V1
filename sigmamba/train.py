# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
SigMamba training.
"""

import json
import os
import time
from datetime import timedelta
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from safetensors.torch import save_file as safe_save_file
from torch.utils.data import DataLoader

import sigmamba.option as option
from sigmamba.console import (
    console,
    make_training_progress,
    print_checkpoint_saved,
    print_run_summary,
    print_val_results,
)
from sigmamba.dataset import Dataset
from sigmamba.mamba import IS_OFFICIAL_MAMBA
from sigmamba.model import Model
from sigmamba.test import test

torch.set_default_dtype(torch.float32)


def sparsity_loss(scores: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalises dense anomaly predictions to encourage temporal sparsity."""
    return weight * torch.mean(torch.norm(scores, dim=0))


def smoothness_loss(scores: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalises abrupt frame-to-frame score changes."""
    diff = scores[1:] - scores[:-1]
    return weight * torch.sum(diff**2)


class RTFMLoss(nn.Module):
    """
    Ranking and Temporal Feature Magnitude loss for weakly-supervised detection.

    Combines binary cross-entropy with a magnitude-based ranking term that
    pushes abnormal features to have high L2 norm while keeping
    normal features compact.
    """

    def __init__(self, alpha: float = 1e-4, margin: float = 100.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.bce = nn.BCELoss()

    def forward(
        self,
        score_normal: torch.Tensor,
        score_abnormal: torch.Tensor,
        label_normal: torch.Tensor,
        label_abnormal: torch.Tensor,
        feat_normal: torch.Tensor,
        feat_abnormal: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.cat([score_normal, score_abnormal], dim=0).squeeze()
        labels = torch.cat([label_normal, label_abnormal], dim=0).squeeze()
        loss_cls = self.bce(scores, labels)

        mean_abn = torch.mean(feat_abnormal, dim=1)
        mean_nor = torch.mean(feat_normal, dim=1)
        loss_abn = torch.abs(self.margin - torch.norm(mean_abn, p=2, dim=1))
        loss_nor = torch.norm(mean_nor, p=2, dim=1)
        loss_mag = torch.mean((loss_abn + loss_nor) ** 2)

        return loss_cls + self.alpha * loss_mag


def train_step(
    normal_iter,
    abnormal_iter,
    model: nn.Module,
    args,
    optimizer: optim.Optimizer,
    accelerator: Accelerator,
    criterion: RTFMLoss,
) -> float:
    """
    Executes one gradient update using one batch from each class iterator.

    Returns:
        Scalar training loss for this step.
    """
    model.train()

    ninput, nlabel = next(normal_iter)
    ainput, alabel = next(abnormal_iter)
    input_feat = torch.cat([ninput, ainput], dim=0)

    score_abnormal, score_normal, feat_abn, feat_nor, scores = model(input_feat)

    flat_scores = scores.view(args.batch_size * args.seg_num * 2)
    abn_scores = flat_scores[args.batch_size * args.seg_num :]

    loss = (
        criterion(
            score_normal, score_abnormal, nlabel[: args.batch_size], alabel[: args.batch_size], feat_nor, feat_abn
        )
        + smoothness_loss(abn_scores, args.lambda_smooth)
        + sparsity_loss(abn_scores, args.lambda_sparse)
    )

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()

    return loss.item()


def save_checkpoint(
    name: str,
    save_models_dir: str,
    accelerator: Accelerator,
    model: nn.Module,
    args,
    step: int,
    auc: float,
    ap: float,
    f1: float,
    total_seconds: float,
    progress_console=None,
) -> None:
    """Saves accelerator state, safetensors weights, and a JSON metadata file."""
    save_dir = os.path.join(save_models_dir, f"checkpoint-{name}")
    os.makedirs(save_dir, exist_ok=True)

    accelerator.save_state(save_dir)
    unwrapped = accelerator.unwrap_model(model)
    safe_save_file(unwrapped.state_dict(), os.path.join(save_dir, "model.safetensors"))

    readable_time = str(timedelta(seconds=int(total_seconds)))
    metadata = {
        "step": step,
        "auc": float(auc),
        "average_precision": float(ap),
        "f1_score": float(f1),
        "training_duration": f"{readable_time} ({int(total_seconds)}s)",
        "feature_size": args.feature_size,
        "model_dim": args.model_dim,
        "model_depth": args.model_depth,
        "batch_size": args.batch_size,
        "seg_num": args.seg_num,
        "vision_model_id": args.vision_model,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print_checkpoint_saved(name, step, readable_time, print_to=progress_console)


def main():
    args = option.train_parser.parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_local_main_process

    train_nloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    train_aloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=False),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if len(train_nloader) == 0 or len(train_aloader) == 0:
        if is_main:
            console.log(
                f"[bold red]Batch size ({args.batch_size}) exceeds dataset size.[/bold red]  "
                f"Normal: {len(train_nloader.dataset)}, Abnormal: {len(train_aloader.dataset)}"
            )
        return

    if args.resume is None:
        os.makedirs(args.save_models, exist_ok=True)

    model = Model(
        feature_dim=args.feature_size,
        batch_size=args.batch_size,
        seg_num=args.seg_num,
        d_model=args.model_dim,
        depth=args.model_depth,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)
    criterion = RTFMLoss(alpha=1e-4, margin=100.0)

    if is_main:
        backend = "Official CUDA" if IS_OFFICIAL_MAMBA else "Pure PyTorch"
        print_run_summary(
            model_name=args.model_name,
            backend=backend,
            total_params=sum(p.numel() for p in model.parameters()),
            device=str(device),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            feature_size=args.feature_size,
            model_dim=args.model_dim,
            model_depth=args.model_depth,
            seg_num=args.seg_num,
            lambda_sparse=args.lambda_sparse,
            lambda_smooth=args.lambda_smooth,
        )

    model, optimizer, train_nloader, train_aloader, test_loader = accelerator.prepare(
        model, optimizer, train_nloader, train_aloader, test_loader
    )

    start_step = 1
    resumed_seconds = 0.0

    if args.resume:
        if is_main:
            console.log(f"[cyan]Resume[/cyan] loading checkpoint from [white]{args.resume}[/white]")
        accelerator.load_state(args.resume)
        meta_path = os.path.join(args.resume, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            start_step = meta.get("step", 0) + 1
            try:
                resumed_seconds = float(meta.get("training_duration", "0 (0)").split("(")[-1].strip(")s"))
            except (ValueError, IndexError):
                resumed_seconds = 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    best_result = -1.0
    overall_auc, ap, f1 = 0.5, 0.0, 0.0
    session_start = time.time()

    normal_iter = cycle(train_nloader)
    abnormal_iter = cycle(train_aloader)

    progress = make_training_progress()
    total_steps = args.num_steps - start_step + 1

    with progress:
        task = progress.add_task(
            f"Training {args.model_name}",
            total=total_steps,
            loss="n/a",
            auc=f"{overall_auc:.4f}",
        )

        for step in range(start_step, args.num_steps + 1):
            loss_val = train_step(normal_iter, abnormal_iter, model, args, optimizer, accelerator, criterion)

            if is_main:
                progress.update(task, advance=1, loss=f"{loss_val:.4f}", auc=f"{overall_auc:.4f}")

                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss_val}) + "\n")

            if is_main and step % args.save_steps == 0:
                total_secs = resumed_seconds + (time.time() - session_start)
                save_checkpoint(
                    str(step),
                    args.save_models,
                    accelerator,
                    model,
                    args,
                    step,
                    overall_auc,
                    ap,
                    f1,
                    total_secs,
                    progress_console=progress.console,
                )

            if step % 50 == 0:
                _, overall_auc, ap, f1 = test(dataloader=test_loader, model=model, device=device, verbose=False)

                if is_main:
                    metric_value = {"AUC": overall_auc, "AP": ap, "F1": f1}[args.metric]

                    print_val_results(
                        step=step,
                        auc=overall_auc,
                        ap=ap,
                        f1=f1,
                        best=best_result,
                        metric=args.metric,
                        print_to=progress.console,
                    )

                    with open(metrics_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "val_auc": overall_auc,
                                    "val_ap": ap,
                                    "val_f1": f1,
                                }
                            )
                            + "\n"
                        )

                    if metric_value > best_result:
                        best_result = metric_value
                        total_secs = resumed_seconds + (time.time() - session_start)
                        save_checkpoint(
                            "best",
                            args.save_models,
                            accelerator,
                            model,
                            args,
                            step,
                            overall_auc,
                            ap,
                            f1,
                            total_secs,
                            progress_console=progress.console,
                        )

                    progress.update(task, auc=f"{overall_auc:.4f}")


if __name__ == "__main__":
    main()
