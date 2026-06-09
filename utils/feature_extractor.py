# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
SigMamba Feature Extractor.

Extracts per-frame visual features from a video dataset using a SigLIP2 vision encoder.
Supports single and multi-GPU extraction with automatic resumption of incomplete runs.
"""

import argparse
import glob
import os
import sys
import threading
from queue import Empty, Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

sigmamba_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if sigmamba_root not in sys.path:
    sys.path.insert(0, sigmamba_root)

from sigmamba.console import console, make_progress, print_extractor_summary


class VideoFolderDataset(Dataset):
    """Iterates video files and yields sampled PIL frames at a fixed interval."""

    def __init__(self, video_files: list, frame_interval: int = 16):
        self.video_files = video_files
        self.frame_interval = frame_interval

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        count = 0
        success = True

        while success:
            success, frame = cap.read()
            if success and count % self.frame_interval == 0:
                try:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                except Exception:
                    pass
            count += 1

        cap.release()
        return frames, os.path.basename(video_path), len(frames) > 0


def _collate_single(batch):
    """Unwraps the batch wrapper added by DataLoader for batch_size=1."""
    return batch[0]


def get_pending_videos(dataset_path: str, save_dir: str) -> tuple[list, int, int]:
    """
    Scans dataset_path for video files and filters out already-processed ones.

    Returns:
        pending:  List of video paths not yet extracted.
        done:     Number of already-extracted videos.
        total:    Total video count in the dataset.
    """
    all_videos = [
        f
        for f in glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))
    ]

    pending, done = [], 0
    for video_path in all_videos:
        feat_path = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0] + ".txt")
        if os.path.exists(feat_path):
            done += 1
        else:
            pending.append(video_path)

    return pending, done, len(all_videos)


def extract_on_gpu(gpu_id: int, video_list: list, args, progress_queue) -> None:
    """
    Worker: loads model onto one GPU and extracts features for each video in video_list.
    Communicates completion back to the main process via progress_queue.

    Rich console is intentionally not used here.
    """
    device = torch.device(f"cuda:{gpu_id}")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).to(device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = VideoFolderDataset(video_list, frame_interval=args.frame_interval)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate_single)

    with torch.no_grad():
        for frames, video_name, success in loader:
            if not success:
                progress_queue.put((gpu_id, video_name, False))
                continue

            features_list = []
            for i in range(0, len(frames), args.batch_size):
                chunk = frames[i : i + args.batch_size]
                try:
                    inputs = processor(images=chunk, return_tensors="pt").to(device)
                    if hasattr(model, "get_image_features"):
                        feats = model.get_image_features(**inputs)
                    elif hasattr(model, "encode_image"):
                        feats = model.encode_image(**inputs)
                    else:
                        feats = model(**inputs).pooler_output
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    features_list.append(feats.cpu())
                except Exception:
                    continue

            if features_list:
                save_path = os.path.join(args.save_dir, f"{os.path.splitext(video_name)[0]}.txt")
                np.savetxt(save_path, torch.cat(features_list, dim=0).numpy(), fmt="%.6f")

            progress_queue.put((gpu_id, video_name, True))


def _drain_queue(queue, total: int, progress, task) -> None:
    """Reads queue until total items are done, advancing the progress bar."""
    completed = 0
    while completed < total:
        try:
            _gpu_id, video_name, success = queue.get(timeout=1.0)
            completed += 1
            progress.advance(task)
            if not success:
                progress.console.log(f"[yellow]Skipped[/yellow] {video_name} (no frames decoded)")
        except Empty:
            continue


def main():
    parser = argparse.ArgumentParser(description="SigMamba Dataset Feature Extractor")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to video dataset")
    parser.add_argument("--save_dir", type=str, default="features/siglip2", help="Output directory for features")
    parser.add_argument("--model_id", type=str, default="google/siglip2-large-patch16-384", help="HuggingFace model ID")
    parser.add_argument("--batch_size", type=int, default=128, help="Frames per inference batch")
    parser.add_argument("--frame_interval", type=int, default=16, help="Sample 1 frame every N frames")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (auto-detected if unset)")
    args = parser.parse_args()

    if args.num_gpus is None:
        args.num_gpus = max(1, torch.cuda.device_count())

    os.makedirs(args.save_dir, exist_ok=True)
    pending, done, total = get_pending_videos(args.dataset_path, args.save_dir)

    print_extractor_summary(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        total=total,
        done=done,
        remaining=len(pending),
    )

    if not pending:
        console.log("[green]All videos already processed![/green]")
        return

    progress = make_progress(color="yellow")

    if args.num_gpus == 1:
        console.log("[yellow]Mode[/yellow] single-GPU")
        queue: Queue = Queue()
        with progress:
            task = progress.add_task("Extracting features...", total=len(pending))
            worker = threading.Thread(target=extract_on_gpu, args=(0, pending, args, queue))
            worker.start()
            _drain_queue(queue, len(pending), progress, task)
            worker.join()

    else:
        console.log(f"[yellow]Mode[/yellow] multi-GPU ({args.num_gpus} GPUs)")
        mp.set_start_method("spawn", force=True)
        mp_queue = mp.Queue()

        chunk = len(pending) // args.num_gpus
        gpu_lists = [
            pending[i * chunk :] if i == args.num_gpus - 1 else pending[i * chunk : (i + 1) * chunk]
            for i in range(args.num_gpus)
        ]
        processes = [
            mp.Process(target=extract_on_gpu, args=(gpu_id, gpu_lists[gpu_id], args, mp_queue))
            for gpu_id in range(args.num_gpus)
        ]
        for p in processes:
            p.start()

        with progress:
            task = progress.add_task("Extracting features...", total=len(pending))
            _drain_queue(mp_queue, len(pending), progress, task)

        for p in processes:
            p.join()

    _, final_done, _ = get_pending_videos(args.dataset_path, args.save_dir)
    console.log(f"[green]Done![/green] {final_done}/{total} videos extracted to [white]{args.save_dir}[/white]")


if __name__ == "__main__":
    main()
