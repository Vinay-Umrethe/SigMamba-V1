# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Dataset management for SigMamba Training.
"""

import os

import numpy as np
import torch
import torch.utils.data as data

from sigmamba.console import console

torch.set_default_dtype(torch.float32)


def read_features(feature_path: str) -> torch.Tensor:
    """
    Loads pre-extracted video features from a .txt file.

    Returns:
        Tensor of shape (T, C).
    """
    return torch.from_numpy(np.loadtxt(feature_path).astype(np.float32))


def process_features(feat: np.ndarray, length: int) -> torch.Tensor:
    """
    Uniformly resamples a variable-length feature sequence to a fixed temporal length.

    Args:
        feat:   NumPy array of shape (T, C).
        length: Target number of segments.

    Returns:
        Tensor of shape (length, C).
    """
    new_feat = np.zeros((length, feat.shape[1]), dtype=np.float32)
    boundaries = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        start, end = boundaries[i], boundaries[i + 1]
        new_feat[i] = np.mean(feat[start:end], axis=0) if start != end else feat[start].copy()
    return torch.from_numpy(new_feat)


class Dataset(data.Dataset):
    """
    Loads and preprocesses pre-extracted video features for MIL-based anomaly detection.

    In training mode the dataset returns fixed-length sampled segments.
    In test mode it returns the full unsampled sequence alongside annotation metadata.
    """

    def __init__(self, args, is_normal: bool = True, test_mode: bool = False):
        self.is_normal = is_normal
        self.test_mode = test_mode
        self.seg_num = args.seg_num

        annotation_path = args.test_anno if self.test_mode else args.train_anno
        self.list = self._build_features_list(args.features_path, annotation_path)

    def __len__(self) -> int:
        return len(self.list)

    def __getitem__(self, index):
        label = torch.tensor(0.0 if self.is_normal else 1.0)

        if self.test_mode:
            feat_path, num_frames, start_end_couples, file = self.list[index]
            features = read_features(feat_path)
            return features, label, start_end_couples, num_frames, file

        feat_path = self.list[index]
        features = process_features(read_features(feat_path).numpy(), self.seg_num)
        return features, label

    def _build_features_list(self, features_path: str, annotation_path: str) -> list:
        assert os.path.exists(features_path), f"Feature directory not found: {features_path}"

        file_map = {}
        with console.status(f"[cyan]Indexing features in [white]{features_path}[/white]..."):
            for root, _, files in os.walk(features_path):
                for filename in files:
                    if not filename.endswith(".txt"):
                        continue
                    basename = os.path.splitext(filename)[0]
                    key = basename.lower().replace("_x264", "").replace("_264", "").replace("_", "")
                    file_map[key] = os.path.join(root, filename)

        console.log(f"[cyan]Dataset[/cyan] indexed [white]{len(file_map)}[/white] feature files")

        with open(annotation_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        console.log(
            f"[cyan]Dataset[/cyan] matching [white]{len(lines)}[/white] annotations from [white]{annotation_path}[/white]"
        )

        features_list = []
        missing = 0
        last_missing_key = ""

        for line in lines:
            items = line.split()
            if len(items) < 2:
                continue

            original_file = items[0].replace("\\", "/")
            basename = os.path.splitext(os.path.basename(original_file))[0]
            key = basename.lower().replace("_x264", "").replace("_264", "").replace("_", "")
            final_path = file_map.get(key)

            if final_path is None:
                if missing < 10:
                    console.log(f"[yellow]Missing[/yellow]  '{items[0]}' → key='{key}'")
                last_missing_key = key
                missing += 1
                continue

            cls_name = items[1]

            if self.test_mode:
                if len(items) < 4:
                    continue
                num_frames = int(items[2])
                try:
                    start_end_couples = [int(x) for x in items[3:]]
                except ValueError:
                    start_end_couples = [-1, -1]
                features_list.append((final_path, num_frames, start_end_couples, basename))
            else:
                if (cls_name.lower() == "normal") == self.is_normal:
                    features_list.append(final_path)

        label_tag = "normal" if self.is_normal else "abnormal"
        skipped = f"  [yellow]({missing} skipped)[/yellow]" if missing else ""
        console.log(f"[cyan]Dataset[/cyan] ready:[white]{len(features_list)}[/white] {label_tag} samples{skipped}")

        if not features_list:
            console.log("[bold red]CRITICAL[/bold red] No samples matched. Check paths and annotation format.")
            if missing:
                console.log(f"[red]  Last missing key :[/red] {last_missing_key}")
            console.log(f"[red]  Sample dir keys  :[/red] {list(file_map.keys())[:10]}")
            raise ValueError("Dataset is empty. Verify --features_path and annotation file.")

        return features_list
