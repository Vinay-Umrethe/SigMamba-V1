# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
SigMamba List Generator.

Produces train_list.txt and test_list.txt from pre-extracted feature files
and dataset annotation metadata. These list files are used by
the Dataset class in sigmamba/dataset.py.
"""

import argparse
import glob
import os
import sys

import pandas as pd

sigmamba_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if sigmamba_root not in sys.path:
    sys.path.insert(0, sigmamba_root)

from sigmamba.console import console, make_progress


def _clean_key(filename: str) -> str:
    """Normalises a filename to a lookup key matching dataset.py."""
    filename = filename.replace("\\", "/")
    basename = os.path.splitext(os.path.basename(filename))[0]
    return basename.lower().replace("_x264", "").replace("_264", "").replace("_", "")


def get_frame_count(feat_path: str) -> int:
    """
    Estimates the total video frame count from a feature file.

    Features are extracted at one per frame_interval frames (default 16),
    so the estimate is: num_feature_rows x frame_interval.
    """
    with open(feat_path) as f:
        num_rows = sum(1 for _ in f)
    return num_rows * 16


def _build_file_map(features_dir: str) -> dict[str, str]:
    """Returns a {clean_key: absolute_path} map for all .txt feature files."""
    file_map = {}
    with console.status(f"[blue]Indexing features in [white]{features_dir}[/white]..."):
        for path in glob.glob(os.path.join(features_dir, "*.txt")):
            file_map[_clean_key(path)] = path.replace("\\", "/")
    console.log(f"[blue]Lists[/blue] indexed [white]{len(file_map)}[/white] feature files")
    return file_map


def generate_train_list(file_map: dict, train_csv: str, output_path: str) -> None:
    """
    Builds a training list from the CSV with video_name and label columns.

    Output format per line: feature_path label
    """
    console.log(f"[blue]Lists[/blue] processing train split from [white]{train_csv}[/white]")
    df = pd.read_csv(train_csv)

    valid_lines = []
    missing = 0

    progress = make_progress(color="blue")
    with progress:
        task = progress.add_task("Building train list...", total=len(df))
        for _, row in df.iterrows():
            feature_path = file_map.get(_clean_key(row["video_name"]))
            if feature_path:
                valid_lines.append(f"{feature_path} {row['label']}")
            else:
                missing += 1
            progress.advance(task)

    with open(output_path, "w") as f:
        f.write("\n".join(valid_lines))

    skipped = f"  [yellow]({missing} missing)[/yellow]" if missing else ""
    console.log(
        f"[green]Train list generated[/green] [white]{len(valid_lines)}[/white] entries: {output_path}{skipped}"
    )


def generate_test_list(file_map: dict, temporal_anno: str, output_path: str) -> None:
    """
    Builds a test list from the temporal annotation file.

    Each annotation line is expected as:
    video_name class start end [start end ...] -1 -1

    Output format per line:
    feature_path class num_frames start end ...
    """
    console.log(f"[blue]Lists[/blue] processing test split from [white]{temporal_anno}[/white]")
    with open(temporal_anno) as f:
        anno_lines = [ln for ln in f.read().splitlines() if ln.strip()]

    valid_lines = []
    missing = 0

    progress = make_progress(color="blue")
    with progress:
        task = progress.add_task("Building test list...", total=len(anno_lines))
        for line in anno_lines:
            parts = line.split()
            if len(parts) < 2:
                progress.advance(task)
                continue

            video_name, cls_name, *timestamps = parts
            feature_path = file_map.get(_clean_key(video_name))

            if feature_path:
                num_frames = get_frame_count(feature_path)
                valid_lines.append(f"{feature_path} {cls_name} {num_frames} {' '.join(timestamps)}")
            else:
                missing += 1
            progress.advance(task)

    with open(output_path, "w") as f:
        f.write("\n".join(valid_lines))

    skipped = f"  [yellow]({missing} missing)[/yellow]" if missing else ""
    console.log(f"[green]Test list generated[/green] [white]{len(valid_lines)}[/white] entries: {output_path}{skipped}")


def main():
    parser = argparse.ArgumentParser(description="SigMamba Dataset List Generator")
    parser.add_argument("--features", type=str, required=True, help="Path to feature files directory")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to train.csv (video_name, label columns)")
    parser.add_argument("--temporal_anno", type=str, default=None, help="Path to Temporal_Anomaly_Annotation_Test.txt")
    parser.add_argument("--train_output", type=str, default="train_list.txt", help="Output path for the training list")
    parser.add_argument("--test_output", type=str, default="test_list.txt", help="Output path for the test list")
    args = parser.parse_args()

    if not args.train_csv and not args.temporal_anno:
        parser.error("At least one of --train_csv or --temporal_anno must be provided.")

    file_map = _build_file_map(args.features)

    if args.train_csv:
        generate_train_list(file_map, args.train_csv, args.train_output)

    if args.temporal_anno:
        generate_test_list(file_map, args.temporal_anno, args.test_output)

    console.log("[green]List generation complete.[/green]")


if __name__ == "__main__":
    main()
