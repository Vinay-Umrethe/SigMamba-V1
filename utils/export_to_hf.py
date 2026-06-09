# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Vinay Umrethe. See the LICENSE file for details.

"""
Exports the trained SigMamba checkpoint and merges it with its vision encoder,
compatible with HuggingFace transformers.
"""

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModel

sigmamba_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if sigmamba_root not in sys.path:
    sys.path.insert(0, sigmamba_root)

from sigmamba.console import console, print_export_success, print_export_summary
from sigmamba_release.configuration_sigmamba import SigMambaConfig
from sigmamba_release.modeling_sigmamba import SigMambaForVideoClassification


def _resolve_vision_id(meta: dict, cli_arg: str | None) -> str:
    """Returns the vision model ID to use, with CLI taking priority over metadata."""
    if cli_arg is not None:
        meta_id = meta.get("vision_model_id")
        if meta_id and cli_arg != meta_id:
            console.log(
                f"[yellow]Export[/yellow] CLI [white]{cli_arg!r}[/white] overrides metadata [white]{meta_id!r}[/white]"
            )
        return cli_arg
    if meta.get("vision_model_id"):
        return meta["vision_model_id"]
    raise ValueError("No vision_model_id in metadata and --vision_model was not provided.")


def _load_weights(checkpoint_path: str) -> dict:
    """Loads weights from a checkpoint directory."""
    safe_path = os.path.join(checkpoint_path, "model.safetensors")
    # Fallback.
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safe_path):
        return load_file(safe_path)
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(f"No weights found in {checkpoint_path!r}.")


def _patch_config_source(source: str, config: SigMambaConfig, vision_id: str) -> str:
    """Overwrites default parameter values in configuration_sigmamba.py from the checkpoint config."""
    int_params = {
        "d_conv": config.d_conv,
        "d_model": config.d_model,
        "d_state": config.d_state,
        "depth": config.depth,
        "expand": config.expand,
        "feature_dim": config.feature_dim,
        "num_classes": getattr(config, "num_classes", 1),
        "seg_num": config.seg_num,
    }
    str_params = {
        "dtype": "float32",
        "model_type": "sigmamba",
        "vision_model_id": vision_id,
    }
    for param, val in int_params.items():
        source = re.sub(rf"\b{param}\s*=\s*\d+", f"{param}={val}", source)
    for param, val in str_params.items():
        source = re.sub(rf'\b{param}\s*=\s*"[^"]*"', f'{param}="{val}"', source)
    return source


def export(checkpoint_path: str, output_dir: str, vision_model_cli: str | None) -> None:
    console.log(f"[cyan]Export[/cyan] checkpoint=[white]{checkpoint_path}[/white]")

    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_path!r}.")

    with open(metadata_path) as f:
        meta = json.load(f)

    vision_id = _resolve_vision_id(meta, vision_model_cli)
    print_export_summary(meta, vision_id)

    config = SigMambaConfig(
        feature_dim=meta.get("feature_size", 1024),
        d_model=meta.get("model_dim", 768),
        depth=meta.get("model_depth", 8),
        seg_num=meta.get("seg_num", 32),
        d_state=meta.get("d_state", 16),
        d_conv=meta.get("d_conv", 4),
        expand=meta.get("expand", 2),
        vision_model_id=vision_id,
    )
    config.auto_map = {
        "AutoConfig": "configuration_sigmamba.SigMambaConfig",
        "AutoModel": "modeling_sigmamba.SigMambaForVideoClassification",
        "AutoProcessor": "preprocessor_sigmamba.SigMambaImageProcessor",
        "AutoImageProcessor": "preprocessor_sigmamba.SigMambaImageProcessor",
    }
    config.architectures = ["SigMambaForVideoClassification"]

    with console.status("[cyan]Loading Mamba checkpoint...[/cyan]"):
        state_dict = _load_weights(checkpoint_path)
    console.log(f"[cyan]Export[/cyan] Mamba weights: [white]{len(state_dict):,}[/white] keys")

    with console.status(f"[cyan]Downloading {vision_id!r}...[/cyan]"):
        vision_state = AutoModel.from_pretrained(vision_id).state_dict()
    console.log(f"[cyan]Export[/cyan] Vision weights: [white]{len(vision_state):,}[/white] keys")

    merged = {**state_dict, **{f"vision_model.{k}": v for k, v in vision_state.items()}}
    console.log(f"[cyan]Export[/cyan] Merged: [white]{len(merged):,}[/white] total keys")

    with console.status("[cyan]Initialising unified architecture...[/cyan]"):
        model = SigMambaForVideoClassification(config)
        missing, unexpected = model.load_state_dict(merged, strict=True)

    if missing:
        console.log(f"[yellow]Missing keys:[/yellow] {missing}")
    if unexpected:
        console.log(f"[yellow]Unexpected keys:[/yellow] {unexpected}")
    console.log("[green]Weights loaded successfully![/green]")

    os.makedirs(output_dir, exist_ok=True)

    with console.status(f"[cyan]Saving to {output_dir!r}...[/cyan]"):
        model.save_pretrained(output_dir)
        config.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "preprocessor_config.json"), "w") as f:
            json.dump(
                {
                    "image_processor_type": "SigMambaImageProcessor",
                    "seg_num": config.seg_num,
                    "auto_map": {"AutoImageProcessor": "preprocessor_sigmamba.SigMambaImageProcessor"},
                },
                f,
                indent=4,
            )

        with open(os.path.join("sigmamba_release", "configuration_sigmamba.py")) as f:
            config_src = f.read()
        with open(os.path.join(output_dir, "configuration_sigmamba.py"), "w") as f:
            f.write(_patch_config_source(config_src, config, vision_id))

        shutil.copy(os.path.join("sigmamba_release", "modeling_sigmamba.py"), output_dir)
        shutil.copy(os.path.join("sigmamba_release", "preprocessor_sigmamba.py"), output_dir)

    console.log(f"[green]Saved to[/green] [white]{output_dir}[/white]")
    print_export_success(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="SigMamba HuggingFace transformers Export")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--output", type=str, default="sigmamba_hf_model", help="Output directory")
    parser.add_argument(
        "--vision_model",
        type=str,
        default=None,
        help="Vision encoder used, HuggingFace ID (overrides metadata when provided)",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.vision_model)


if __name__ == "__main__":
    main()
