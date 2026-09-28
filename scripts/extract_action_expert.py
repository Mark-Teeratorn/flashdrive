#!/usr/bin/env python3
"""Tool to extract and package fine-tuned Action Expert weights from Alpamayo checkpoints.

Extracts the 416 Action Expert parameters (Diffusion denoiser transformer,
action_in_proj, action_out_proj, delta_tokenizer) from full 10B training
checkpoints and saves them into a compact standalone file (~4.2 GB instead of 21 GB).

Usage:
    python scripts/extract_action_expert.py \
        --checkpoint /home/aimslab/checkpoints/checkpoint-600 \
        --output-dir /home/aimslab/checkpoints/action_experts/checkpoint-600
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractActionExpert")

EXPERT_MODULE_KEYWORDS = ("action", "expert", "diffusion", "delta")


def extract_action_expert(
    checkpoint_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if output_dir is None:
        # Default destination: checkpoints/action_experts/<checkpoint_name>
        output_dir = Path("/home/aimslab/checkpoints/action_experts") / checkpoint_dir.name
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_safetensors = output_dir / "action_expert.safetensors"

    if out_safetensors.exists() and not overwrite:
        logger.info(f"Target action_expert.safetensors already exists at: {out_safetensors}")
        return out_safetensors

    logger.info(f"Scanning checkpoint shards in: {checkpoint_dir}")
    shards = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
    # Exclude any existing action_expert.safetensors from source shards
    shards = [s for s in shards if not s.endswith("action_expert.safetensors")]

    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {checkpoint_dir}")

    logger.info(f"Found {len(shards)} checkpoint shards. Extracting Action Expert weights...")
    t0 = time.perf_counter()

    expert_weights: dict[str, torch.Tensor] = {}
    for shard_path in shards:
        logger.info(f"  Reading shard: {Path(shard_path).name} ...")
        shard_data = load_file(shard_path, device="cpu")
        for key, tensor in shard_data.items():
            if any(kw in key for kw in EXPERT_MODULE_KEYWORDS):
                expert_weights[key] = tensor.to(dtype=torch.bfloat16)
        del shard_data

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Extraction complete in {elapsed:.1f}s. "
        f"Extracted {len(expert_weights)} Action Expert tensors."
    )

    if not expert_weights:
        raise ValueError(f"No Action Expert tensors matching {EXPERT_MODULE_KEYWORDS} found in shards!")

    # Save to compact safetensors
    logger.info(f"Saving standalone Action Expert weights to: {out_safetensors} ...")
    save_file(expert_weights, str(out_safetensors))
    size_mb = out_safetensors.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {out_safetensors.name} ({size_mb:.1f} MB, {size_mb/1024:.2f} GB).")

    # Copy config.json if present
    src_config = checkpoint_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, output_dir / "config.json")
        logger.info("Copied config.json.")

    # Write extraction metadata
    meta = {
        "source_checkpoint": str(checkpoint_dir),
        "num_expert_tensors": len(expert_weights),
        "file_size_gb": round(size_mb / 1024, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expert_keys_sample": list(expert_weights.keys())[:10],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata.json.")

    logger.info("=" * 60)
    logger.info(f"Action Expert successfully packaged into: {output_dir}")
    logger.info("=" * 60)
    return out_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and package fine-tuned Action Expert from an Alpamayo checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to full checkpoint directory (e.g. /home/aimslab/checkpoints/checkpoint-600)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Target directory (default: /home/aimslab/checkpoints/action_experts/<checkpoint_name>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing action_expert.safetensors if it already exists",
    )
    args = parser.parse_args()

    extract_action_expert(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
