#!/usr/bin/env python3
"""Expand an RSL-RL feed-forward actor input while preserving its old policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-dim", type=int, required=True)
    parser.add_argument("--target-dim", type=int, required=True)
    parser.add_argument(
        "--target-indices",
        nargs="+",
        type=int,
        help=(
            "Target first-layer columns for source columns 0..source-dim-1. "
            "Defaults to a prefix-preserving expansion."
        ),
    )
    return parser.parse_args()


def expand_actor_weight(
    weight: torch.Tensor, target_dim: int, target_indices: list[int] | None = None
) -> tuple[torch.Tensor, list[int]]:
    if weight.ndim != 2:
        raise ValueError(f"Actor weight must be two-dimensional, got {weight.shape}")
    source_dim = weight.shape[1]
    indices = list(range(source_dim)) if target_indices is None else target_indices
    if len(indices) != source_dim:
        raise ValueError(f"Expected {source_dim} target indices, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError("Target indices must be unique")
    if any(index < 0 or index >= target_dim for index in indices):
        raise ValueError(f"Target indices must be inside [0, {target_dim})")

    expanded = weight.new_zeros((weight.shape[0], target_dim))
    expanded[:, indices] = weight
    return expanded, indices


def main() -> None:
    args = parse_args()
    if args.source_dim <= 0 or args.target_dim <= args.source_dim:
        raise SystemExit("Require target-dim > source-dim > 0")
    if args.source.resolve() == args.output.resolve():
        raise SystemExit("Source and output checkpoints must differ")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")
    key = "actor.0.weight"
    weight = state.get(key)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(f"Checkpoint does not contain a two-dimensional {key}")
    if weight.shape[1] != args.source_dim:
        raise ValueError(
            f"{key} has {weight.shape[1]} inputs, expected {args.source_dim}"
        )

    expanded, target_indices = expand_actor_weight(
        weight, args.target_dim, args.target_indices
    )
    state[key] = expanded
    checkpoint["actor_observation_expansion"] = {
        "source_dim": args.source_dim,
        "target_dim": args.target_dim,
        "source_to_target_indices": target_indices,
        "new_input_initialization": "zero",
        "optimizer_reset_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"Expanded {args.source} -> {args.output}: "
        f"actor input {args.source_dim} -> {args.target_dim}"
    )
    print("Resume with CUSTOM_DOG_LOAD_OPTIMIZER=0")


if __name__ == "__main__":
    main()
