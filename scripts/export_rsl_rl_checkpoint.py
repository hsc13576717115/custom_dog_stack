#!/usr/bin/env python3
"""Export a feed-forward PPO actor or distillation student to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation", choices=ACTIVATIONS, default="elu")
    return parser.parse_args()


def build_actor(state_dict: dict[str, torch.Tensor], activation: str) -> nn.Sequential:
    if any(key.startswith("student.") for key in state_dict):
        prefix = "student."
    elif any(key.startswith("actor.") for key in state_dict):
        prefix = "actor."
    else:
        raise ValueError("Checkpoint contains neither actor nor student parameters")
    actor_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    weight_indices = sorted(
        int(key.split(".", maxsplit=1)[0])
        for key in actor_state
        if key.endswith(".weight")
    )
    if not weight_indices or weight_indices != list(range(0, weight_indices[-1] + 1, 2)):
        raise ValueError(f"Unsupported actor layer indices: {weight_indices}")

    modules: list[nn.Module] = []
    for position, index in enumerate(weight_indices):
        weight = actor_state[f"{index}.weight"]
        bias = actor_state[f"{index}.bias"]
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"Invalid actor layer {index} shapes: {weight.shape}, {bias.shape}")
        modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if position + 1 < len(weight_indices):
            modules.append(ACTIVATIONS[activation]())

    actor = nn.Sequential(*modules)
    actor.load_state_dict(actor_state, strict=True)
    return actor.eval()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")
    if any(key.startswith("actor_obs_normalizer.") for key in state_dict):
        raise ValueError("Empirical-normalization checkpoints require the official Isaac Lab exporter")

    actor = build_actor(state_dict, args.activation)
    input_dim = actor[0].in_features
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        torch.zeros(1, input_dim),
        args.output,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"Exported {args.checkpoint} -> {args.output} ({input_dim} observations)")


if __name__ == "__main__":
    main()
