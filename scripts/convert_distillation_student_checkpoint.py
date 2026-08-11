#!/usr/bin/env python3
"""Convert a distillation checkpoint's student into a PPO actor checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_checkpoint", type=Path)
    parser.add_argument("ppo_template", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student = torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
    template = torch.load(args.ppo_template, map_location="cpu", weights_only=True)
    student_state = student.get("model_state_dict")
    template_state = template.get("model_state_dict")
    if not isinstance(student_state, dict) or not isinstance(template_state, dict):
        raise ValueError("Both checkpoints must contain model_state_dict")

    student_actor = {
        f"actor.{key.removeprefix('student.')}" : value
        for key, value in student_state.items()
        if key.startswith("student.")
    }
    template_actor = {key: value for key, value in template_state.items() if key.startswith("actor.")}
    if not student_actor or set(student_actor) != set(template_actor):
        raise ValueError("Student and PPO template actor layer sets do not match")
    for key, value in student_actor.items():
        if value.shape != template_actor[key].shape:
            raise ValueError(f"Actor shape mismatch for {key}: {value.shape} vs {template_actor[key].shape}")

    output_state = dict(template)
    output_state["model_state_dict"] = dict(template_state)
    output_state["model_state_dict"].update(student_actor)
    output_state["iter"] = 0
    output_state["infos"] = {
        "converted_from_distillation": str(args.student_checkpoint.resolve()),
        "ppo_template": str(args.ppo_template.resolve()),
        "optimizer_reset_required": True,
    }
    output_state.pop("optimizer_state_dict", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_state, args.output)
    print(f"Converted student {args.student_checkpoint} -> PPO actor {args.output}")


if __name__ == "__main__":
    main()
