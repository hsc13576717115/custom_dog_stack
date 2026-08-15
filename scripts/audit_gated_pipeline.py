#!/usr/bin/env python3
"""Audit the complete gated locomotion pipeline from durable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


FOUNDATION_RUN = "2026-08-14_15-13-55_omni_trot_closed_loop_foundation_seed42"
FOUNDATION_CHECKPOINTS = (625, 775, 875, 900, 975, 999)


def read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def training_asset_path(selection_path: Path) -> str | None:
    env_path = selection_path.parent.parent / "params/env.yaml"
    try:
        # RSL-RL snapshots include Python tuple tags.  BaseLoader is sufficient
        # here because the audit needs one scalar path and must not construct
        # arbitrary Python objects from a training artifact.
        value = yaml.load(env_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        asset_path = value["scene"]["robot"]["spawn"]["asset_path"]
    except (FileNotFoundError, OSError, TypeError, KeyError, yaml.YAMLError):
        return None
    return str(asset_path) if asset_path is not None else None


def latest(paths) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def record(status: str, evidence: Path | None = None, **details) -> dict[str, object]:
    value: dict[str, object] = {"status": status}
    if evidence is not None:
        value["evidence"] = str(evidence.resolve())
    value.update(details)
    return value


def accepted_selection(
    log_root: Path,
    pattern: str,
    *,
    prefix: str | None = None,
    required_asset: str | None = None,
) -> dict[str, object]:
    path = latest(log_root.glob(pattern))
    if path is None:
        return record("pending")
    result = read_json(path)
    if result is None:
        return record("invalid", path)
    selected = result.get("selected")
    prefix_ok = prefix is None or (
        isinstance(selected, str) and selected.startswith(prefix)
    )
    accepted = result.get("accepted") is True and prefix_ok
    asset_path = training_asset_path(path) if required_asset is not None else None
    asset_ok = required_asset is None or (
        asset_path is not None and Path(asset_path).name == required_asset
    )
    if accepted and not asset_ok:
        status = "stale_asset"
    else:
        status = "complete" if accepted else "failed"
    return record(
        status,
        path,
        accepted=accepted and asset_ok,
        selected=selected,
        selected_candidate=result.get("selected_candidate"),
        **({"training_asset_path": asset_path} if required_asset is not None else {}),
    )


def max_checkpoint(run_dir: Path) -> int | None:
    iterations = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def initial_foundation(log_root: Path) -> dict[str, object]:
    run = log_root / FOUNDATION_RUN
    evaluation = run / "evaluation"
    missing = []
    for iteration in FOUNDATION_CHECKPOINTS:
        candidate = evaluation / "candidates" / f"model_{iteration}"
        for relative in ("exported/policy.onnx", "params/deploy.yaml"):
            if not (candidate / relative).is_file():
                missing.append(f"model_{iteration}/{relative}")
    grid_path = evaluation / "stage_a_grid_10s.json"
    grid = read_json(grid_path)
    counts = {f"model_{iteration}": 0 for iteration in FOUNDATION_CHECKPOINTS}
    if grid is not None and isinstance(grid.get("rows"), list):
        for row in grid["rows"]:
            if isinstance(row, dict) and row.get("candidate") in counts:
                counts[str(row["candidate"])] += 1
    complete = not missing and all(count == 15 for count in counts.values())
    return record(
        "complete" if complete else "incomplete",
        grid_path if grid_path.is_file() else None,
        exported_checkpoints=list(FOUNDATION_CHECKPOINTS),
        command_counts=counts,
        missing=missing,
    )


def second_seed(log_root: Path) -> dict[str, object]:
    runs = list(log_root.glob("*_closed_loop_robust_foundation_seed73"))
    if not runs:
        return record("pending")
    run = max(runs, key=lambda path: path.stat().st_mtime)
    iteration = max_checkpoint(run)
    selection = run / "evaluation/routed_robust_foundation_selection.json"
    if iteration is None or iteration < 999:
        return record("training", run, latest_checkpoint=iteration, target_checkpoint=999)
    result = read_json(selection)
    if result is None:
        return record("evaluating", run, latest_checkpoint=iteration)
    selected = result.get("selected")
    accepted = result.get("accepted") is True and isinstance(selected, str) and selected.startswith("RFR_")
    return record(
        "complete" if accepted else "failed",
        selection,
        accepted=accepted,
        selected=selected,
        selected_candidate=result.get("selected_candidate"),
    )


def collision_runtime(project_root: Path) -> dict[str, object]:
    report = project_root / "reports/selective_collision_isaac_runtime.json"
    result = read_json(report)
    if result is None:
        return record("pending")
    implementation_files = (
        project_root / "rl/src/custom_dog_rl/assets/custom_dog.py",
        project_root / "rl/src/custom_dog_rl/assets/collision_contract.py",
        project_root / "rl/scripts/validate_selective_self_collision.py",
        project_root
        / "ros2/src/custom_dog_description/urdf/custom_dog_selective_collision.urdf",
        project_root / "scripts/generate_selective_collision_urdf.py",
    )
    if report.stat().st_mtime < max(path.stat().st_mtime for path in implementation_files):
        return record("pending_revalidation", report)
    selective_asset = implementation_files[3].resolve()
    asset_hash = hashlib.sha256(selective_asset.read_bytes()).hexdigest()
    passed = (
        result.get("filtered_pair_count") == result.get("expected_filtered_pair_count")
        and result.get("nominal_pose_held") is True
        and result.get("nominal_nonfoot_contact_steps") == 0
        and isinstance(result.get("forced_cross_leg_contact_steps"), int)
        and int(result["forced_cross_leg_contact_steps"]) > 0
        and result.get("asset_path") == str(selective_asset)
        and result.get("asset_sha256") == asset_hash
    )
    return record("complete" if passed else "failed", report, accepted=passed)


def keyboard_review(project_root: Path) -> dict[str, object]:
    path = project_root / "reports/keyboard_sim2sim_review.json"
    result = read_json(path)
    ranking_path = project_root / "reports/stage_a_six_checkpoint_ranking.json"
    ranking = read_json(ranking_path)
    ranked_candidates = []
    if ranking is not None and isinstance(ranking.get("top_candidates"), list):
        ranked_candidates = [
            str(candidate["label"])
            for candidate in ranking["top_candidates"][:3]
            if isinstance(candidate, dict) and isinstance(candidate.get("label"), str)
        ]
    expected = set(ranked_candidates or ("model_999", "model_975", "model_875"))
    if result is None:
        return record(
            "ready_for_user",
            ranking_path
            if ranking_path.is_file()
            else project_root
            / f"logs/rsl_rl/custom_dog_velocity/{FOUNDATION_RUN}/evaluation/stage_a_grid_10s.json",
            candidates=sorted(expected),
        )
    reviewed = result.get("reviewed_candidates")
    reviewed_set = set(reviewed) if isinstance(reviewed, list) else set()
    complete = (
        result.get("accepted") is True
        and len(reviewed_set & expected) >= 2
        and isinstance(result.get("notes"), str)
        and bool(str(result["notes"]).strip())
    )
    return record(
        "complete" if complete else "incomplete",
        path,
        accepted=result.get("accepted") is True,
        reviewed_candidates=sorted(reviewed_set),
    )


def audit(project_root: Path) -> dict[str, object]:
    log_root = project_root / "logs/rsl_rl/custom_dog_velocity"
    selective_asset = "custom_dog_selective_collision.urdf"
    stages = {
        "initial_six_checkpoint_grid": initial_foundation(log_root),
        "keyboard_sim2sim_review": keyboard_review(project_root),
        "routed_stage_a_seed42": accepted_selection(
            log_root, "*/evaluation/routed_stage_a_selection.json", prefix="ROUTED_"
        ),
        "routed_stage_a_seed73": second_seed(log_root),
        "selective_collision_runtime": collision_runtime(project_root),
        "selective_collision_stage_a": accepted_selection(
            log_root,
            "*_closed_loop_selective_collision_stage_a/evaluation/selective_collision_selection.json",
            prefix="SC_",
            required_asset=selective_asset,
        ),
    }
    for stage in ("b", "c", "d"):
        stages[f"closed_loop_stage_{stage}"] = accepted_selection(
            log_root,
            f"*_closed_loop_stage_{stage}_seed42/evaluation/closed_loop_stage_{stage}_selection.json",
            prefix=f"{stage.upper()}_",
            required_asset=selective_asset,
        )
    for stage in ("r0", "r1", "r2"):
        stages[f"self_righting_{stage}"] = accepted_selection(
            log_root,
            f"*_self_righting_{stage}_seed42/evaluation/self_righting_{stage}_selection.json",
            required_asset=selective_asset,
        )
    stages["gait_robust"] = accepted_selection(
        log_root,
        "*_closed_loop_gait_robust_seed42/evaluation/gait_robust_selection.json",
        prefix="GR_",
        required_asset=selective_asset,
    )
    stages["dynamics_teacher"] = accepted_selection(
        log_root,
        "*_dynamics_teacher_seed42/evaluation/dynamics_teacher_selection.json",
        prefix="DYN_",
        required_asset=selective_asset,
    )
    for stage in ("t0", "t1"):
        stages[f"terrain_{stage}"] = accepted_selection(
            log_root,
            f"*_terrain_{stage}_seed42/evaluation/terrain_{stage}_selection.json",
            prefix=f"{stage.upper()}_",
            required_asset=selective_asset,
        )
    stages["history213_distillation"] = accepted_selection(
        log_root,
        "*_closed_loop_history213_final_seed42/evaluation/history213_selection.json",
        prefix="H213_",
        required_asset=selective_asset,
        )
    handoff_path = latest(log_root.glob("*_self_righting_r2_seed42/evaluation/recovery_handoff_r2.json"))
    handoff = read_json(handoff_path) if handoff_path is not None else None
    handoff_passed = handoff is not None and handoff.get("passes_all") is True
    stages["recovery_locomotion_handoff"] = record(
        "complete" if handoff_passed else "pending",
        handoff_path,
        accepted=handoff_passed,
    )

    required = list(stages)
    return {
        "all_complete": all(stages[name]["status"] == "complete" for name in required),
        "manual_review_ready": stages["keyboard_sim2sim_review"]["status"]
        == "ready_for_user",
        "required_stages": required,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.project_root.resolve())
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    if args.require_complete and not result["all_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
