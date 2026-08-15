#!/usr/bin/env python3
"""Run the custom-dog position policy in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from pathlib import Path
from threading import Lock
from typing import TextIO

import mujoco
import numpy as np
import yaml


SDK_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

BASE_OBSERVATIONS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]
HIMLOCO_BASE_OBSERVATIONS = [
    "velocity_commands",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]
SUPPORTED_OBSERVATIONS = (
    BASE_OBSERVATIONS,
    BASE_OBSERVATIONS + ["gait_phase"],
    BASE_OBSERVATIONS + ["trot_clock"],
    BASE_OBSERVATIONS + ["base_lin_vel_xy"],
    BASE_OBSERVATIONS + ["trot_clock", "base_lin_vel_xy"],
    BASE_OBSERVATIONS + ["trot_clock", "base_lin_vel_xy", "dynamics_context"],
)

HOME_SDK = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5],
    dtype=np.float64,
)
PRONE_SDK = np.array(
    [0.0, math.radians(71.0), math.radians(-161.0)] * 4,
    dtype=np.float64,
)
PRONE_ROOT_HEIGHT = 0.07
RECOVERY_INITIAL_STATE_ROLL = {
    "recovery-belly": 0.0,
    "recovery-back": math.pi,
    "recovery-left": math.pi / 2.0,
    "recovery-right": -math.pi / 2.0,
}
RECOVERY_INITIAL_STATE_HEIGHT = {
    "recovery-belly": 0.09,
    "recovery-back": 0.15,
    "recovery-left": 0.14,
    "recovery-right": 0.14,
}

DIRECT_TORQUE_SPEED = (13.5, 30.0, 20.2, 23.4)
CALF_TORQUE_SPEED = (6.75, 15.0, 40.4, 46.8)


class DeploymentYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that accepts tuples emitted by HimLoco's exporter."""


DeploymentYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node),
)


def update_current_first_history(history: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Prepend one observation while retaining the newest available past frames."""

    history = np.asarray(history)
    current = np.asarray(current)
    if history.ndim != 1 or current.ndim != 1 or history.size < current.size:
        raise ValueError("History and current observation must be compatible flat vectors")
    if history.size % current.size != 0:
        raise ValueError("History width must be an integer multiple of one observation")
    return np.concatenate((current, history[:-current.size]))


def object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]


def vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{label} must contain {length} values, got shape {result.shape}")
    return result


def reset_to_prone_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the measured folded pose used by the recovery training task."""

    mujoco.mj_resetData(model, data)
    data.qpos[2] = PRONE_ROOT_HEIGHT
    data.qpos[3] = 1.0
    for value, joint_name in zip(PRONE_SDK, SDK_JOINT_ORDER):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MJCF is missing {joint_name}")
        data.qpos[model.jnt_qposadr[joint_id]] = value
    data.ctrl[:] = PRONE_SDK
    mujoco.mj_forward(model, data)


def reset_to_recovery_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_state: str,
    orientation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    joint_noise: float = 0.0,
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> None:
    """Reset directly into a folded recovery pose without a scripted stand-up ramp."""

    if initial_state not in RECOVERY_INITIAL_STATE_ROLL:
        raise ValueError(f"Unknown recovery initial state: {initial_state}")
    mujoco.mj_resetData(model, data)
    data.qpos[2] = RECOVERY_INITIAL_STATE_HEIGHT[initial_state]
    half_roll = 0.5 * RECOVERY_INITIAL_STATE_ROLL[initial_state]
    canonical_quat = np.array(
        (math.cos(half_roll), math.sin(half_roll), 0.0, 0.0),
        dtype=np.float64,
    )
    roll, pitch, yaw = np.radians(np.asarray(orientation_offset_deg, dtype=np.float64))
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    offset_quat = np.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )
    orientation = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(orientation, canonical_quat, offset_quat)
    data.qpos[3:7] = orientation
    rng = np.random.default_rng(seed)
    for value, joint_name in zip(PRONE_SDK, SDK_JOINT_ORDER):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MJCF is missing {joint_name}")
        perturbation = rng.uniform(-joint_noise, joint_noise) if joint_noise > 0.0 else 0.0
        data.qpos[model.jnt_qposadr[joint_id]] = value + perturbation
    data.qvel[:3] = np.asarray(linear_velocity, dtype=np.float64)
    data.qvel[3:6] = np.asarray(angular_velocity, dtype=np.float64)
    data.ctrl[:] = PRONE_SDK
    mujoco.mj_forward(model, data)


def quintic_smoothstep(progress: float) -> float:
    """Blend from zero to one with zero velocity and acceleration at both ends."""

    value = float(np.clip(progress, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def trapezoid_integral(values: np.ndarray, times: np.ndarray) -> float:
    """Integrate samples across both NumPy 1.x and 2.x environments."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, times))
    return float(np.trapz(values, times))


def ground_foot_contacts(
    contacts: object,
    geom_body_ids: np.ndarray,
    foot_body_ids: np.ndarray,
) -> np.ndarray:
    """Return foot contacts against world-owned terrain, excluding robot self-contact."""

    grounded = np.zeros(len(foot_body_ids), dtype=bool)
    for contact in contacts:
        if contact.efc_address < 0:
            continue
        body_ids = (geom_body_ids[contact.geom1], geom_body_ids[contact.geom2])
        if 0 not in body_ids:
            continue
        other_body_id = body_ids[1] if body_ids[0] == 0 else body_ids[0]
        grounded |= foot_body_ids == other_body_id
    return grounded


def robot_self_contact_pair_count(contacts: object, geom_body_ids: np.ndarray) -> int:
    """Count active contacts whose geoms belong to different robot bodies."""

    count = 0
    for contact in contacts:
        if contact.efc_address < 0:
            continue
        body1 = int(geom_body_ids[contact.geom1])
        body2 = int(geom_body_ids[contact.geom2])
        if body1 != 0 and body2 != 0 and body1 != body2:
            count += 1
    return count


def illegal_ground_contact_pair_count(
    contacts: object,
    geom_body_ids: np.ndarray,
    foot_body_ids: np.ndarray,
) -> int:
    """Count active terrain contacts made by any non-foot robot body."""

    foot_ids = set(int(body_id) for body_id in foot_body_ids)
    count = 0
    for contact in contacts:
        if contact.efc_address < 0:
            continue
        body1 = int(geom_body_ids[contact.geom1])
        body2 = int(geom_body_ids[contact.geom2])
        if (body1 == 0) == (body2 == 0):
            continue
        robot_body = body2 if body1 == 0 else body1
        if robot_body not in foot_ids:
            count += 1
    return count


class PolicyController:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: Path,
        deploy_path: Path,
        command: np.ndarray,
        trace_path: Path | None = None,
        trace_limit: int = 500,
        explicit_pd: bool = False,
        encoder_path: Path | None = None,
    ) -> None:
        import onnxruntime as ort

        with deploy_path.open(encoding="utf-8") as stream:
            self.cfg = yaml.load(stream, Loader=DeploymentYamlLoader)

        self.observation_names = list(self.cfg["observations"])
        self.use_encoder = bool(self.cfg.get("use_encoder", False))
        if self.use_encoder:
            if self.observation_names != HIMLOCO_BASE_OBSERVATIONS:
                raise ValueError(
                    "HimLoco observation contract mismatch: expected "
                    f"{HIMLOCO_BASE_OBSERVATIONS}, got {self.observation_names}"
                )
            if encoder_path is None:
                raise ValueError("HimLoco deploy.yaml requires --encoder encoder.onnx")
        elif encoder_path is not None:
            raise ValueError("--encoder was supplied but deploy.yaml does not enable use_encoder")
        elif self.observation_names not in SUPPORTED_OBSERVATIONS:
            raise ValueError(
                "Observation contract mismatch: expected the 45-D base terms with an optional "
                "final gait_phase, trot_clock, base_lin_vel_xy or trot_clock+base_lin_vel_xy term, "
                f"got {self.observation_names}"
            )

        self.joint_map = np.asarray(self.cfg["joint_ids_map"], dtype=np.int32)
        if sorted(self.joint_map.tolist()) != list(range(12)):
            raise ValueError(f"joint_ids_map must be a permutation of 0..11: {self.joint_map.tolist()}")
        self.policy_joint_names = [SDK_JOINT_ORDER[index] for index in self.joint_map]

        step_ratio = float(self.cfg["step_dt"]) / model.opt.timestep
        self.decimation = int(round(step_ratio))
        if self.decimation < 1 or abs(step_ratio - self.decimation) > 1e-9:
            raise ValueError(
                f"Policy step_dt {self.cfg['step_dt']} is not an integer multiple of simulation dt {model.opt.timestep}"
            )

        action_cfg = self.cfg["actions"]["JointPositionAction"]
        self.default_position = vector(self.cfg["default_joint_pos"], 12, "default_joint_pos")
        self.action_scale = vector(action_cfg["scale"], 12, "action scale")
        self.action_offset = vector(action_cfg["offset"], 12, "action offset")
        if not np.allclose(self.default_position, self.action_offset):
            raise ValueError("default_joint_pos and JointPositionAction offset differ")
        self.sdk_default_position = np.empty(12, dtype=np.float64)
        self.sdk_default_position[self.joint_map] = self.default_position
        self.sdk_position_target = self.sdk_default_position.copy()
        self.explicit_pd = explicit_pd
        self.action_clip = np.asarray(action_cfg["clip"], dtype=np.float64)
        if self.action_clip.shape != (12, 2):
            raise ValueError(f"Action clip must have shape (12, 2), got {self.action_clip.shape}")

        # Optional sim2sim calibration, expressed in policy joint order.  The
        # bias is applied after inference so the raw action remains the value
        # fed back through the last_action observation term.
        bias_cfg = self.cfg.get("joint_target_bias")
        self.constant_target_bias_policy = vector(
            self.cfg.get("constant_joint_target_bias", [0.0] * 12),
            12,
            "constant_joint_target_bias",
        )
        if not np.isfinite(self.constant_target_bias_policy).all():
            raise ValueError("constant_joint_target_bias must be finite")
        self.target_bias_policy = np.zeros(12, dtype=np.float64)
        self.bias_vx_min = 0.0
        self.bias_vx_max = 0.0
        if bias_cfg is not None:
            self.target_bias_policy = vector(
                bias_cfg["values"], 12, "joint_target_bias.values"
            )
            vx_range = vector(bias_cfg["vx_range"], 2, "joint_target_bias.vx_range")
            self.bias_vx_min, self.bias_vx_max = (float(vx_range[0]), float(vx_range[1]))
            if not np.isfinite(vx_range).all() or self.bias_vx_max <= self.bias_vx_min:
                raise ValueError("joint_target_bias.vx_range must be strictly increasing")
            if not np.isfinite(self.target_bias_policy).all():
                raise ValueError("joint_target_bias.values must be finite")

        command_cfg = self.cfg["commands"]["base_velocity"]
        ranges = command_cfg["ranges"]
        external_ranges = command_cfg.get("external_ranges", ranges)
        self.command_ranges = np.asarray(
            [
                external_ranges["lin_vel_x"],
                external_ranges["lin_vel_y"],
                external_ranges["ang_vel_z"],
            ],
            dtype=np.float64,
        )
        if (
            self.command_ranges.shape != (3, 2)
            or not np.isfinite(self.command_ranges).all()
            or np.any(self.command_ranges[:, 0] > self.command_ranges[:, 1])
        ):
            raise ValueError("Velocity command ranges must be finite lower/upper pairs")
        policy_ranges = command_cfg.get("policy_ranges", ranges)
        self.policy_command_ranges = np.asarray(
            [
                policy_ranges["lin_vel_x"],
                policy_ranges["lin_vel_y"],
                policy_ranges["ang_vel_z"],
            ],
            dtype=np.float64,
        )
        if (
            self.policy_command_ranges.shape != (3, 2)
            or not np.isfinite(self.policy_command_ranges).all()
            or np.any(self.policy_command_ranges[:, 0] > self.policy_command_ranges[:, 1])
        ):
            raise ValueError("Policy command ranges must be finite lower/upper pairs")
        self.command = np.zeros(3, dtype=np.float64)
        self.policy_command = np.zeros(3, dtype=np.float64)
        self.set_command(command)

        joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.policy_joint_names
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise ValueError("Could not resolve all policy joints in MJCF")
        self.qpos_addresses = model.jnt_qposadr[joint_ids].copy()
        self.dof_addresses = model.jnt_dofadr[joint_ids].copy()
        sdk_joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in SDK_JOINT_ORDER],
            dtype=np.int32,
        )
        self.sdk_qpos_addresses = model.jnt_qposadr[sdk_joint_ids].copy()
        self.sdk_dof_addresses = model.jnt_dofadr[sdk_joint_ids].copy()

        # unitree_rl_lab exports stiffness/damping in canonical SDK order.
        # Only policy-order joint targets pass through joint_ids_map.
        self.sdk_stiffness = vector(self.cfg["stiffness"], 12, "stiffness")
        self.sdk_damping = vector(self.cfg["damping"], 12, "damping")
        if np.any(self.sdk_stiffness <= 0.0) or np.any(self.sdk_damping < 0.0):
            raise ValueError("stiffness must be positive and damping must be non-negative")

        # The generated MJCF contains position actuators, but their XML gains
        # are only generation-time defaults.  The frozen candidate YAML is the
        # deployment contract and may define per-joint gains, so apply those
        # gains to the live MuJoCo model as well as the torque-speed limiter.
        if (
            np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
        ):
            raise ValueError("Policy sim2sim requires fixed-gain joint position actuators")
        model.actuator_gainprm[:, 0] = self.sdk_stiffness
        model.actuator_biasprm[:, 1] = -self.sdk_stiffness
        model.actuator_biasprm[:, 2] = -self.sdk_damping
        if self.explicit_pd:
            # A fixed-gain position actuator becomes a direct effort actuator
            # when its gain is one and its affine position/velocity bias is zero.
            model.actuator_gainprm[:, 0] = 1.0
            model.actuator_biasprm[:, 1:3] = 0.0
            model.actuator_ctrllimited[:] = 0

        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "base_gyro")
        if self.base_id < 0 or self.gyro_id < 0:
            raise ValueError("MJCF must contain base body and base_gyro sensor")
        self.gyro_address = model.sensor_adr[self.gyro_id]

        self.session = ort.InferenceSession(str(policy_path.resolve()), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError("Policy must have exactly one input and at least one output")
        self.observation_history: dict[str, list[np.ndarray]] = {}
        self.encoder_session = None
        self.himloco_history: np.ndarray | None = None
        self.single_observation_dim = sum(
            len(term["scale"]) for term in self.cfg["observations"].values()
        )
        if self.use_encoder:
            self.history_frames = int(self.cfg.get("history_length", 0))
            if self.history_frames < 1:
                raise ValueError("HimLoco history_length must be positive")
            self.observation_dim = self.single_observation_dim * self.history_frames
            self.encoder_session = ort.InferenceSession(
                str(encoder_path.resolve()), providers=["CPUExecutionProvider"]
            )
            encoder_inputs = self.encoder_session.get_inputs()
            encoder_outputs = self.encoder_session.get_outputs()
            if len(encoder_inputs) != 1 or len(encoder_outputs) != 1:
                raise ValueError("HimLoco encoder must have exactly one input and one output")
            if (
                encoder_inputs[0].shape[-1] != self.observation_dim
                or encoder_outputs[0].shape[-1] != 19
                or inputs[0].shape[-1] != self.single_observation_dim + 19
                or outputs[0].shape[-1] != 12
            ):
                raise ValueError(
                    "Expected HimLoco ONNX shapes "
                    f"(*,{self.observation_dim})->(*,19)->(*,{self.single_observation_dim + 19})->(*,12), "
                    f"got {encoder_inputs[0].shape}->{encoder_outputs[0].shape} and "
                    f"{inputs[0].shape}->{outputs[0].shape}"
                )
            self.encoder_input_name = encoder_inputs[0].name
            self.encoder_output_name = encoder_outputs[0].name
            self.himloco_history = np.zeros(self.observation_dim, dtype=np.float32)
        else:
            self.history_frames = 1
            self.observation_dim = 0
            for name, term in self.cfg["observations"].items():
                history_length = int(term.get("history_length", 1))
                if history_length < 1:
                    raise ValueError(f"history_length must be positive, got {name}={history_length}")
                self.observation_dim += len(term["scale"]) * history_length
            if inputs[0].shape[-1] != self.observation_dim or outputs[0].shape[-1] != 12:
                raise ValueError(
                    f"Expected ONNX shapes (*,{self.observation_dim})->(*,12), "
                    f"got {inputs[0].shape}->{outputs[0].shape}"
                )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.current_action = np.zeros(12, dtype=np.float64)
        self.inference_times_ms: list[float] = []
        self.policy_steps = 0
        self.phase_steps = 0
        self.trace_limit = trace_limit
        self.trace_stream: TextIO | None = None
        self.trace_writer: csv.writer | None = None
        if trace_path is not None:
            trace_path = trace_path.resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_stream = trace_path.open("w", encoding="utf-8", newline="")
            self.trace_writer = csv.writer(self.trace_stream)
            header = ["step", "time_s"]
            for prefix, count in (
                ("obs", self.observation_dim),
                ("action", 12),
                ("target_q", 12),
                ("joint_q", 12),
                ("joint_dq", 12),
                ("ang_vel", 3),
                ("projected_gravity", 3),
                ("base_lin_vel", 3),
                ("base_pos", 3),
                ("base_quat", 4),
            ):
                header.extend(f"{prefix}_{index}" for index in range(count))
            self.trace_writer.writerow(header)

    def target_bias(self) -> np.ndarray:
        """Return the forward-command-dependent target calibration."""
        if self.bias_vx_max <= self.bias_vx_min:
            return self.constant_target_bias_policy.copy()
        blend = np.clip(
            (float(self.command[0]) - self.bias_vx_min)
            / (self.bias_vx_max - self.bias_vx_min),
            0.0,
            1.0,
        )
        return self.constant_target_bias_policy + blend * self.target_bias_policy

    def set_command(self, command: np.ndarray) -> None:
        """Set the external command and recalculate its policy calibration."""

        requested_command = vector(command, 3, "requested velocity command")
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        for index, name in enumerate(names):
            lower, upper = self.command_ranges[index]
            if not lower <= requested_command[index] <= upper:
                raise ValueError(
                    f"Command {name}={requested_command[index]} is outside [{lower}, {upper}]"
                )

        self.command = requested_command.copy()
        self.policy_command = requested_command.copy()
        calibration_root = self.cfg.get("command_calibration") or {}
        if not isinstance(calibration_root, dict):
            raise ValueError("command_calibration must be a mapping")
        for index, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            calibration_cfg = calibration_root.get(name)
            if calibration_cfg is None:
                continue
            requested = np.asarray(calibration_cfg["requested"], dtype=np.float64)
            policy = np.asarray(calibration_cfg["policy"], dtype=np.float64)
            if (
                requested.ndim != 1
                or requested.size < 2
                or policy.shape != requested.shape
                or not np.isfinite(requested).all()
                or not np.isfinite(policy).all()
                or np.any(np.diff(requested) <= 0.0)
                or np.any(np.diff(policy) < 0.0)
            ):
                raise ValueError(f"command_calibration.{name} must contain finite monotonic arrays")
            zero_indices = np.flatnonzero(np.isclose(requested, 0.0, atol=1.0e-12))
            if zero_indices.size != 1 or not np.isclose(policy[zero_indices[0]], 0.0, atol=1.0e-12):
                raise ValueError(f"command_calibration.{name} must map zero request to zero")
            lower, upper = self.command_ranges[index]
            if requested[0] > lower or requested[-1] < upper:
                raise ValueError(f"command calibration must cover the exported {name} range")
            policy_lower, policy_upper = self.policy_command_ranges[index]
            if np.any(policy < policy_lower) or np.any(policy > policy_upper):
                raise ValueError(f"calibrated policy commands must remain inside {name} range")
            self.policy_command[index] = np.interp(self.command[index], requested, policy)

    def reset_history(self, previous_action: np.ndarray | None = None) -> None:
        seed_action = (
            np.zeros(12, dtype=np.float64)
            if previous_action is None
            else vector(previous_action, 12, "previous action")
        )
        self.previous_action = seed_action.copy()
        self.current_action = seed_action.copy()
        self.observation_history.clear()
        if self.use_encoder:
            self.himloco_history = np.zeros(self.observation_dim, dtype=np.float32)
        self.phase_steps = 0

    def _scaled_term(self, name: str, values: np.ndarray) -> np.ndarray:
        term = self.cfg["observations"][name]
        scale = vector(term["scale"], len(values), f"{name} scale")
        clip = term.get("clip")
        if clip is None:
            return values * scale
        lower, upper = clip
        # Isaac Lab and unitree_rl_lab clip the raw observation before scale.
        return np.clip(values, lower, upper) * scale

    def _history_term(self, name: str, current: np.ndarray) -> np.ndarray:
        """Match Isaac Lab's per-term oldest-to-newest history flattening."""

        history_length = int(self.cfg["observations"][name].get("history_length", 1))
        history = self.observation_history.get(name)
        if history is None:
            history = [current.copy() for _ in range(history_length)]
            self.observation_history[name] = history
        else:
            history.append(current.copy())
            del history[:-history_length]
        return np.concatenate(history)

    def observation(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        angular_velocity = data.sensordata[self.gyro_address : self.gyro_address + 3].copy()
        rotation_world_from_base = data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
        base_linear_velocity = rotation_world_from_base.T @ data.qvel[:3]
        joint_position = data.qpos[self.qpos_addresses]
        joint_velocity = data.qvel[self.dof_addresses]

        raw_terms = {
            "base_ang_vel": angular_velocity,
            "projected_gravity": projected_gravity,
            "velocity_commands": self.policy_command,
            "joint_pos_rel": joint_position - self.default_position,
            "joint_vel_rel": joint_velocity,
            "last_action": self.previous_action,
            "base_lin_vel_xy": base_linear_velocity[:2],
        }
        if "dynamics_context" in self.observation_names:
            # The privileged teacher is validated at nominal MuJoCo physics.
            # Its normalized nominal context, including zero command delay,
            # is exactly zero.  This policy is never a deployment artifact.
            raw_terms["dynamics_context"] = np.zeros(
                len(self.cfg["observations"]["dynamics_context"]["scale"]),
                dtype=np.float64,
            )
        if "gait_phase" in self.observation_names:
            phase_cfg = self.cfg["observations"]["gait_phase"]["params"]
            period = float(phase_cfg["period"])
            command_threshold = float(phase_cfg.get("command_threshold", 0.1))
            if period <= 0.0 or command_threshold < 0.0:
                raise ValueError("gait_phase period must be positive and threshold non-negative")
            phase = (self.phase_steps * float(self.cfg["step_dt"])) % period / period
            gait_phase = np.array(
                [math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)],
                dtype=np.float64,
            )
            if np.linalg.norm(self.policy_command) <= command_threshold:
                gait_phase.fill(0.0)
            raw_terms["gait_phase"] = gait_phase
        if "trot_clock" in self.observation_names:
            clock_cfg = self.cfg["observations"]["trot_clock"]["params"]
            command_threshold = float(clock_cfg.get("command_threshold", 0.1))
            yaw_command_threshold = clock_cfg.get("yaw_command_threshold")
            if yaw_command_threshold is not None:
                yaw_command_threshold = float(yaw_command_threshold)
            min_frequency = float(clock_cfg.get("min_frequency", 1.4))
            max_frequency = float(clock_cfg.get("max_frequency", 3.2))
            full_speed = float(clock_cfg.get("full_speed", 3.0))
            yaw_speed_scale = float(clock_cfg.get("yaw_speed_scale", 0.35))
            if (
                command_threshold < 0.0
                or (yaw_command_threshold is not None and yaw_command_threshold < 0.0)
                or min_frequency <= 0.0
                or max_frequency < min_frequency
                or full_speed <= 0.0
                or yaw_speed_scale < 0.0
            ):
                raise ValueError("invalid trot_clock parameters")
            motion_speed = np.linalg.norm(self.policy_command[:2])
            motion_speed += yaw_speed_scale * abs(self.policy_command[2])
            blend = float(np.clip(motion_speed / full_speed, 0.0, 1.0))
            frequency = min_frequency + blend * (max_frequency - min_frequency)
            phase = (self.phase_steps * float(self.cfg["step_dt"]) * frequency) % 1.0
            foot_phase = (phase + np.array([0.0, 0.5, 0.5, 0.0])) % 1.0
            trot_clock = np.sin(2.0 * math.pi * foot_phase)
            if yaw_command_threshold is None:
                moving = motion_speed > command_threshold
            else:
                moving = (
                    np.linalg.norm(self.policy_command[:2]) > command_threshold
                    or abs(self.policy_command[2]) > yaw_command_threshold
                )
            if not moving:
                trot_clock.fill(0.0)
            raw_terms["trot_clock"] = trot_clock
        if self.use_encoder:
            current_observation = np.concatenate(
                [self._scaled_term(name, raw_terms[name]) for name in self.observation_names]
            ).astype(np.float32)
            self.himloco_history = update_current_first_history(
                self.himloco_history, current_observation
            ).astype(np.float32, copy=False)
            observation = self.himloco_history.copy()
        else:
            observation = np.concatenate(
                [
                    self._history_term(name, self._scaled_term(name, raw_terms[name]))
                    for name in self.observation_names
                ]
            ).astype(np.float32)
        if observation.shape != (self.observation_dim,) or not np.isfinite(observation).all():
            raise RuntimeError(f"Invalid policy observation: shape={observation.shape}")
        return observation

    def infer_action(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Run the policy without mutating actuator targets."""

        observation = self.observation(model, data)
        started = time.perf_counter()
        if self.use_encoder:
            encoder_output = self.encoder_session.run(
                [self.encoder_output_name],
                {self.encoder_input_name: observation[np.newaxis, :]},
            )[0]
            policy_input = np.concatenate(
                (observation[np.newaxis, : self.single_observation_dim], encoder_output), axis=1
            ).astype(np.float32, copy=False)
        else:
            policy_input = observation[np.newaxis, :]
        action = self.session.run(
            [self.output_name], {self.input_name: policy_input}
        )[0][0].astype(np.float64)
        self.inference_times_ms.append((time.perf_counter() - started) * 1000.0)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid policy action: shape={action.shape}")

        self.policy_steps += 1
        self.phase_steps += 1
        return observation, action

    def apply_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> None:
        """Apply a raw policy action, including any router transition blend."""

        target_policy_order = np.clip(
            self.action_offset + self.action_scale * action,
            self.action_clip[:, 0],
            self.action_clip[:, 1],
        )
        target_policy_order += self.target_bias()
        self.sdk_position_target[self.joint_map] = target_policy_order
        if not self.explicit_pd:
            data.ctrl[self.joint_map] = target_policy_order
        self.current_action = action.copy()
        self.previous_action = action.copy()

        if self.trace_writer is not None and (
            self.trace_limit == 0 or self.policy_steps <= self.trace_limit
        ):
            angular_velocity = data.sensordata[
                self.gyro_address : self.gyro_address + 3
            ].copy()
            rotation_world_from_base = data.xmat[self.base_id].reshape(3, 3)
            projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
            base_linear_velocity = rotation_world_from_base.T @ data.qvel[:3]
            row = [self.policy_steps, data.time]
            for values in (
                observation,
                action,
                target_policy_order,
                data.qpos[self.qpos_addresses],
                data.qvel[self.dof_addresses],
                angular_velocity,
                projected_gravity,
                base_linear_velocity,
                data.qpos[:3],
                data.qpos[3:7],
            ):
                row.extend(float(value) for value in values)
            self.trace_writer.writerow(row)
            self.trace_stream.flush()

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        observation, action = self.infer_action(model, data)
        self.apply_action(model, data, observation, action)

    def update_with_blend(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        start_action: np.ndarray,
        blend: float,
    ) -> None:
        """Infer locomotion while blending from a completed recovery action."""

        observation, action = self.infer_action(model, data)
        start = vector(start_action, 12, "recovery handoff action")
        action = (1.0 - float(np.clip(blend, 0.0, 1.0))) * start + float(
            np.clip(blend, 0.0, 1.0)
        ) * action
        self.apply_action(model, data, observation, action)

    def apply_actuator_limits(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Match the Unitree torque-speed clipping used during Isaac training."""
        joint_pos = data.qpos[self.sdk_qpos_addresses]
        joint_vel = data.qvel[self.sdk_dof_addresses]
        position_target = self.sdk_position_target if self.explicit_pd else data.ctrl
        desired_effort = self.sdk_stiffness * (position_target - joint_pos) - self.sdk_damping * joint_vel

        for index, joint_name in enumerate(SDK_JOINT_ORDER):
            x1, x2, y1, y2 = CALF_TORQUE_SPEED if "calf" in joint_name else DIRECT_TORQUE_SPEED
            full_torque = y1 if joint_vel[index] * desired_effort[index] > 0.0 else y2
            speed = abs(joint_vel[index])
            if speed < x1:
                limit = full_torque
            else:
                limit = max(0.0, full_torque * (x2 - speed) / (x2 - x1))
            model.actuator_forcerange[index] = (-limit, limit)
            if self.explicit_pd:
                data.ctrl[index] = np.clip(desired_effort[index], -limit, limit)

    def close(self) -> None:
        if self.trace_stream is not None:
            self.trace_stream.close()
            self.trace_stream = None
            self.trace_writer = None


class RoutedPolicyController:
    """Route exact-zero commands to a stand expert and motion to locomotion."""

    STAND = "stand"
    LOCOMOTION = "locomotion"

    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: Path,
        deploy_path: Path,
        stand_policy_path: Path,
        stand_deploy_path: Path,
        command: np.ndarray,
        trace_path: Path | None = None,
        trace_limit: int = 500,
        explicit_pd: bool = False,
        stand_enter_planar: float = 0.015,
        stand_enter_yaw: float = 0.025,
        stand_exit_planar: float = 0.025,
        stand_exit_yaw: float = 0.04,
        blend_seconds: float = 0.30,
    ) -> None:
        if not 0.0 <= stand_enter_planar <= stand_exit_planar:
            raise ValueError("stand planar thresholds must be non-negative and ordered")
        if not 0.0 <= stand_enter_yaw <= stand_exit_yaw:
            raise ValueError("stand yaw thresholds must be non-negative and ordered")
        if blend_seconds < 0.0:
            raise ValueError("router blend duration must be non-negative")

        self.locomotion = PolicyController(
            model,
            policy_path,
            deploy_path,
            command,
            trace_path,
            trace_limit,
            explicit_pd,
        )
        self.stand = PolicyController(
            model,
            stand_policy_path,
            stand_deploy_path,
            np.zeros(3, dtype=np.float64),
            None,
            trace_limit,
            explicit_pd,
        )
        self._validate_contracts()
        self.stand_enter_planar = stand_enter_planar
        self.stand_enter_yaw = stand_enter_yaw
        self.stand_exit_planar = stand_exit_planar
        self.stand_exit_yaw = stand_exit_yaw
        self.blend_steps = int(round(blend_seconds / float(self.locomotion.cfg["step_dt"])))
        self.mode = self._initial_mode(np.asarray(command, dtype=np.float64))
        self._blend_start_action = np.zeros(12, dtype=np.float64)
        self._blend_index = self.blend_steps
        self.current_action = np.zeros(12, dtype=np.float64)
        self.set_command(command, allow_transition=False)

    def _validate_contracts(self) -> None:
        left = self.locomotion
        right = self.stand
        for name in (
            "observation_names",
            "observation_dim",
            "decimation",
            "joint_map",
            "action_scale",
            "action_offset",
            "action_clip",
            "sdk_stiffness",
            "sdk_damping",
        ):
            left_value = getattr(left, name)
            right_value = getattr(right, name)
            equal = (
                left_value == right_value
                if isinstance(left_value, (int, list))
                else np.array_equal(left_value, right_value)
            )
            if not equal:
                raise ValueError(f"Stand/locomotion policy contract mismatch: {name}")

    def _initial_mode(self, command: np.ndarray) -> str:
        planar = float(np.linalg.norm(command[:2]))
        return (
            self.STAND
            if planar <= self.stand_enter_planar and abs(float(command[2])) <= self.stand_enter_yaw
            else self.LOCOMOTION
        )

    @property
    def active(self) -> PolicyController:
        return self.stand if self.mode == self.STAND else self.locomotion

    @property
    def command(self) -> np.ndarray:
        return self.locomotion.command

    @property
    def policy_command(self) -> np.ndarray:
        return self.locomotion.policy_command

    @property
    def command_ranges(self) -> np.ndarray:
        return self.locomotion.command_ranges

    @property
    def decimation(self) -> int:
        return self.locomotion.decimation

    @property
    def sdk_default_position(self) -> np.ndarray:
        return self.locomotion.sdk_default_position

    @property
    def gyro_address(self) -> int:
        return self.locomotion.gyro_address

    @property
    def inference_times_ms(self) -> list[float]:
        return self.locomotion.inference_times_ms + self.stand.inference_times_ms

    def set_command(self, command: np.ndarray, allow_transition: bool = True) -> None:
        self.locomotion.set_command(command)
        self.stand.set_command(np.zeros(3, dtype=np.float64))
        planar = float(np.linalg.norm(self.command[:2]))
        requested_mode = self.mode
        if self.mode == self.LOCOMOTION:
            if planar <= self.stand_enter_planar and abs(float(self.command[2])) <= self.stand_enter_yaw:
                requested_mode = self.STAND
        elif planar >= self.stand_exit_planar or abs(float(self.command[2])) >= self.stand_exit_yaw:
            requested_mode = self.LOCOMOTION
        if requested_mode != self.mode:
            self.mode = requested_mode
            self._blend_start_action = self.current_action.copy()
            self._blend_index = 0 if allow_transition else self.blend_steps
            self.active.reset_history(self.current_action)

    def reset_history(self) -> None:
        self.locomotion.reset_history(self.current_action)
        self.stand.reset_history(self.current_action)
        self._blend_start_action = self.current_action.copy()
        self._blend_index = self.blend_steps

    def begin_external_handoff(self, start_action: np.ndarray) -> None:
        """Start a smooth route handoff from an external recovery actor."""

        self.current_action = vector(start_action, 12, "recovery handoff action").copy()
        self._blend_start_action = self.current_action.copy()
        self._blend_index = 0
        self.active.reset_history(self.current_action)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        observation, requested_action = self.active.infer_action(model, data)
        if self._blend_index < self.blend_steps and self.blend_steps > 0:
            self._blend_index += 1
            blend = quintic_smoothstep(self._blend_index / self.blend_steps)
            action = (1.0 - blend) * self._blend_start_action + blend * requested_action
        else:
            action = requested_action
        self.active.apply_action(model, data, observation, action)
        self.current_action = action.copy()

    def apply_actuator_limits(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.active.apply_actuator_limits(model, data)

    def close(self) -> None:
        self.locomotion.close()
        self.stand.close()


class InteractiveControls:
    """Thread-safe keyboard state for the passive MuJoCo viewer."""

    PASSIVE = "passive"
    FIX_STAND = "fix_stand"
    POLICY_HOLD = "policy_hold"
    VELOCITY = "velocity"

    def __init__(
        self,
        policy_controller: PolicyController | None,
        initial_mode: str = FIX_STAND,
    ) -> None:
        self.policy_controller = policy_controller
        if initial_mode not in (
            self.PASSIVE,
            self.FIX_STAND,
            self.POLICY_HOLD,
            self.VELOCITY,
        ):
            raise ValueError(f"Unknown initial interactive mode: {initial_mode}")
        if initial_mode == self.VELOCITY and policy_controller is None:
            raise ValueError("Velocity mode requires a policy controller")
        self._mode = initial_mode
        self._command = (
            policy_controller.command.copy()
            if policy_controller is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._revision = 0
        self._lock = Lock()

    def snapshot(self) -> tuple[str, np.ndarray, int]:
        with self._lock:
            return self._mode, self._command.copy(), self._revision

    def _set_mode(self, mode: str) -> str:
        if mode in (self.PASSIVE, self.FIX_STAND):
            self._command.fill(0.0)
        self._mode = mode
        self._revision += 1
        return f"interactive mode={mode}, command={self._command.tolist()}"

    def complete_fix_stand(self) -> str | None:
        """Hand a completed stand ramp to the zero-command policy exactly once."""

        with self._lock:
            if self._mode != self.FIX_STAND:
                return None
            return self._set_mode(self.POLICY_HOLD)

    def _change_command(self, index: int, delta: float) -> str:
        if self.policy_controller is None:
            return "Velocity commands require --policy and --deploy-yaml"
        if self._mode == self.POLICY_HOLD:
            self._set_mode(self.VELOCITY)
        elif self._mode != self.VELOCITY:
            return "interactive command ignored; use R to stand, then enter a command"
        lower, upper = self.policy_controller.command_ranges[index]
        previous = self._command[index]
        self._command[index] = np.clip(previous + delta, lower, upper)
        self._revision += 1
        labels = ("vx", "vy", "yaw")
        return f"interactive {labels[index]}={self._command[index]:.3f}"

    def key_callback(self, key: int) -> None:
        glfw = mujoco.glfw.glfw
        message = None
        with self._lock:
            # MuJoCo reserves Space and 0..5 for simulation/visibility controls.
            if key == glfw.KEY_P:
                message = self._set_mode(self.PASSIVE)
            elif key == glfw.KEY_R:
                message = self._set_mode(self.FIX_STAND)
            elif key == glfw.KEY_V:
                if self.policy_controller is None:
                    message = "Velocity mode requires --policy and --deploy-yaml"
                else:
                    message = self._set_mode(self.VELOCITY)
            elif key == glfw.KEY_W:
                message = self._change_command(0, 0.1)
            elif key == glfw.KEY_S:
                message = self._change_command(0, -0.1)
            elif key == glfw.KEY_A:
                message = self._change_command(1, 0.1)
            elif key == glfw.KEY_D:
                message = self._change_command(1, -0.1)
            elif key == glfw.KEY_Q:
                message = self._change_command(2, 0.1)
            elif key == glfw.KEY_E:
                message = self._change_command(2, -0.1)
            elif key == glfw.KEY_X:
                message = self._set_mode(self.FIX_STAND)
        if message is not None:
            print(message, flush=True)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=model_dir / "custom_dog.xml")
    parser.add_argument("--policy", type=Path, help="Exported policy.onnx; omit for position-hold test")
    parser.add_argument("--encoder", type=Path, help="HimLoco encoder.onnx paired with --policy")
    parser.add_argument("--deploy-yaml", type=Path, help="params/deploy.yaml from the same training run")
    parser.add_argument("--stand-policy", type=Path, help="Optional zero-command stand expert ONNX")
    parser.add_argument(
        "--stand-deploy-yaml",
        type=Path,
        help="params/deploy.yaml exported with --stand-policy",
    )
    parser.add_argument("--stand-enter-planar", type=float, default=0.015)
    parser.add_argument("--stand-enter-yaw", type=float, default=0.025)
    parser.add_argument("--stand-exit-planar", type=float, default=0.025)
    parser.add_argument("--stand-exit-yaw", type=float, default=0.04)
    parser.add_argument("--policy-blend", type=float, default=0.30, help="Expert handoff duration in seconds")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"), default=(0, 0, 0))
    parser.add_argument(
        "--command-step",
        nargs=4,
        type=float,
        action="append",
        metavar=("TIME", "VX", "VY", "YAW"),
        help="Change the requested command at simulation TIME; may be repeated",
    )
    parser.add_argument(
        "--initial-state",
        choices=("home", "prone", *RECOVERY_INITIAL_STATE_ROLL),
        default="home",
        help=(
            "Start from home, the legacy scripted prone transition, or a folded direct "
            "self-righting pose (recovery-belly/back/left/right)"
        ),
    )
    parser.add_argument(
        "--recovery-orientation-offset-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Euler offset applied to a direct recovery initial state",
    )
    parser.add_argument("--recovery-joint-noise", type=float, default=0.0)
    parser.add_argument(
        "--recovery-linear-velocity",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
    )
    parser.add_argument(
        "--recovery-angular-velocity",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("WX", "WY", "WZ"),
    )
    parser.add_argument("--recovery-seed", type=int, default=0)
    parser.add_argument(
        "--recovery-ramp",
        type=float,
        default=2.0,
        help="Seconds used to smoothly move from the measured prone pose to HOME_SDK",
    )
    parser.add_argument(
        "--recovery-hold",
        type=float,
        default=1.0,
        help="Seconds of zero policy command after the recovery ramp before releasing --command",
    )
    parser.add_argument(
        "--recovery-policy",
        type=Path,
        help="Optional exported self-righting ONNX policy for direct fall recovery",
    )
    parser.add_argument(
        "--recovery-deploy-yaml",
        type=Path,
        help="deploy.yaml paired with --recovery-policy",
    )
    parser.add_argument(
        "--recovery-upright-dwell",
        type=float,
        default=0.4,
        help="Stable upright dwell before recovery hands off to locomotion",
    )
    parser.add_argument(
        "--recovery-locomotion-hold",
        type=float,
        default=1.0,
        help="Zero-command locomotion hold after recovery handoff",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds excluded from tracking metrics")
    parser.add_argument("--trace", type=Path, help="Write policy observations and actions to CSV")
    parser.add_argument(
        "--explicit-pd",
        action="store_true",
        help="Diagnostic mode: apply position PD as explicit effort every physics step.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=500,
        help="Maximum policy rows to trace; use 0 for unlimited (default: 500)",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--camera-mode",
        choices=("tracking", "free"),
        default="tracking",
        help="Use a camera that follows the base body or MuJoCo's free camera",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable keyboard Passive/FixStand/Velocity control in the MuJoCo viewer",
    )
    parser.add_argument("--realtime", action="store_true", help="Pace headless simulation at wall-clock speed")
    args = parser.parse_args()
    if (args.policy is None) != (args.deploy_yaml is None):
        parser.error("--policy and --deploy-yaml must be supplied together")
    if args.encoder is not None and args.policy is None:
        parser.error("--encoder requires --policy and --deploy-yaml")
    if args.encoder is not None and args.stand_policy is not None:
        parser.error("HimLoco --encoder cannot currently be combined with --stand-policy")
    if (args.stand_policy is None) != (args.stand_deploy_yaml is None):
        parser.error("--stand-policy and --stand-deploy-yaml must be supplied together")
    if (args.recovery_policy is None) != (args.recovery_deploy_yaml is None):
        parser.error("--recovery-policy and --recovery-deploy-yaml must be supplied together")
    if args.stand_policy is not None and args.policy is None:
        parser.error("--stand-policy requires --policy and --deploy-yaml")
    if args.explicit_pd and args.policy is None:
        parser.error("--explicit-pd requires --policy and --deploy-yaml")
    if args.explicit_pd and (args.initial_state != "home" or args.interactive):
        parser.error("--explicit-pd diagnostic currently requires a non-interactive home start")
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    if args.duration == 0 and not args.interactive:
        parser.error("--duration 0 is only valid with --interactive")
    if args.interactive and not args.viewer:
        parser.error("--interactive requires --viewer")
    if args.interactive and args.trace is not None:
        parser.error("--trace is only supported for fixed-command runs")
    if args.interactive and args.command_step:
        parser.error("--command-step cannot be combined with --interactive")
    if args.recovery_policy is not None and args.interactive:
        parser.error("--recovery-policy currently requires a fixed-command (non-interactive) run")
    if args.recovery_policy is not None and args.policy is None:
        parser.error("--recovery-policy requires the locomotion --policy and --deploy-yaml")
    if args.recovery_policy is not None and args.initial_state == "home":
        parser.error("--recovery-policy requires --initial-state recovery-*")
    if args.command_step and args.policy is None:
        parser.error("--command-step requires --policy and --deploy-yaml")
    if args.command_step and args.initial_state != "home":
        parser.error("--command-step currently requires --initial-state home")
    if args.command_step:
        step_times = [step[0] for step in args.command_step]
        if step_times[0] != 0.0 or any(
            right <= left for left, right in zip(step_times, step_times[1:])
        ):
            parser.error("--command-step times must start at 0 and be strictly increasing")
        if args.duration > 0 and step_times[-1] >= args.duration:
            parser.error("the final --command-step time must be shorter than --duration")
    if args.recovery_ramp < 0.0 or args.recovery_hold < 0.0:
        parser.error("--recovery-ramp and --recovery-hold must be non-negative")
    if args.recovery_joint_noise < 0.0:
        parser.error("--recovery-joint-noise must be non-negative")
    if args.recovery_upright_dwell <= 0.0 or args.recovery_locomotion_hold < 0.0:
        parser.error("recovery upright dwell must be positive and locomotion hold non-negative")
    if args.warmup < 0 or (args.duration > 0 and args.warmup >= args.duration):
        parser.error("--warmup must be non-negative and shorter than --duration")
    if args.trace_limit < 0:
        parser.error("--trace-limit must be non-negative")
    if args.policy_blend < 0.0:
        parser.error("--policy-blend must be non-negative")
    if not 0.0 <= args.stand_enter_planar <= args.stand_exit_planar:
        parser.error("stand planar thresholds must be non-negative and ordered")
    if not 0.0 <= args.stand_enter_yaw <= args.stand_exit_yaw:
        parser.error("stand yaw thresholds must be non-negative and ordered")
    return args


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    actuator_names = object_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if actuator_names != SDK_JOINT_ORDER:
        raise ValueError(f"MJCF actuator order mismatch: {actuator_names}")

    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key < 0:
        raise ValueError("MJCF does not contain the home keyframe")
    if args.initial_state == "home":
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        data.ctrl[:] = HOME_SDK
        mujoco.mj_forward(model, data)
    elif args.initial_state == "prone":
        reset_to_prone_state(model, data)
    else:
        reset_to_recovery_state(
            model,
            data,
            args.initial_state,
            orientation_offset_deg=tuple(args.recovery_orientation_offset_deg),
            joint_noise=args.recovery_joint_noise,
            linear_velocity=tuple(args.recovery_linear_velocity),
            angular_velocity=tuple(args.recovery_angular_velocity),
            seed=args.recovery_seed,
        )

    scripted_prone = args.initial_state == "prone"
    recovery_initial_state = args.initial_state != "home"

    controller = None
    recovery_controller = None
    command_schedule = [
        (float(step[0]), np.asarray(step[1:], dtype=np.float64))
        for step in (args.command_step or [])
    ]
    fixed_command = (
        command_schedule[0][1].copy()
        if command_schedule
        else np.asarray(args.command, dtype=np.float64)
    )
    if args.policy is not None:
        if args.stand_policy is None:
            controller = PolicyController(
                model,
                args.policy,
                args.deploy_yaml,
                fixed_command,
                args.trace,
                args.trace_limit,
                args.explicit_pd,
                encoder_path=args.encoder,
            )
            run_mode = f"policy={args.policy}"
            if args.encoder is not None:
                run_mode += f", encoder={args.encoder}"
        else:
            controller = RoutedPolicyController(
                model,
                args.policy,
                args.deploy_yaml,
                args.stand_policy,
                args.stand_deploy_yaml,
                fixed_command,
                args.trace,
                args.trace_limit,
                args.explicit_pd,
                args.stand_enter_planar,
                args.stand_enter_yaw,
                args.stand_exit_planar,
                args.stand_exit_yaw,
                args.policy_blend,
            )
            run_mode = f"policy={args.policy}, stand_policy={args.stand_policy}"
        if args.recovery_policy is not None:
            recovery_controller = PolicyController(
                model,
                args.recovery_policy,
                args.recovery_deploy_yaml,
                np.zeros(3, dtype=np.float64),
                trace_limit=0,
                explicit_pd=args.explicit_pd,
            )
            recovery_controller.set_command(np.zeros(3, dtype=np.float64))
            run_mode += f", recovery_policy={args.recovery_policy}"
    else:
        run_mode = "position hold"

    home_target_sdk = (
        controller.sdk_default_position.copy() if controller is not None else HOME_SDK.copy()
    )
    if args.initial_state == "home":
        data.ctrl[:] = home_target_sdk
        mujoco.mj_forward(model, data)

    total_steps = None if args.duration == 0 else int(np.ceil(args.duration / model.opt.timestep))
    started = time.perf_counter()
    viewer_handle = None
    viewer_thread = None
    interactive_initial_mode = (
        InteractiveControls.VELOCITY
        if recovery_initial_state and controller is not None
        else InteractiveControls.FIX_STAND
    )
    interactive_controls = (
        InteractiveControls(controller, interactive_initial_mode) if args.interactive else None
    )
    if args.viewer:
        from mujoco import viewer

        existing_threads = set(threading.enumerate())
        viewer_handle = viewer.launch_passive(
            model,
            data,
            key_callback=interactive_controls.key_callback if interactive_controls is not None else None,
        )
        new_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in existing_threads and thread is not threading.current_thread()
        ]
        if len(new_threads) == 1:
            viewer_thread = new_threads[0]
        if args.camera_mode == "tracking":
            with viewer_handle.lock():
                viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer_handle.cam.trackbodyid = model.body("base").id
                viewer_handle.cam.distance = 1.4
                viewer_handle.cam.azimuth = 135.0
                viewer_handle.cam.elevation = -20.0

    if interactive_controls is not None:
        print(
            "interactive keys: P=Passive, R=FixStand->PolicyHold, V=Velocity, "
            "W/S=vx, A/D=vy, Q/E=yaw, X=smooth stop; Space and 0..5 remain MuJoCo keys",
            flush=True,
        )
        print(f"viewer camera={args.camera_mode}", flush=True)

    metric_start_pos = None
    metric_start_time = None
    heights: list[float] = []
    tilt_angles: list[float] = []
    body_linear_velocities: list[np.ndarray] = []
    body_yaw_rates: list[float] = []
    metric_times: list[float] = []
    metric_commands: list[np.ndarray] = []
    metric_policy_commands: list[np.ndarray] = []
    hip_outward_angles: list[np.ndarray] = []
    action_first_differences: list[float] = []
    action_second_differences: list[float] = []
    foot_slip_speeds: list[float] = []
    foot_impact_speeds: list[float] = []
    previous_foot_contacts: np.ndarray | None = None
    previous_foot_world_velocities: np.ndarray | None = None
    metric_previous_action = np.zeros(12, dtype=np.float64)
    metric_previous_previous_action = np.zeros(12, dtype=np.float64)
    foot_body_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in ("FR", "FL", "RR", "RL")],
        dtype=np.int32,
    )
    if np.any(foot_body_ids < 0):
        raise ValueError("MJCF must contain all four foot bodies")
    contact_samples: list[np.ndarray] = []
    self_contact_pair_counts: list[int] = []
    illegal_ground_pair_counts: list[int] = []

    sdk_joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in SDK_JOINT_ORDER],
        dtype=np.int32,
    )
    sdk_qpos_addresses = model.jnt_qposadr[sdk_joint_ids]
    sdk_dof_addresses = model.jnt_dofadr[sdk_joint_ids]
    nominal_forcerange = model.actuator_forcerange.copy()
    nominal_forcelimited = model.actuator_forcelimited.copy()
    last_interactive_mode = None
    last_command_revision = -1
    last_recovery_hold_active = None
    stand_start_positions = home_target_sdk.copy()
    stand_start_time = 0.0
    velocity_handoff_end_time = 0.0
    collect_metrics = interactive_controls is None
    step = 0
    recovery_first_upright_time = None
    recovery_stable_time = 0.0
    recovery_success_time = None
    recovery_release_time = args.recovery_ramp + args.recovery_hold
    recovery_policy_started = not scripted_prone or args.recovery_ramp == 0.0
    recovery_command_released = not scripted_prone or recovery_release_time == 0.0
    if recovery_controller is not None:
        recovery_policy_started = False
        recovery_command_released = False
        recovery_release_time = float("inf")
    recovery_handoff_started = False
    recovery_handoff_start_action: np.ndarray | None = None
    recovery_handoff_start_time = float("inf")
    recovery_locomotion_hold_end = float("inf")
    command_schedule_index = 0
    recovery_heights: list[float] = []
    recovery_tilts: list[float] = []
    recovery_direct_forces: list[float] = []
    recovery_calf_forces: list[float] = []
    recovery_joint_speeds: list[float] = []
    if controller is not None and not recovery_command_released:
        controller.set_command(np.zeros(3, dtype=np.float64))

    try:
        while total_steps is None or step < total_steps:
            if viewer_handle is not None and not viewer_handle.is_running():
                break
            if interactive_controls is None:
                recovery_ramp_active = (
                    scripted_prone and data.time < args.recovery_ramp
                )
                if recovery_ramp_active:
                    blend = quintic_smoothstep(data.time / args.recovery_ramp)
                    data.ctrl[:] = (1.0 - blend) * PRONE_SDK + blend * home_target_sdk
                    if controller is not None:
                        controller.apply_actuator_limits(model, data)
                else:
                    if controller is not None:
                        if recovery_controller is not None and not recovery_handoff_started:
                            if step % recovery_controller.decimation == 0:
                                recovery_controller.update(model, data)
                            recovery_controller.apply_actuator_limits(model, data)
                        elif command_schedule:
                            while (
                                command_schedule_index + 1 < len(command_schedule)
                                and data.time >= command_schedule[command_schedule_index + 1][0]
                            ):
                                command_schedule_index += 1
                                controller.set_command(command_schedule[command_schedule_index][1])
                        if recovery_controller is None or recovery_handoff_started:
                            policy_just_started = not recovery_policy_started
                            if policy_just_started:
                                controller.reset_history()
                                recovery_policy_started = True
                            release_deadline = (
                                recovery_locomotion_hold_end
                                if recovery_controller is not None
                                else recovery_release_time
                            )
                            if not recovery_command_released and data.time >= release_deadline:
                                controller.set_command(fixed_command)
                                recovery_command_released = True
                                if recovery_controller is not None:
                                    print(
                                        f"recovery_handoff: locomotion command released at {data.time:.3f} s",
                                        flush=True,
                                    )
                            if policy_just_started or step % controller.decimation == 0:
                                if (
                                    recovery_handoff_start_action is not None
                                    and data.time < recovery_handoff_start_time + args.policy_blend
                                    and not isinstance(controller, RoutedPolicyController)
                                ):
                                    blend = quintic_smoothstep(
                                        (data.time - recovery_handoff_start_time) / args.policy_blend
                                    ) if args.policy_blend > 0.0 else 1.0
                                    controller.update_with_blend(
                                        model, data, recovery_handoff_start_action, blend
                                    )
                                else:
                                    controller.update(model, data)
                            controller.apply_actuator_limits(model, data)
                    elif scripted_prone:
                        data.ctrl[:] = home_target_sdk
            else:
                interactive_mode, command, revision = interactive_controls.snapshot()
                if interactive_mode != last_interactive_mode:
                    if interactive_mode == InteractiveControls.FIX_STAND:
                        stand_start_positions = data.qpos[sdk_qpos_addresses].copy()
                        stand_start_time = data.time
                    elif (
                        interactive_mode == InteractiveControls.POLICY_HOLD
                        and controller is not None
                    ):
                        last_command_revision = -1
                    elif interactive_mode == InteractiveControls.VELOCITY and controller is not None:
                        if last_interactive_mode != InteractiveControls.POLICY_HOLD:
                            controller.reset_history()
                            velocity_handoff_end_time = data.time + args.recovery_hold
                        else:
                            velocity_handoff_end_time = data.time
                        last_command_revision = -1
                    last_interactive_mode = interactive_mode

                if interactive_mode == InteractiveControls.PASSIVE:
                    model.actuator_forcelimited[:] = 1
                    model.actuator_forcerange[:] = 0.0
                elif interactive_mode == InteractiveControls.FIX_STAND:
                    model.actuator_forcelimited[:] = nominal_forcelimited
                    model.actuator_forcerange[:] = nominal_forcerange
                    blend = (
                        1.0
                        if args.recovery_ramp == 0.0
                        else quintic_smoothstep((data.time - stand_start_time) / args.recovery_ramp)
                    )
                    data.ctrl[:] = (1.0 - blend) * stand_start_positions + blend * home_target_sdk
                    if controller is not None:
                        controller.apply_actuator_limits(model, data)
                    if data.time - stand_start_time >= args.recovery_ramp:
                        message = interactive_controls.complete_fix_stand()
                        if message is not None:
                            print(f"{message}; enter a velocity command to resume", flush=True)
                elif interactive_mode == InteractiveControls.POLICY_HOLD:
                    if controller is None:
                        raise RuntimeError("PolicyHold mode requires a policy controller")
                    if revision != last_command_revision:
                        controller.set_command(np.zeros(3, dtype=np.float64))
                        last_command_revision = revision
                    if step % controller.decimation == 0:
                        controller.update(model, data)
                    controller.apply_actuator_limits(model, data)
                elif interactive_mode == InteractiveControls.VELOCITY:
                    if controller is None:
                        raise RuntimeError("Velocity mode requires a policy controller")
                    recovery_ramp_active = (
                        scripted_prone and data.time < args.recovery_ramp
                    )
                    if recovery_ramp_active:
                        blend = quintic_smoothstep(data.time / args.recovery_ramp)
                        data.ctrl[:] = (1.0 - blend) * PRONE_SDK + blend * home_target_sdk
                        controller.apply_actuator_limits(model, data)
                    else:
                        policy_just_started = not recovery_policy_started
                        if policy_just_started:
                            controller.reset_history()
                            recovery_policy_started = True
                        recovery_hold_active = data.time < velocity_handoff_end_time
                        if (
                            revision != last_command_revision
                            or recovery_hold_active != last_recovery_hold_active
                        ):
                            effective_command = (
                                np.zeros(3, dtype=np.float64) if recovery_hold_active else command
                            )
                            controller.set_command(effective_command)
                            last_command_revision = revision
                            last_recovery_hold_active = recovery_hold_active
                        if policy_just_started or step % controller.decimation == 0:
                            controller.update(model, data)
                        controller.apply_actuator_limits(model, data)
                else:
                    raise RuntimeError(f"Unknown interactive mode: {interactive_mode}")
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Simulation became non-finite at step {step}")
            if recovery_initial_state:
                rotation_world_from_base = data.xmat[model.body("base").id].reshape(3, 3)
                projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
                tilt = math.degrees(math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0))))
                if recovery_first_upright_time is None and data.qpos[2] >= 0.25 and tilt <= 15.0:
                    recovery_first_upright_time = float(data.time)
                recovery_contacts = ground_foot_contacts(
                    data.contact,
                    model.geom_bodyid,
                    foot_body_ids,
                )
                recovery_stable = (
                    data.qpos[2] >= 0.27
                    and tilt <= 15.0
                    and np.linalg.norm(data.qvel[3:6]) <= 0.5
                    and np.all(recovery_contacts)
                )
                recovery_stable_time = (
                    recovery_stable_time + model.opt.timestep if recovery_stable else 0.0
                )
                if (
                    recovery_success_time is None
                    and recovery_stable_time >= args.recovery_upright_dwell
                ):
                    recovery_success_time = float(data.time)
                    if recovery_controller is not None and not recovery_handoff_started:
                        recovery_handoff_start_action = recovery_controller.current_action.copy()
                        controller.set_command(np.zeros(3, dtype=np.float64))
                        if isinstance(controller, RoutedPolicyController):
                            controller.begin_external_handoff(recovery_handoff_start_action)
                        else:
                            controller.reset_history(recovery_handoff_start_action)
                        recovery_handoff_start_time = float(data.time)
                        recovery_locomotion_hold_end = (
                            recovery_handoff_start_time + args.policy_blend + args.recovery_locomotion_hold
                        )
                        recovery_policy_started = True
                        recovery_handoff_started = True
                        print(
                            f"recovery_handoff: locomotion hold started at {data.time:.3f} s",
                            flush=True,
                        )
                if scripted_prone and data.time <= recovery_release_time:
                    recovery_heights.append(float(data.qpos[2]))
                    recovery_tilts.append(tilt)
                    actuator_forces = np.abs(data.actuator_force)
                    recovery_direct_forces.append(float(np.max(actuator_forces[[0, 1, 3, 4, 6, 7, 9, 10]])))
                    recovery_calf_forces.append(float(np.max(actuator_forces[[2, 5, 8, 11]])))
                    recovery_joint_speeds.append(float(np.max(np.abs(data.qvel[sdk_dof_addresses]))))
            if viewer_handle is not None:
                viewer_handle.sync()
            if args.realtime or viewer_handle is not None:
                target_time = started + (step + 1) * model.opt.timestep
                delay = target_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if collect_metrics and data.time >= args.warmup:
                if metric_start_pos is None:
                    metric_start_pos = data.qpos[:3].copy()
                    metric_start_time = float(data.time)
                heights.append(float(data.qpos[2]))
                rotation_world_from_base = data.xmat[model.body("base").id].reshape(3, 3)
                projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
                tilt_angles.append(math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0))))
                body_linear_velocities.append(rotation_world_from_base.T @ data.qvel[:3])
                body_yaw_rates.append(float(data.sensordata[controller.gyro_address + 2]) if controller else 0.0)
                metric_times.append(float(data.time))
                metric_commands.append(
                    controller.command.copy() if controller is not None else np.zeros(3)
                )
                metric_policy_commands.append(
                    controller.policy_command.copy() if controller is not None else np.zeros(3)
                )
                hip_positions = data.qpos[sdk_qpos_addresses[[0, 3, 6, 9]]]
                hip_outward_angles.append(hip_positions * np.array([-1.0, 1.0, -1.0, 1.0]))
                foot_contacts = ground_foot_contacts(
                    data.contact,
                    model.geom_bodyid,
                    foot_body_ids,
                )
                contact_samples.append(foot_contacts)
                self_contact_pair_counts.append(
                    robot_self_contact_pair_count(data.contact, model.geom_bodyid)
                )
                illegal_ground_pair_counts.append(
                    illegal_ground_contact_pair_count(
                        data.contact, model.geom_bodyid, foot_body_ids
                    )
                )
                if controller is not None:
                    current_action = controller.current_action
                    action_first_differences.append(
                        float(np.mean(np.square(current_action - metric_previous_action)))
                    )
                    action_second_differences.append(
                        float(
                            np.mean(
                                np.square(
                                    current_action
                                    - 2.0 * metric_previous_action
                                    + metric_previous_previous_action
                                )
                            )
                        )
                    )
                    metric_previous_previous_action = metric_previous_action.copy()
                    metric_previous_action = current_action.copy()
                    # MuJoCo exposes body spatial velocity through cvel.  Its
                    # linear part is expressed in the body frame; the norm is
                    # frame invariant and therefore suitable for contact slip.
                    foot_speed = np.linalg.norm(data.cvel[foot_body_ids, 3:6], axis=1)
                    foot_slip_speeds.append(float(np.mean(foot_speed[foot_contacts])) if np.any(foot_contacts) else 0.0)
                    foot_world_velocities = np.zeros((4, 3), dtype=np.float64)
                    object_velocity = np.zeros(6, dtype=np.float64)
                    for foot_index, body_id in enumerate(foot_body_ids):
                        mujoco.mj_objectVelocity(
                            model,
                            data,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(body_id),
                            object_velocity,
                            0,
                        )
                        foot_world_velocities[foot_index] = object_velocity[3:]
                    if previous_foot_contacts is not None and previous_foot_world_velocities is not None:
                        first_contacts = foot_contacts & ~previous_foot_contacts
                        if np.any(first_contacts):
                            downward_speed = np.maximum(-previous_foot_world_velocities[:, 2], 0.0)
                            foot_impact_speeds.extend(downward_speed[first_contacts].tolist())
                    previous_foot_contacts = foot_contacts.copy()
                    previous_foot_world_velocities = foot_world_velocities
            step += 1
    finally:
        if viewer_handle is not None:
            viewer_handle.close()
        if viewer_thread is not None:
            viewer_thread.join(timeout=5.0)
            if viewer_thread.is_alive():
                raise RuntimeError("MuJoCo viewer did not close within 5 seconds")
        if controller is not None:
            controller.close()
        if recovery_controller is not None:
            recovery_controller.close()

    elapsed = time.perf_counter() - started
    print(f"sim2sim OK: {run_mode}")
    print(f"initial_state={args.initial_state}")
    if scripted_prone and controller is not None and interactive_controls is None:
        print(
            f"recovery_ramp={args.recovery_ramp:.3f} s, "
            f"recovery_zero_command_hold={args.recovery_hold:.3f} s"
        )
    print(f"simulated={data.time:.3f} s, wall={elapsed:.3f} s, real-time-factor={data.time / elapsed:.2f}")
    print(f"base position=[{data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f}]")
    if controller is not None and metric_start_pos is not None and metric_start_time is not None:
        metric_duration = float(data.time) - metric_start_time
        displacement = data.qpos[:3] - metric_start_pos
        mean_body_velocity = np.mean(np.asarray(body_linear_velocities), axis=0)
        forward_speed = float(mean_body_velocity[0])
        lateral_speed = float(mean_body_velocity[1])
        mean_yaw_rate = float(np.mean(body_yaw_rates))
        measured_velocity = np.array(
            [forward_speed, lateral_speed, mean_yaw_rate], dtype=np.float64
        )
        tracking_error = np.abs(measured_velocity - controller.command)
        contacts = np.asarray(contact_samples, dtype=np.int8)
        transitions = np.count_nonzero(np.diff(contacts, axis=0), axis=0)
        duty_factors = contacts.mean(axis=0)
        hip_outward_deg = np.degrees(np.asarray(hip_outward_angles))
        self_contact_counts = np.asarray(self_contact_pair_counts, dtype=np.int64)
        illegal_ground_counts = np.asarray(illegal_ground_pair_counts, dtype=np.int64)
        if command_schedule:
            print(
                "tracking aggregate: measured=[{:.3f}, {:.3f}, {:.3f}] over changing commands".format(
                    forward_speed,
                    lateral_speed,
                    mean_yaw_rate,
                )
            )
        else:
            print(
                "tracking: command=[{:.3f}, {:.3f}, {:.3f}], "
                "policy_command=[{:.3f}, {:.3f}, {:.3f}], "
                "measured=[{:.3f}, {:.3f}, {:.3f}], abs_error=[{:.3f}, {:.3f}, {:.3f}]".format(
                    *controller.command,
                    *controller.policy_command,
                    forward_speed,
                    lateral_speed,
                    mean_yaw_rate,
                    *tracking_error,
                )
            )
        if len(metric_times) >= 2:
            integrated_yaw = trapezoid_integral(
                np.asarray(body_yaw_rates), np.asarray(metric_times)
            )
            if command_schedule:
                requested_yaw_integral = trapezoid_integral(
                    np.asarray(metric_commands)[:, 2],
                    np.asarray(metric_times),
                )
            else:
                requested_yaw_integral = float(controller.command[2] * metric_duration)
            print(
                "yaw_integral: measured={:.3f} rad, requested={:.3f} rad, bias={:.3f} rad".format(
                    integrated_yaw,
                    requested_yaw_integral,
                    integrated_yaw - requested_yaw_integral,
                )
            )
        print(f"world displacement=[{displacement[0]:.3f}, {displacement[1]:.3f}] m")
        print(
            f"stability: min_height={min(heights):.3f} m, "
            f"mean_height={np.mean(heights):.3f} m, "
            f"max_tilt={math.degrees(max(tilt_angles)):.2f} deg"
        )
        print(
            "posture: mean_hip_outward=[{}] deg, max_hip_outward={:.2f} deg".format(
                ", ".join(f"{value:.2f}" for value in hip_outward_deg.mean(axis=0)),
                float(hip_outward_deg.max()),
            )
        )
        print(
            "contacts: duty=[{}], transitions=[{}]".format(
                ", ".join(f"{value:.2f}" for value in duty_factors),
                ", ".join(str(int(value)) for value in transitions),
            )
        )
        print(
            "self_collision: contact_steps={}/{}, mean_pairs={:.6f}, max_pairs={}".format(
                int(np.count_nonzero(self_contact_counts)),
                len(self_contact_counts),
                float(np.mean(self_contact_counts)),
                int(np.max(self_contact_counts)),
            )
        )
        print(
            "illegal_ground_contact: contact_steps={}/{}, mean_pairs={:.6f}, max_pairs={}".format(
                int(np.count_nonzero(illegal_ground_counts)),
                len(illegal_ground_counts),
                float(np.mean(illegal_ground_counts)),
                int(np.max(illegal_ground_counts)),
            )
        )
        if action_first_differences:
            print(
                "smoothness: mean_action_delta2={:.6f}, mean_action_second_delta2={:.6f}".format(
                    float(np.mean(action_first_differences)),
                    float(np.mean(action_second_differences)),
                )
            )
        if foot_slip_speeds:
            print(f"foot_slip: mean_contact_speed={float(np.mean(foot_slip_speeds)):.4f} m/s")
        print(
            "foot_impact: mean_velocity={:.4f} m/s, max_velocity={:.4f} m/s".format(
                float(np.mean(foot_impact_speeds)) if foot_impact_speeds else 0.0,
                float(np.max(foot_impact_speeds)) if foot_impact_speeds else 0.0,
            )
        )
        if command_schedule:
            sample_times = np.asarray(metric_times)
            sampled_commands = np.asarray(metric_commands)
            sampled_policy_commands = np.asarray(metric_policy_commands)
            sampled_velocity = np.column_stack(
                (
                    np.asarray(body_linear_velocities)[:, :2],
                    np.asarray(body_yaw_rates),
                )
            )
            for index, (start_time, requested) in enumerate(command_schedule):
                end_time = (
                    command_schedule[index + 1][0]
                    if index + 1 < len(command_schedule)
                    else float(data.time)
                )
                # Exclude one second after each command edge from steady-state
                # tracking, but keep every sample in the global stability gate.
                segment_start = start_time + min(1.0, 0.25 * (end_time - start_time))
                mask = (sample_times >= segment_start) & (sample_times < end_time)
                if not np.any(mask):
                    continue
                measured = sampled_velocity[mask].mean(axis=0)
                error = np.abs(measured - requested)
                policy_command = sampled_policy_commands[mask][0]
                print(
                    "sequence segment {}: requested=[{:.3f}, {:.3f}, {:.3f}], "
                    "policy_command=[{:.3f}, {:.3f}, {:.3f}], "
                    "measured=[{:.3f}, {:.3f}, {:.3f}], "
                    "abs_error=[{:.3f}, {:.3f}, {:.3f}]".format(
                        index,
                        *requested,
                        *policy_command,
                        *measured,
                        *error,
                    )
                )
    elif interactive_controls is not None:
        print("interactive session ended; tracking metrics are omitted because commands changed during the run")
    if recovery_initial_state:
        if recovery_first_upright_time is None:
            print("recovery: no upright state reached (height >= 0.25 m, tilt <= 15 deg)")
        else:
            print(f"recovery: first upright state at {recovery_first_upright_time:.3f} s")
        if recovery_success_time is None:
            print(
                "self_righting: failed stable gate "
                "(height >= 0.27 m, tilt <= 15 deg, angular speed <= 0.5 rad/s, "
                "four-foot contact for 0.4 s)"
            )
        else:
            print(f"self_righting: success at {recovery_success_time:.3f} s")
        if recovery_heights:
            print(
                "recovery transition: max_height={:.3f} m, max_tilt={:.2f} deg, "
                "max_direct_force={:.2f} Nm, max_calf_force={:.2f} Nm, "
                "max_joint_speed={:.2f} rad/s".format(
                    max(recovery_heights),
                    max(recovery_tilts),
                    max(recovery_direct_forces),
                    max(recovery_calf_forces),
                    max(recovery_joint_speeds),
                )
            )
    if controller is not None and controller.inference_times_ms:
        samples = np.asarray(controller.inference_times_ms)
        print(
            f"ONNX CPU inference: mean={samples.mean():.3f} ms, "
            f"p99={np.percentile(samples, 99):.3f} ms, calls={len(samples)}"
        )


if __name__ == "__main__":
    run(parse_args())
