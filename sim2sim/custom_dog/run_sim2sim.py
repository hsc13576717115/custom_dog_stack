#!/usr/bin/env python3
"""Run the custom-dog position policy in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import math
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

EXPECTED_OBSERVATIONS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
]

HOME_SDK = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5],
    dtype=np.float64,
)

DIRECT_TORQUE_SPEED = (13.5, 30.0, 20.2, 23.4)
CALF_TORQUE_SPEED = (6.75, 15.0, 40.4, 46.8)


def object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) for index in range(count)]


def vector(value: object, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{label} must contain {length} values, got shape {result.shape}")
    return result


class PolicyController:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: Path,
        deploy_path: Path,
        command: np.ndarray,
        trace_path: Path | None = None,
        trace_limit: int = 500,
    ) -> None:
        import onnxruntime as ort

        with deploy_path.open(encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream)

        observation_names = list(self.cfg["observations"])
        if observation_names != EXPECTED_OBSERVATIONS:
            raise ValueError(
                f"Observation contract mismatch: expected {EXPECTED_OBSERVATIONS}, got {observation_names}"
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
        self.action_clip = np.asarray(action_cfg["clip"], dtype=np.float64)
        if self.action_clip.shape != (12, 2):
            raise ValueError(f"Action clip must have shape (12, 2), got {self.action_clip.shape}")

        # Optional sim2sim calibration, expressed in policy joint order.  The
        # bias is applied after inference so the raw action remains the value
        # fed back through the 45-D last_action observation.
        bias_cfg = self.cfg.get("joint_target_bias")
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

        ranges = self.cfg["commands"]["base_velocity"]["ranges"]
        self.command_ranges = np.asarray(
            [ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_z"]],
            dtype=np.float64,
        )
        if (
            self.command_ranges.shape != (3, 2)
            or not np.isfinite(self.command_ranges).all()
            or np.any(self.command_ranges[:, 0] > self.command_ranges[:, 1])
        ):
            raise ValueError("Velocity command ranges must be finite lower/upper pairs")
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

        policy_stiffness = vector(self.cfg["stiffness"], 12, "stiffness")
        policy_damping = vector(self.cfg["damping"], 12, "damping")
        self.sdk_stiffness = np.empty(12, dtype=np.float64)
        self.sdk_damping = np.empty(12, dtype=np.float64)
        self.sdk_stiffness[self.joint_map] = policy_stiffness
        self.sdk_damping[self.joint_map] = policy_damping

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
        if inputs[0].shape[-1] != 45 or outputs[0].shape[-1] != 12:
            raise ValueError(f"Expected ONNX shapes (*,45)->(*,12), got {inputs[0].shape}->{outputs[0].shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.previous_action = np.zeros(12, dtype=np.float64)
        self.inference_times_ms: list[float] = []
        self.policy_steps = 0
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
                ("obs", 45),
                ("action", 12),
                ("target_q", 12),
                ("joint_q", 12),
                ("joint_dq", 12),
                ("ang_vel", 3),
                ("projected_gravity", 3),
                ("base_pos", 3),
                ("base_quat", 4),
            ):
                header.extend(f"{prefix}_{index}" for index in range(count))
            self.trace_writer.writerow(header)

    def target_bias(self) -> np.ndarray:
        """Return the forward-command-dependent target calibration."""
        if self.bias_vx_max <= self.bias_vx_min:
            return np.zeros(12, dtype=np.float64)
        blend = np.clip(
            (float(self.command[0]) - self.bias_vx_min)
            / (self.bias_vx_max - self.bias_vx_min),
            0.0,
            1.0,
        )
        return blend * self.target_bias_policy

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
        calibration_cfg = calibration_root.get("lin_vel_x")
        if calibration_cfg is None:
            return

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
            raise ValueError("command_calibration.lin_vel_x must contain finite monotonic arrays")
        lower, upper = self.command_ranges[0]
        if requested[0] > lower or requested[-1] < upper:
            raise ValueError("command calibration must cover the exported lin_vel_x range")
        if np.any(policy < lower) or np.any(policy > upper):
            raise ValueError("calibrated policy commands must remain inside lin_vel_x range")
        self.policy_command[0] = np.interp(self.command[0], requested, policy)

    def reset_history(self) -> None:
        self.previous_action.fill(0.0)


    def _scaled_term(self, name: str, values: np.ndarray) -> np.ndarray:
        term = self.cfg["observations"][name]
        if int(term["history_length"]) != 1:
            raise ValueError(f"Only history_length=1 is supported, got {name}={term['history_length']}")
        scale = vector(term["scale"], len(values), f"{name} scale")
        lower, upper = term["clip"]
        # Isaac Lab and unitree_rl_lab clip the raw observation before scale.
        return np.clip(values, lower, upper) * scale

    def observation(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        angular_velocity = data.sensordata[self.gyro_address : self.gyro_address + 3].copy()
        rotation_world_from_base = data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
        joint_position = data.qpos[self.qpos_addresses]
        joint_velocity = data.qvel[self.dof_addresses]

        raw_terms = {
            "base_ang_vel": angular_velocity,
            "projected_gravity": projected_gravity,
            "velocity_commands": self.policy_command,
            "joint_pos_rel": joint_position - self.default_position,
            "joint_vel_rel": joint_velocity,
            "last_action": self.previous_action,
        }
        observation = np.concatenate(
            [self._scaled_term(name, raw_terms[name]) for name in EXPECTED_OBSERVATIONS]
        ).astype(np.float32)
        if observation.shape != (45,) or not np.isfinite(observation).all():
            raise RuntimeError(f"Invalid policy observation: shape={observation.shape}")
        return observation

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        observation = self.observation(model, data)
        started = time.perf_counter()
        action = self.session.run(
            [self.output_name], {self.input_name: observation[np.newaxis, :]}
        )[0][0].astype(np.float64)
        self.inference_times_ms.append((time.perf_counter() - started) * 1000.0)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid policy action: shape={action.shape}")

        target_policy_order = np.clip(
            self.action_offset + self.action_scale * action,
            self.action_clip[:, 0],
            self.action_clip[:, 1],
        )
        target_policy_order += self.target_bias()
        data.ctrl[self.joint_map] = target_policy_order
        self.previous_action = action
        self.policy_steps += 1

        if self.trace_writer is not None and (
            self.trace_limit == 0 or self.policy_steps <= self.trace_limit
        ):
            angular_velocity = data.sensordata[
                self.gyro_address : self.gyro_address + 3
            ].copy()
            rotation_world_from_base = data.xmat[self.base_id].reshape(3, 3)
            projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
            row = [self.policy_steps, data.time]
            for values in (
                observation,
                action,
                target_policy_order,
                data.qpos[self.qpos_addresses],
                data.qvel[self.dof_addresses],
                angular_velocity,
                projected_gravity,
                data.qpos[:3],
                data.qpos[3:7],
            ):
                row.extend(float(value) for value in values)
            self.trace_writer.writerow(row)
            self.trace_stream.flush()

    def apply_actuator_limits(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Match the Unitree torque-speed clipping used during Isaac training."""
        joint_pos = data.qpos[self.sdk_qpos_addresses]
        joint_vel = data.qvel[self.sdk_dof_addresses]
        desired_effort = self.sdk_stiffness * (data.ctrl - joint_pos) - self.sdk_damping * joint_vel

        for index, joint_name in enumerate(SDK_JOINT_ORDER):
            x1, x2, y1, y2 = CALF_TORQUE_SPEED if "calf" in joint_name else DIRECT_TORQUE_SPEED
            full_torque = y1 if joint_vel[index] * desired_effort[index] > 0.0 else y2
            speed = abs(joint_vel[index])
            if speed < x1:
                limit = full_torque
            else:
                limit = max(0.0, full_torque * (x2 - speed) / (x2 - x1))
            model.actuator_forcerange[index] = (-limit, limit)

    def close(self) -> None:
        if self.trace_stream is not None:
            self.trace_stream.close()
            self.trace_stream = None
            self.trace_writer = None


class InteractiveControls:
    """Thread-safe keyboard state for the passive MuJoCo viewer."""

    PASSIVE = "passive"
    FIX_STAND = "fix_stand"
    VELOCITY = "velocity"

    def __init__(self, policy_controller: PolicyController | None) -> None:
        self.policy_controller = policy_controller
        self._mode = self.FIX_STAND
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
        self._mode = mode
        self._revision += 1
        return f"interactive mode={mode}, command={self._command.tolist()}"

    def _change_command(self, index: int, delta: float) -> str:
        if self.policy_controller is None:
            return "Velocity commands require --policy and --deploy-yaml"
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
            if key in (glfw.KEY_1, glfw.KEY_P, glfw.KEY_SPACE):
                message = self._set_mode(self.PASSIVE)
            elif key in (glfw.KEY_2, glfw.KEY_R):
                message = self._set_mode(self.FIX_STAND)
            elif key in (glfw.KEY_3, glfw.KEY_V):
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
                self._command.fill(0.0)
                self._revision += 1
                message = "interactive command reset to zero"
        if message is not None:
            print(message)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=model_dir / "custom_dog.xml")
    parser.add_argument("--policy", type=Path, help="Exported policy.onnx; omit for position-hold test")
    parser.add_argument("--deploy-yaml", type=Path, help="params/deploy.yaml from the same training run")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"), default=(0, 0, 0))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds excluded from tracking metrics")
    parser.add_argument("--trace", type=Path, help="Write policy observations and actions to CSV")
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=500,
        help="Maximum policy rows to trace; use 0 for unlimited (default: 500)",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable keyboard Passive/FixStand/Velocity control in the MuJoCo viewer",
    )
    parser.add_argument("--realtime", action="store_true", help="Pace headless simulation at wall-clock speed")
    args = parser.parse_args()
    if (args.policy is None) != (args.deploy_yaml is None):
        parser.error("--policy and --deploy-yaml must be supplied together")
    if args.duration < 0:
        parser.error("--duration must be non-negative")
    if args.duration == 0 and not args.interactive:
        parser.error("--duration 0 is only valid with --interactive")
    if args.interactive and not args.viewer:
        parser.error("--interactive requires --viewer")
    if args.interactive and args.trace is not None:
        parser.error("--trace is only supported for fixed-command runs")
    if args.warmup < 0 or (args.duration > 0 and args.warmup >= args.duration):
        parser.error("--warmup must be non-negative and shorter than --duration")
    if args.trace_limit < 0:
        parser.error("--trace-limit must be non-negative")
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
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    data.ctrl[:] = HOME_SDK
    mujoco.mj_forward(model, data)

    controller = None
    if args.policy is not None:
        controller = PolicyController(
            model,
            args.policy,
            args.deploy_yaml,
            np.asarray(args.command, dtype=np.float64),
            args.trace,
            args.trace_limit,
        )
        run_mode = f"policy={args.policy}"
    else:
        run_mode = "position hold"

    total_steps = None if args.duration == 0 else int(np.ceil(args.duration / model.opt.timestep))
    started = time.perf_counter()
    viewer_handle = None
    interactive_controls = InteractiveControls(controller) if args.interactive else None
    if args.viewer:
        from mujoco import viewer

        viewer_handle = viewer.launch_passive(
            model,
            data,
            key_callback=interactive_controls.key_callback if interactive_controls is not None else None,
        )

    if interactive_controls is not None:
        print(
            "interactive keys: 1/P/Space=Passive, 2/R=FixStand, 3/V=Velocity, "
            "W/S=vx, A/D=vy, Q/E=yaw, X=zero"
        )

    metric_start_pos = None
    metric_start_time = None
    heights: list[float] = []
    tilt_angles: list[float] = []
    body_linear_velocities: list[np.ndarray] = []
    body_yaw_rates: list[float] = []
    foot_body_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in ("FR", "FL", "RR", "RL")],
        dtype=np.int32,
    )
    if np.any(foot_body_ids < 0):
        raise ValueError("MJCF must contain all four foot bodies")
    contact_samples: list[np.ndarray] = []

    sdk_joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in SDK_JOINT_ORDER],
        dtype=np.int32,
    )
    sdk_qpos_addresses = model.jnt_qposadr[sdk_joint_ids]
    nominal_forcerange = model.actuator_forcerange.copy()
    nominal_forcelimited = model.actuator_forcelimited.copy()
    last_interactive_mode = None
    last_command_revision = -1
    stand_start_positions = HOME_SDK.copy()
    stand_start_time = 0.0
    collect_metrics = interactive_controls is None
    step = 0

    try:
        while total_steps is None or step < total_steps:
            if viewer_handle is not None and not viewer_handle.is_running():
                break
            if interactive_controls is None:
                if controller is not None and step % controller.decimation == 0:
                    controller.update(model, data)
                if controller is not None:
                    controller.apply_actuator_limits(model, data)
            else:
                interactive_mode, command, revision = interactive_controls.snapshot()
                if interactive_mode != last_interactive_mode:
                    if interactive_mode == InteractiveControls.FIX_STAND:
                        stand_start_positions = data.qpos[sdk_qpos_addresses].copy()
                        stand_start_time = data.time
                    elif interactive_mode == InteractiveControls.VELOCITY and controller is not None:
                        controller.reset_history()
                        last_command_revision = -1
                    last_interactive_mode = interactive_mode

                if interactive_mode == InteractiveControls.PASSIVE:
                    model.actuator_forcelimited[:] = 1
                    model.actuator_forcerange[:] = 0.0
                elif interactive_mode == InteractiveControls.FIX_STAND:
                    model.actuator_forcelimited[:] = nominal_forcelimited
                    model.actuator_forcerange[:] = nominal_forcerange
                    blend = min(1.0, max(0.0, (data.time - stand_start_time) / 1.0))
                    data.ctrl[:] = (1.0 - blend) * stand_start_positions + blend * HOME_SDK
                    if controller is not None:
                        controller.apply_actuator_limits(model, data)
                elif interactive_mode == InteractiveControls.VELOCITY:
                    if controller is None:
                        raise RuntimeError("Velocity mode requires a policy controller")
                    if revision != last_command_revision:
                        controller.set_command(command)
                        last_command_revision = revision
                    if step % controller.decimation == 0:
                        controller.update(model, data)
                    controller.apply_actuator_limits(model, data)
                else:
                    raise RuntimeError(f"Unknown interactive mode: {interactive_mode}")
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f"Simulation became non-finite at step {step}")
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
                foot_contacts = np.zeros(4, dtype=bool)
                for contact in data.contact:
                    if contact.efc_address < 0:
                        continue
                    body_ids = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
                    foot_contacts |= np.isin(foot_body_ids, body_ids)
                contact_samples.append(foot_contacts)
            step += 1
    finally:
        if viewer_handle is not None:
            viewer_handle.close()
        if controller is not None:
            controller.close()

    elapsed = time.perf_counter() - started
    print(f"sim2sim OK: {run_mode}")
    print(f"simulated={data.time:.3f} s, wall={elapsed:.3f} s, real-time-factor={data.time / elapsed:.2f}")
    print(f"base position=[{data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f}]")
    if controller is not None and metric_start_pos is not None and metric_start_time is not None:
        metric_duration = float(data.time) - metric_start_time
        displacement = data.qpos[:3] - metric_start_pos
        mean_body_velocity = np.mean(np.asarray(body_linear_velocities), axis=0)
        forward_speed = float(mean_body_velocity[0])
        lateral_speed = float(mean_body_velocity[1])
        speed_error = abs(forward_speed - float(controller.command[0]))
        contacts = np.asarray(contact_samples, dtype=np.int8)
        transitions = np.count_nonzero(np.diff(contacts, axis=0), axis=0)
        duty_factors = contacts.mean(axis=0)
        print(
            f"tracking: command_vx={controller.command[0]:.3f} m/s, "
            f"policy_vx={controller.policy_command[0]:.3f} m/s, "
            f"mean_vx={forward_speed:.3f} m/s, abs_error={speed_error:.3f} m/s, "
            f"mean_vy={lateral_speed:.3f} m/s, mean_yaw_rate={np.mean(body_yaw_rates):.3f} rad/s"
        )
        print(f"world displacement=[{displacement[0]:.3f}, {displacement[1]:.3f}] m")
        print(
            f"stability: min_height={min(heights):.3f} m, "
            f"mean_height={np.mean(heights):.3f} m, "
            f"max_tilt={math.degrees(max(tilt_angles)):.2f} deg"
        )
        print(
            "contacts: duty=[{}], transitions=[{}]".format(
                ", ".join(f"{value:.2f}" for value in duty_factors),
                ", ".join(str(int(value)) for value in transitions),
            )
        )
    elif interactive_controls is not None:
        print("interactive session ended; tracking metrics are omitted because commands changed during the run")
    if controller is not None and controller.inference_times_ms:
        samples = np.asarray(controller.inference_times_ms)
        print(
            f"ONNX CPU inference: mean={samples.mean():.3f} ms, "
            f"p99={np.percentile(samples, 99):.3f} ms, calls={len(samples)}"
        )


if __name__ == "__main__":
    run(parse_args())
