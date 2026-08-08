#!/usr/bin/env python3
"""Run the custom-dog position policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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

        self.command = vector(command, 3, "velocity command")
        ranges = self.cfg["commands"]["base_velocity"]["ranges"]
        for value, key in zip(self.command, ("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            lower, upper = ranges[key]
            if not lower <= value <= upper:
                raise ValueError(f"Command {key}={value} is outside [{lower}, {upper}]")

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

    def _scaled_term(self, name: str, values: np.ndarray) -> np.ndarray:
        term = self.cfg["observations"][name]
        if int(term["history_length"]) != 1:
            raise ValueError(f"Only history_length=1 is supported, got {name}={term['history_length']}")
        scale = vector(term["scale"], len(values), f"{name} scale")
        lower, upper = term["clip"]
        return np.clip(values * scale, lower, upper)

    def observation(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        angular_velocity = data.sensordata[self.gyro_address : self.gyro_address + 3].copy()
        rotation_world_from_base = data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation_world_from_base.T @ np.array([0.0, 0.0, -1.0])
        joint_position = data.qpos[self.qpos_addresses]
        joint_velocity = data.qvel[self.dof_addresses]

        raw_terms = {
            "base_ang_vel": angular_velocity,
            "projected_gravity": projected_gravity,
            "velocity_commands": self.command,
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

        action = np.clip(action, self.action_clip[:, 0], self.action_clip[:, 1])
        target_policy_order = self.action_offset + self.action_scale * action
        data.ctrl[self.joint_map] = target_policy_order
        self.previous_action = action


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=model_dir / "custom_dog.xml")
    parser.add_argument("--policy", type=Path, help="Exported policy.onnx; omit for position-hold test")
    parser.add_argument("--deploy-yaml", type=Path, help="params/deploy.yaml from the same training run")
    parser.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "YAW"), default=(0, 0, 0))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace headless simulation at wall-clock speed")
    args = parser.parse_args()
    if (args.policy is None) != (args.deploy_yaml is None):
        parser.error("--policy and --deploy-yaml must be supplied together")
    if args.duration <= 0:
        parser.error("--duration must be positive")
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
        )
        mode = f"policy={args.policy}"
    else:
        mode = "position hold"

    total_steps = int(np.ceil(args.duration / model.opt.timestep))
    started = time.perf_counter()
    viewer_handle = None
    if args.viewer:
        from mujoco import viewer

        viewer_handle = viewer.launch_passive(model, data)

    try:
        for step in range(total_steps):
            if viewer_handle is not None and not viewer_handle.is_running():
                break
            if controller is not None and step % controller.decimation == 0:
                controller.update(model, data)
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
    finally:
        if viewer_handle is not None:
            viewer_handle.close()

    elapsed = time.perf_counter() - started
    print(f"sim2sim OK: {mode}")
    print(f"simulated={data.time:.3f} s, wall={elapsed:.3f} s, real-time-factor={data.time / elapsed:.2f}")
    print(f"base position=[{data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f}]")
    if controller is not None and controller.inference_times_ms:
        samples = np.asarray(controller.inference_times_ms)
        print(
            f"ONNX CPU inference: mean={samples.mean():.3f} ms, "
            f"p99={np.percentile(samples, 99):.3f} ms, calls={len(samples)}"
        )


if __name__ == "__main__":
    run(parse_args())
