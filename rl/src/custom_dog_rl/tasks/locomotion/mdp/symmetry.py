"""Left-right symmetry augmentation for the custom quadruped."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = ["compute_symmetric_states"]


_POLICY_BASE_DIM = 45
_CRITIC_BASE_DIM = 60
_GAIT_PHASE_DIM = 2
_ACTION_DIM = 12


def _counterpart_joint_name(name: str) -> str:
    replacements = {
        "FL_": "FR_",
        "FR_": "FL_",
        "RL_": "RR_",
        "RR_": "RL_",
    }
    for prefix, counterpart in replacements.items():
        if name.startswith(prefix):
            return counterpart + name[len(prefix) :]
    raise ValueError(f"Unsupported custom-dog joint name for symmetry: {name!r}")


def _joint_mirror_metadata(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source indices and coordinate signs in articulation/policy order."""

    joint_names = list(env.scene["robot"].data.joint_names)
    if len(joint_names) != _ACTION_DIM:
        raise ValueError(f"Custom-dog symmetry expects 12 joints, got {len(joint_names)}: {joint_names}")

    name_to_index = {name: index for index, name in enumerate(joint_names)}
    source_indices = []
    signs = []
    for destination_name in joint_names:
        source_name = _counterpart_joint_name(destination_name)
        if source_name not in name_to_index:
            raise ValueError(f"Missing mirrored joint {source_name!r} for {destination_name!r}")
        source_indices.append(name_to_index[source_name])
        signs.append(-1.0 if "_hip_joint" in destination_name else 1.0)

    return (
        torch.tensor(source_indices, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
    )


def _mirror_joint_data(env, joint_data: torch.Tensor) -> torch.Tensor:
    if joint_data.shape[-1] != _ACTION_DIM:
        raise ValueError(f"Custom-dog symmetry expects 12 joint values, got {joint_data.shape[-1]}")
    source_indices, signs = _joint_mirror_metadata(env, joint_data.device)
    return joint_data.index_select(-1, source_indices) * signs.to(dtype=joint_data.dtype)


def _mirror_policy_observation(env, observation: torch.Tensor) -> torch.Tensor:
    valid_dims = (_POLICY_BASE_DIM, _POLICY_BASE_DIM + _GAIT_PHASE_DIM)
    if observation.shape[-1] not in valid_dims:
        raise ValueError(
            f"Custom-dog policy symmetry expects one of {valid_dims}, got {observation.shape[-1]}"
        )
    mirrored = observation.clone()
    # Angular velocity is an axial vector; gravity and linear commands are polar vectors.
    mirrored[..., 0:3] *= observation.new_tensor([-1.0, 1.0, -1.0])
    mirrored[..., 3:6] *= observation.new_tensor([1.0, -1.0, 1.0])
    mirrored[..., 6:9] *= observation.new_tensor([1.0, -1.0, -1.0])
    mirrored[..., 9:21] = _mirror_joint_data(env, observation[..., 9:21])
    mirrored[..., 21:33] = _mirror_joint_data(env, observation[..., 21:33])
    mirrored[..., 33:45] = _mirror_joint_data(env, observation[..., 33:45])
    if observation.shape[-1] == _POLICY_BASE_DIM + _GAIT_PHASE_DIM:
        policy_cfg = env.cfg.observations.policy
        has_velocity_feedback = getattr(policy_cfg, "base_lin_vel_xy", None) is not None
        has_gait_phase = getattr(policy_cfg, "gait_phase", None) is not None
        if has_velocity_feedback == has_gait_phase:
            raise ValueError(
                "A 47-D custom-dog policy must define exactly one of "
                "base_lin_vel_xy or gait_phase"
            )
        if has_velocity_feedback:
            # Body-frame planar velocity is a polar vector under sagittal
            # reflection: forward is unchanged and lateral changes sign.
            mirrored[..., 45:47] *= observation.new_tensor([1.0, -1.0])
        else:
            # A left-right reflection swaps the two trot diagonals, which is a
            # half-period shift of the global gait clock.
            mirrored[..., 45:47] = -observation[..., 45:47]
    return mirrored


def _mirror_critic_observation(env, observation: torch.Tensor) -> torch.Tensor:
    valid_dims = (_CRITIC_BASE_DIM, _CRITIC_BASE_DIM + _GAIT_PHASE_DIM)
    if observation.shape[-1] not in valid_dims:
        raise ValueError(
            f"Custom-dog critic symmetry expects one of {valid_dims}, got {observation.shape[-1]}"
        )
    mirrored = observation.clone()
    mirrored[..., 0:3] *= observation.new_tensor([1.0, -1.0, 1.0])
    mirrored[..., 3:6] *= observation.new_tensor([-1.0, 1.0, -1.0])
    mirrored[..., 6:9] *= observation.new_tensor([1.0, -1.0, 1.0])
    mirrored[..., 9:12] *= observation.new_tensor([1.0, -1.0, -1.0])
    mirrored[..., 12:24] = _mirror_joint_data(env, observation[..., 12:24])
    mirrored[..., 24:36] = _mirror_joint_data(env, observation[..., 24:36])
    mirrored[..., 36:48] = _mirror_joint_data(env, observation[..., 36:48])
    mirrored[..., 48:60] = _mirror_joint_data(env, observation[..., 48:60])
    if observation.shape[-1] == _CRITIC_BASE_DIM + _GAIT_PHASE_DIM:
        mirrored[..., 60:62] = -observation[..., 60:62]
    return mirrored


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append a sagittal-plane mirror of each observation/action sample."""

    unwrapped = env.unwrapped
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        if "policy" in obs.keys():
            obs_aug["policy"][:batch_size] = obs["policy"]
            obs_aug["policy"][batch_size:] = _mirror_policy_observation(unwrapped, obs["policy"])
        if "critic" in obs.keys():
            obs_aug["critic"][:batch_size] = obs["critic"]
            obs_aug["critic"][batch_size:] = _mirror_critic_observation(unwrapped, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty((batch_size * 2, actions.shape[-1]), device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _mirror_joint_data(unwrapped, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
