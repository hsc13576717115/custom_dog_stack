"""Custom-dog task derived from the working Go2 locomotion task."""

from isaaclab.utils import configclass

from custom_dog_rl.assets.custom_dog import CUSTOM_DOG_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg as Go2RobotEnvCfg,
)


@configclass
class RobotEnvCfg(Go2RobotEnvCfg):
    """Go2 locomotion MDP using the custom robot asset."""

    def __post_init__(self):
        self.scene.robot = CUSTOM_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Headless WSL training does not need the remote USD velocity arrows.
        self.commands.base_velocity.debug_vis = False
        # These rendering-only assets require MDL/Vulkan, which Isaac Sim does
        # not expose under WSL. Physics material and terrain collision remain.
        self.scene.terrain.visual_material = None
        self.scene.sky_light = None
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_hip",
            ".*_thigh",
            ".*_calf",
        ]
        super().__post_init__()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotRobustEnvCfg(RobotEnvCfg):
    """Low-speed fine-tuning task with deployment-like reset perturbations."""

    def __post_init__(self):
        super().__post_init__()

        # The SDK2 hand-off reaches the policy with small body tilt and joint
        # sag. Train on those states instead of always resetting perfectly.
        self.events.reset_base.params["pose_range"].update(
            {"roll": (-0.2, 0.2), "pitch": (-0.25, 0.25)}
        )
        self.events.reset_robot_joints.params["position_range"] = (0.85, 1.15)

        self.commands.base_velocity.rel_standing_envs = 0.3
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # unitree_rl_lab exports limit_ranges to deploy.yaml. Keep those limits
        # equal to this task's actual training distribution.
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)

        # Keep this stage focused on stable low-speed behavior. A later stage
        # can restore the full velocity curriculum after sim2sim acceptance.
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.joint_vel.weight = -0.002
        self.rewards.action_rate.weight = -0.15


@configclass
class RobotRobustPlayEnvCfg(RobotRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
