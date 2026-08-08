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
