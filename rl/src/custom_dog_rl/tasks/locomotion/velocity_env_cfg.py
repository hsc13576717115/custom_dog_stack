"""Custom-dog task derived from the working Go2 locomotion task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from custom_dog_rl.assets.custom_dog import CUSTOM_DOG_CFG
from custom_dog_rl.tasks.locomotion import mdp as custom_mdp
from unitree_rl_lab.tasks.locomotion import mdp
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
class RobotGaitEnvCfg(RobotEnvCfg):
    """Flat-ground stage for discovering a natural low-speed gait."""

    def __post_init__(self):
        super().__post_init__()

        # Most environments receive a clear forward command. A small standing
        # subset preserves the zero-command behavior without dominating PPO.
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.15, 0.15)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.15, 0.15)

        # Learn locomotion before robustness. Keep enough variation to avoid a
        # single exact simulator, but remove disturbances that favor standing.
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.3)
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.push_robot = None

        # Make forward tracking worth moving for while retaining basic safety
        # regularization. No explicit gait phase or feet_gait reward is used.
        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.5
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.joint_pos.weight = -0.2
        self.rewards.action_rate.weight = -0.05
        # The inherited air-time term is negative until a foot exceeds its
        # threshold. Enable foot-style shaping only after locomotion emerges.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.feet_slide.weight = -0.05

        # This stage has a fixed command distribution and flat terrain. Command
        # expansion and stronger domain randomization belong to the next stage.
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotGaitPlayEnvCfg(RobotGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedEnvCfg(RobotGaitEnvCfg):
    """Forward-speed curriculum that expands from standing to 3 m/s."""

    def __post_init__(self):
        super().__post_init__()

        # Cover zero and low speeds continuously. The upper bound grows only
        # after the current range reaches the tracking-reward threshold.
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.15, 0.15)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 3.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.15, 0.15)

        # Randomize the quantities that differ most across PhysX, MuJoCo and
        # the real motor loop, while keeping this stage focused on flat ground.
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.2)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.08)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.scale_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.7, 1.5),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.5, 0.5)
        self.events.push_robot = None

        # Tighten velocity tracking after gait discovery, but retain enough
        # freedom for stride length and frequency to increase with speed.
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 0.75
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.joint_pos.weight = -0.1
        self.rewards.action_rate.weight = -0.05
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.air_time_variance.weight = -0.1
        self.rewards.feet_slide.weight = -0.1

        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=custom_mdp.forward_vel_cmd_levels,
            params={"increment": 0.25, "success_threshold": 0.75},
        )


@configclass
class RobotSpeedPlayEnvCfg(RobotSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedHighEnvCfg(RobotSpeedEnvCfg):
    """Continuation stage that consolidates 0-2 m/s before expanding to 3 m/s."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 1.0
        self.curriculum.lin_vel_cmd_levels = CurrTerm(
            func=custom_mdp.forward_vel_cmd_levels,
            params={"increment": 0.25, "success_threshold": 0.7},
        )


@configclass
class RobotSpeedHighPlayEnvCfg(RobotSpeedHighEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedFullEnvCfg(RobotSpeedHighEnvCfg):
    """Final fixed-distribution stage for continuous 0-3 m/s tracking."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
        self.curriculum.lin_vel_cmd_levels = None


@configclass
class RobotSpeedFullPlayEnvCfg(RobotSpeedFullEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedStraightEnvCfg(RobotSpeedFullEnvCfg):
    """Straight-line final tune with zero lateral and yaw commands."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_ang_vel_z.params["std"] = 0.1
        # Preserve a non-saturating yaw-error gradient during the dedicated
        # straight-line tune without changing the general speed tasks.
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )


@configclass
class RobotSpeedStraightPlayEnvCfg(RobotSpeedStraightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedBalancedTuneEnvCfg(RobotSpeedStraightEnvCfg):
    """Experimental full-range tune that oversamples the low-speed band."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.MixedForwardVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.05,
            rel_low_speed_envs=0.5,
            low_speed_range=(0.1, 0.5),
            debug_vis=False,
            ranges=custom_mdp.MixedForwardVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
            # Keep the policy/deployment command contract at the full range.
            limit_ranges=custom_mdp.MixedForwardVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
        self.rewards.track_lin_vel_xy_low_speed_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_low_speed_relative_l1,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.1,
                "command_max": 0.5,
            },
        )


@configclass
class RobotSpeedBalancedTunePlayEnvCfg(RobotSpeedBalancedTuneEnvCfg):
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
