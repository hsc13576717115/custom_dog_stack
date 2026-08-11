"""Custom-dog task derived from the working Go2 locomotion task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from custom_dog_rl.assets.custom_dog import (
    CUSTOM_DOG_CFG,
    CUSTOM_DOG_COMPACT_HIP_CFG,
)
from custom_dog_rl.tasks.locomotion import mdp as custom_mdp
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg as Go2RobotEnvCfg,
)


@configclass
class Go2ReferenceEnvCfg(Go2RobotEnvCfg):
    """Unmodified Go2 locomotion physics with WSL-safe rendering disabled."""

    def __post_init__(self):
        self.commands.base_velocity.debug_vis = False
        self.scene.terrain.visual_material = None
        self.scene.sky_light = None
        super().__post_init__()


@configclass
class Go2ReferencePlayEnvCfg(Go2ReferenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


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
class RobotSpeedOmniStyle45EnvCfg(RobotSpeedStraightEnvCfg):
    """Add omni commands and soft posture/contact style to the validated 45-D policy."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the old 45-D observation contract while exposing all joystick
        # axes. Forward commands retain the full 0-3 m/s envelope; low-speed
        # samples are oversampled so side stepping and turning do not erase the
        # existing fast forward gait.
        self.commands.base_velocity = custom_mdp.SpeedBandOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.55, 0.20, 0.15, 0.10),
            rel_standing_envs=0.05,
            rel_low_speed_forward=0.35,
            low_speed_range=(0.10, 0.60),
            lateral_min_fraction=0.45,
            yaw_min_fraction=0.45,
            debug_vis=False,
            ranges=custom_mdp.SpeedBandOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.10, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
            limit_ranges=custom_mdp.SpeedBandOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
        )

        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 1.25
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.track_velocity_components_relative_l1 = RewTerm(
            func=custom_mdp.track_velocity_components_relative_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.10, 0.15),
                "axis_weights": (1.0, 1.25, 1.0),
            },
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-0.5,
            params={"command_name": "base_velocity"},
        )

        # No phase observation and no body-height target. Hip splay is softened
        # as speed/side/yaw demand rises, rather than forced to zero.
        self.rewards.hip_outward_speed_style = RewTerm(
            func=custom_mdp.hip_outward_speed_style_l2,
            weight=-0.75,
            params={
                "standing_limit": 0.24,
                "walking_limit": 0.29,
                "high_speed_limit": 0.36,
                "walking_speed": 0.35,
                "high_speed": 1.50,
                "lateral_limit_gain": 0.12,
                "yaw_limit_gain": 0.05,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.base_height = None
        self.rewards.speed_adaptive_base_height = None

        feet_cfg = SceneEntityCfg(
            "robot",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        contact_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        self.rewards.foot_clearance_style = RewTerm(
            func=custom_mdp.foot_clearance_speed_style,
            weight=0.10,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.065,
                "std": 0.045,
                "command_name": "base_velocity",
            },
        )
        self.rewards.foot_soft_landing = RewTerm(
            func=custom_mdp.foot_soft_landing_l2,
            weight=-0.03,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "vertical_speed_std": 0.80,
                "command_name": "base_velocity",
            },
        )
        self.rewards.joint_pos.weight = -0.08
        self.rewards.action_rate.weight = -0.08
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.20


@configclass
class RobotSpeedOmniStyle45PlayEnvCfg(RobotSpeedOmniStyle45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxis45EnvCfg(RobotSpeedOmniStyle45EnvCfg):
    """Make lateral and yaw motion identifiable without dropping fast-forward samples."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.omni_mode_probabilities = (0.45, 0.25, 0.25, 0.05)
        self.commands.base_velocity.rel_low_speed_forward = 0.25
        self.commands.base_velocity.lateral_min_fraction = 0.55
        self.commands.base_velocity.yaw_min_fraction = 0.55

        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.5,
            1.25,
        )
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.lateral_command_progress = RewTerm(
            func=custom_mdp.lateral_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 0.55,
            },
        )
        self.rewards.yaw_command_progress = RewTerm(
            func=custom_mdp.yaw_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 0.80,
            },
        )


@configclass
class RobotSpeedOmniAxis45PlayEnvCfg(RobotSpeedOmniAxis45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniCurriculumEnvCfg(RobotEnvCfg):
    """Go2-style omni curriculum adapted to the custom robot geometry."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve the upstream Go2 curriculum: begin with small, learnable
        # planar commands and expand x/y together only after tracking succeeds.
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.10, 0.10)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.10, 0.10)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)

        # Learn the behavior before adding hard sim-to-real perturbations.
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.3)
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.push_robot = None

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-5.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.60,
                "crouch_full_speed": 1.20,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = RewTerm(
            func=custom_mdp.hip_nominal_l2,
            weight=-0.10,
            params={"target_positions": (-0.08, 0.08, -0.08, 0.08)},
        )
        self.rewards.hip_outward_excess = RewTerm(
            func=custom_mdp.hip_outward_excess_l2,
            weight=-3.0,
            params={
                "outward_limit": 0.22,
                "lateral_limit_gain": 0.30,
                "command_name": "base_velocity",
            },
        )
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotOmniCurriculumPlayEnvCfg(RobotOmniCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniFoundationEnvCfg(RobotGaitEnvCfg):
    """Moderate, identifiable command bands for learning the first omni gait."""

    def __post_init__(self):
        super().__post_init__()

        # Small +/-0.1 m/s samples let a stationary policy collect almost all
        # of the exponential tracking reward.  These pure/combined command
        # bands are large enough to require motion without demanding the final
        # deployment envelope before a lateral gait and turning gait exist.
        self.commands.base_velocity = custom_mdp.RecoveryVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(6.0, 6.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.35, 0.25, 0.25, 0.15),
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.2, 0.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.6, 0.6),
            ),
            limit_ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.8),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.6, 0.6),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None

        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.5
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_ang_vel_z.params["std"] = 0.5
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.10},
        )
        self.rewards.track_ang_vel_z_relative_l1 = RewTerm(
            func=custom_mdp.track_ang_vel_z_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.15},
        )
        self.rewards.joint_pos.weight = -0.10
        self.rewards.action_rate.weight = -0.05
        self.rewards.air_time_variance.weight = -0.10
        self.rewards.feet_slide.weight = -0.08
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-10.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.45,
                "crouch_full_speed": 1.20,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = RewTerm(
            func=custom_mdp.hip_nominal_l2,
            weight=-0.20,
            params={"target_positions": (-0.08, 0.08, -0.08, 0.08)},
        )
        self.rewards.hip_outward_excess = RewTerm(
            func=custom_mdp.hip_outward_excess_l2,
            weight=-5.0,
            params={
                "outward_limit": 0.22,
                "lateral_limit_gain": 0.30,
                "command_name": "base_velocity",
            },
        )


@configclass
class RobotOmniFoundationPlayEnvCfg(RobotOmniFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniSymmetryEnvCfg(RobotGaitEnvCfg):
    """Full planar command task using standard tracking rewards and LR symmetry."""

    def __post_init__(self):
        super().__post_init__()

        # Isolate forward, lateral and yaw behaviors often enough that a new
        # policy gets an identifiable learning signal before it sees combined
        # joystick commands. Final reverse/high-speed expansion is a later stage.
        self.commands.base_velocity = custom_mdp.RecoveryVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.25, 0.30, 0.25, 0.20),
            rel_standing_envs=0.05,
            debug_vis=False,
            ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.25, 0.65),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-0.8, 0.8),
            ),
            limit_ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.8),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-0.8, 0.8),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None

        # Follow the proven Go2/Go1 velocity task shape: exponential command
        # tracking plus light physical regularization. Avoid the aggressive
        # progress rewards that caused the lateral-bootstrap policy to lunge.
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.joint_pos.weight = -0.10
        self.rewards.action_rate.weight = -0.03
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.20
        self.rewards.air_time_variance.weight = -0.10
        self.rewards.feet_slide.weight = -0.08
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-8.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.45,
                "crouch_full_speed": 1.20,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = RewTerm(
            func=custom_mdp.hip_nominal_l2,
            weight=-0.20,
            params={"target_positions": (-0.08, 0.08, -0.08, 0.08)},
        )
        self.rewards.hip_outward_excess = RewTerm(
            func=custom_mdp.hip_outward_excess_l2,
            weight=-4.0,
            params={
                "outward_limit": 0.20,
                "lateral_limit_gain": 0.35,
                "command_name": "base_velocity",
            },
        )


@configclass
class RobotOmniSymmetryPlayEnvCfg(RobotOmniSymmetryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniPhaseEnvCfg(RobotOmniSymmetryEnvCfg):
    """Clock-conditioned trot task that removes the all-feet-down shortcut."""

    def __post_init__(self):
        super().__post_init__()

        gait_period = 0.70
        gait_phase_params = {
            "period": gait_period,
            "command_name": "base_velocity",
            "command_threshold": 0.10,
        }
        self.observations.policy.gait_phase = ObsTerm(
            func=custom_mdp.command_gait_phase,
            params=gait_phase_params,
        )
        self.observations.critic.gait_phase = ObsTerm(
            func=custom_mdp.command_gait_phase,
            params=gait_phase_params,
        )

        # FR+RL and FL+RR alternate.  Preserve this explicit body order so the
        # offsets cannot silently follow importer/link ordering.
        self.rewards.gait = RewTerm(
            func=mdp.feet_gait,
            weight=1.0,
            params={
                "period": gait_period,
                "offset": [0.0, 0.5, 0.5, 0.0],
                "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    preserve_order=True,
                ),
            },
        )
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.feet_air_time.params["threshold"] = 0.20
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.joint_pos.weight = -0.05


@configclass
class RobotOmniPhasePlayEnvCfg(RobotOmniPhaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLeggedGymPhaseTuneEnvCfg(RobotOmniPhaseEnvCfg):
    """Tune the proven phase gait using generic legged-gym objectives only."""

    def __post_init__(self):
        super().__post_init__()

        # Recovery is handled by the deployment state machine.  This task
        # contains only standing-start locomotion commands, with extra
        # combined samples to improve joystick operation after handoff.
        self.commands.base_velocity.omni_mode_probabilities = (0.20, 0.25, 0.20, 0.35)
        self.commands.base_velocity.rel_standing_envs = 0.05

        # Remove the previous machine-specific posture shaping.  Natural body
        # height and hip motion are left to the dynamics plus the generic
        # all-joint deviation, orientation, energy and smoothness penalties.
        self.rewards.speed_adaptive_base_height = None
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.joint_pos.weight = -0.20
        self.rewards.action_rate.weight = -0.05
        self.rewards.gait.weight = 0.60

        # Keep the legged-gym exponential rewards around the target and add
        # mild non-saturating tracking errors for combined commands.  No
        # progress/lunging reward is used here.
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.10},
        )
        self.rewards.track_ang_vel_z_relative_l1 = RewTerm(
            func=custom_mdp.track_ang_vel_z_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.15},
        )


@configclass
class RobotLeggedGymPhaseTunePlayEnvCfg(RobotLeggedGymPhaseTuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniRefineEnvCfg(RobotOmniPhaseEnvCfg):
    """Refine three-axis tracking with generic legged-gym regularization."""

    def __post_init__(self):
        super().__post_init__()

        # Get-up belongs to the deployment state machine.  Locomotion starts
        # upright and oversamples combined joystick commands, while retaining
        # pure-axis samples so every command remains identifiable.
        self.commands.base_velocity.omni_mode_probabilities = (0.20, 0.20, 0.20, 0.40)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.80)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.40, 0.40)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.80, 0.80)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.80)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.40, 0.40)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.80, 0.80)

        # No custom hip comfort zone or commanded body-height curve is used.
        # The upstream all-joint posture term leaves motion available when its
        # tracking benefit outweighs the generic deviation cost, and therefore
        # also permits the policy to crouch naturally at higher speed.
        self.rewards.speed_adaptive_base_height = None
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.joint_pos.weight = -0.70
        self.rewards.action_rate.weight = -0.10
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.20
        self.rewards.air_time_variance.weight = -0.20
        self.rewards.gait.weight = 0.40

        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.track_velocity_components_relative_l1 = RewTerm(
            func=custom_mdp.track_velocity_components_relative_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.10, 0.15),
            },
        )


@configclass
class RobotOmniRefinePlayEnvCfg(RobotOmniRefineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniRefineCompactHipEnvCfg(RobotOmniRefineEnvCfg):
    """Use compact +/-0.05 rad nominal hips with the measured 25/0.5 loop."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CUSTOM_DOG_COMPACT_HIP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # After the nominal-pose transition is stable, make the joystick axes
        # individually identifiable.  These are still generic velocity errors;
        # no hip angle, foot width or body-height target is introduced.
        self.commands.base_velocity.omni_mode_probabilities = (0.25, 0.25, 0.25, 0.25)
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_velocity_components_relative_l1.weight = -4.0


@configclass
class RobotOmniRefineCompactHipPlayEnvCfg(RobotOmniRefineCompactHipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactHeightPolishEnvCfg(RobotOmniRefineCompactHipEnvCfg):
    """Polish the accepted compact policy around standing height on its validated envelope."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.50)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.50, 0.50)

        # Both terms cover the complete body/joint set.  They impose no hip
        # comfort interval and remain soft so tracking can pay for a small
        # stability crouch or a larger transient joint excursion.
        self.rewards.joint_pos = RewTerm(
            func=custom_mdp.joint_deviation_l2,
            weight=-0.35,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
        )
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-200.0,
            params={"target_height": 0.29},
        )
        self.rewards.action_rate.weight = -0.12
        self.rewards.gait.weight = 0.40


@configclass
class RobotCompactHeightPolishPlayEnvCfg(RobotCompactHeightPolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactOmniAdaptiveHeightEnvCfg(RobotCompactHeightPolishEnvCfg):
    """Improve lateral/combined commands while allowing a small speed crouch."""

    def __post_init__(self):
        super().__post_init__()

        # Keep every deployed axis in the training distribution. Lateral and
        # combined samples are oversampled because they are the weak axes of
        # the accepted policy, not because the command envelope is narrowed.
        self.commands.base_velocity.omni_mode_probabilities = (0.20, 0.35, 0.20, 0.25)
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.05, 0.05, 0.10),
                "axis_weights": (1.0, 2.25, 1.0),
            }
        )
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.gait.weight = 0.25
        self.rewards.base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-200.0,
            params={
                "standing_height": 0.295,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.15,
                "crouch_full_speed": 0.65,
                "yaw_speed_scale": 0.30,
                "command_name": "base_velocity",
            },
        )


@configclass
class RobotCompactOmniAdaptiveHeightPlayEnvCfg(RobotCompactOmniAdaptiveHeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniStabilityStage1EnvCfg(RobotCompactHeightPolishEnvCfg):
    """Conservatively expand the accepted controller to a wider omni envelope."""

    def __post_init__(self):
        super().__post_init__()

        # Expand one step beyond the frozen candidate instead of exposing the
        # policy to the final high-speed range in a single PPO update.  Pure
        # forward samples remain the largest group so lateral refinement cannot
        # erase the already-valid forward gait.
        self.commands.base_velocity.omni_mode_probabilities = (0.35, 0.25, 0.20, 0.20)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.60)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.60, 0.60)

        # Height is deliberately unconstrained: the controller may crouch when
        # that improves stability.  These remaining terms are generic velocity,
        # whole-body posture, orientation, contact and smooth-action objectives.
        self.rewards.base_height = None
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.05, 0.05, 0.10),
                "axis_weights": (1.0, 1.5, 1.0),
            }
        )
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.action_rate.weight = -0.12
        self.rewards.gait.weight = 0.35


@configclass
class RobotOmniStabilityStage1PlayEnvCfg(RobotOmniStabilityStage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniStabilityStage2EnvCfg(RobotOmniStabilityStage1EnvCfg):
    """Bridge the measured 0.56/0.58 m/s forward-stability boundary."""

    def __post_init__(self):
        super().__post_init__()

        # Concentrate forward and combined samples around the MuJoCo failure
        # boundary.  Pure lateral and yaw modes remain present so the policy
        # does not forget either joystick axis while learning the bridge.
        self.commands.base_velocity.omni_mode_probabilities = (0.55, 0.20, 0.15, 0.10)
        self.commands.base_velocity.ranges.lin_vel_x = (0.52, 0.62)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.62)

        # A wide exponential kernel plus signed command progress keeps a useful
        # gradient when the current policy falls before attaining the command.
        # Height remains unconstrained; orientation and termination terms are
        # the stability criteria.
        self.rewards.track_lin_vel_xy.weight = 6.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.45
        self.rewards.track_velocity_components_relative_l1.weight = -4.5
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.25,
            1.5,
            1.0,
        )
        self.rewards.planar_command_progress = RewTerm(
            func=custom_mdp.planar_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.45,
                "max_progress": 0.80,
            },
        )
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.termination_penalty.weight = -300.0
        self.rewards.action_rate.weight = -0.10
        self.rewards.gait.weight = 0.40


@configclass
class RobotOmniStabilityStage2PlayEnvCfg(RobotOmniStabilityStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniStabilityStage3EnvCfg(RobotOmniStabilityStage2EnvCfg):
    """Bridge lateral tracking without discarding the forward/yaw behaviors."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.omni_mode_probabilities = (0.30, 0.45, 0.15, 0.10)
        self.commands.base_velocity.ranges.lin_vel_x = (0.54, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.42, 0.42)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.42, 0.42)
        self.commands.base_velocity.lateral_min_fraction = 0.75

        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.25,
            2.0,
            1.0,
        )
        self.rewards.lateral_command_progress = RewTerm(
            func=custom_mdp.lateral_command_progress,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.25,
                "max_progress": 0.60,
            },
        )


@configclass
class RobotOmniStabilityStage3PlayEnvCfg(RobotOmniStabilityStage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotNaturalGaitEnvCfg(RobotOmniStabilityStage3EnvCfg):
    """Polish model800 with low-speed walk, high-speed trot and soft foot style."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve all three joystick axes and oversample the 0.10-0.60 m/s
        # band where a sequential walk is preferable. The remaining forward
        # samples cover the full 3 m/s policy contract.
        self.commands.base_velocity = custom_mdp.SpeedBandOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.45, 0.20, 0.15, 0.20),
            rel_standing_envs=0.08,
            rel_low_speed_forward=0.55,
            low_speed_range=(0.10, 0.60),
            lateral_min_fraction=0.50,
            yaw_min_fraction=0.50,
            debug_vis=False,
            ranges=custom_mdp.SpeedBandOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.10, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
            limit_ranges=custom_mdp.SpeedBandOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
        )

        # Keep the 47-D phase-conditioned observation and change only the
        # contact objective: sequential walk below 0.75 m/s, smooth transition,
        # diagonal trot from 1.25 m/s upward.
        self.rewards.gait = RewTerm(
            func=custom_mdp.speed_adaptive_feet_gait,
            weight=0.35,
            params={
                "period": 0.70,
                "threshold": 0.55,
                "walk_speed": 0.75,
                "trot_speed": 1.25,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    preserve_order=True,
                ),
            },
        )

        # No body-height target. Hip freedom grows with command speed and with
        # lateral/yaw demand; only excessive outward splay is softly penalized.
        self.rewards.base_height = None
        self.rewards.speed_adaptive_base_height = None
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.hip_outward_speed_style = RewTerm(
            func=custom_mdp.hip_outward_speed_style_l2,
            weight=-1.0,
            params={
                "standing_limit": 0.28,
                "walking_limit": 0.34,
                "high_speed_limit": 0.42,
                "walking_speed": 0.35,
                "high_speed": 1.50,
                "lateral_limit_gain": 0.10,
                "yaw_limit_gain": 0.04,
                "command_name": "base_velocity",
            },
        )
        feet_cfg = SceneEntityCfg(
            "robot",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        contact_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        self.rewards.foot_clearance_style = RewTerm(
            func=custom_mdp.foot_clearance_speed_style,
            weight=0.15,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.065,
                "std": 0.045,
                "command_name": "base_velocity",
            },
        )
        self.rewards.foot_soft_landing = RewTerm(
            func=custom_mdp.foot_soft_landing_l2,
            weight=-0.04,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "vertical_speed_std": 0.80,
                "command_name": "base_velocity",
            },
        )
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.20
        self.rewards.action_rate.weight = -0.10


@configclass
class RobotNaturalGaitPlayEnvCfg(RobotNaturalGaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniStabilityStage4EnvCfg(RobotOmniStabilityStage3EnvCfg):
    """Refine mixed vx/vy/wz commands after the single-axis bridge."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.omni_mode_probabilities = (0.25, 0.20, 0.15, 0.40)
        self.commands.base_velocity.ranges.lin_vel_x = (0.50, 0.62)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.lateral_min_fraction = 0.65

        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            2.5,
            1.0,
        )
        self.rewards.track_velocity_components_progress = RewTerm(
            func=custom_mdp.track_velocity_components_progress,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "command_min": (0.20, 0.15, 0.20),
                "axis_weights": (1.0, 2.5, 1.0),
                "max_progress": (0.80, 0.50, 0.80),
            },
        )
        self.rewards.planar_command_progress.weight = 1.0


@configclass
class RobotOmniStabilityStage4PlayEnvCfg(RobotOmniStabilityStage4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniStabilityStage5EnvCfg(RobotOmniStabilityStage4EnvCfg):
    """Reduce uncommanded-axis coupling inside the accepted speed envelope."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.omni_mode_probabilities = (0.20, 0.30, 0.30, 0.20)
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.55)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.62)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.30, 0.30)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.30, 0.30)

        # Unlike active-axis-only relative errors, these generic L2 terms also
        # penalize forward/sideways creep when the corresponding command is zero.
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-4.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_velocity_components_progress.weight = 0.75
        self.rewards.lateral_command_progress.weight = 1.5
        self.rewards.planar_command_progress.weight = 0.5


@configclass
class RobotOmniStabilityStage5PlayEnvCfg(RobotOmniStabilityStage5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniPolishEnvCfg(RobotOmniRefineCompactHipEnvCfg):
    """Conservative all-axis polish without robot-specific posture targets."""

    def __post_init__(self):
        super().__post_init__()

        # Lateral motion is mechanically harder than forward/yaw motion.  Give
        # it more samples and gradient without changing the command envelope or
        # introducing a hip-angle, foot-width or body-height objective.
        self.commands.base_velocity.omni_mode_probabilities = (0.25, 0.35, 0.20, 0.20)
        self.rewards.track_lin_vel_xy.params["std"] = 0.30
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.75,
            1.0,
        )

        # These regularizers are generic across all 12 joints and all actions.
        # The compact default is a neutral action offset, not a hard hip target.
        self.rewards.joint_pos.weight = -0.90
        self.rewards.action_rate.weight = -0.12


@configclass
class RobotOmniPolishPlayEnvCfg(RobotOmniPolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactFoundationEnvCfg(RobotOmniPolishEnvCfg):
    """Learn a stable compact phase gait from scratch on gentle three-axis commands."""

    def __post_init__(self):
        super().__post_init__()

        # Stage one must first retain the compact standing equilibrium.  Give
        # forward motion the clearest signal, but keep enough lateral and yaw
        # samples that the 47-D policy can later expand without an axis reset.
        # The final joystick envelope belongs to the subsequent polish stage.
        self.commands.base_velocity.omni_mode_probabilities = (0.55, 0.15, 0.20, 0.10)
        self.commands.base_velocity.rel_standing_envs = 0.20
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.30)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.30, 0.30)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.30)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.30, 0.30)

        # Wide tracking kernels and light generic regularization avoid the
        # early all-feet-down/fall trade-off seen with the first scratch run.
        # No hip-angle or body-height target is introduced.
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.track_velocity_components_relative_l1.weight = -1.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.25,
            1.0,
        )
        self.rewards.joint_pos.weight = -0.40
        self.rewards.action_rate.weight = -0.05
        self.rewards.gait.weight = 0.20
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.air_time_variance.weight = -0.10


@configclass
class RobotCompactFoundationPlayEnvCfg(RobotCompactFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactMotionEnvCfg(RobotCompactFoundationEnvCfg):
    """Release the stable compact foundation into a low-speed phase gait."""

    def __post_init__(self):
        super().__post_init__()

        # A stable foundation can tolerate a clearer non-zero command.  Keep
        # pure-axis samples dominant before adding the final joystick envelope.
        self.commands.base_velocity.omni_mode_probabilities = (0.60, 0.15, 0.20, 0.05)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.18, 0.18)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.40)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.18, 0.18)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.35, 0.35)

        # The foundation learned that standing still is safe.  A narrower
        # exponential kernel plus non-saturating generic errors now makes
        # actual motion necessary, without prescribing hip angles or height.
        self.rewards.track_lin_vel_xy.weight = 6.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.15
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.20
        self.rewards.track_velocity_components_relative_l1.weight = -3.0
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.gait.weight = 0.50
        self.rewards.feet_air_time.weight = 0.20


@configclass
class RobotCompactMotionPlayEnvCfg(RobotCompactMotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactForwardBootstrapEnvCfg(RobotCompactFoundationEnvCfg):
    """Bootstrap a forward contact-switching gait from the stable foundation."""

    def __post_init__(self):
        super().__post_init__()

        # With only 64 CPU environments, discovering three independent motion
        # axes at once is sample-starved.  First make forward motion observable;
        # lateral and yaw are introduced after contact switching appears.
        self.commands.base_velocity.omni_mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.50)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

        self.rewards.track_lin_vel_xy.weight = 6.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        # This bounded, command-conditioned term is only a discovery aid.  It
        # cannot reward sideways lunging and is removed in the polish stage.
        self.rewards.planar_command_progress = RewTerm(
            func=custom_mdp.planar_command_progress,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.20,
                "max_progress": 0.60,
            },
        )
        self.rewards.gait.weight = 0.50
        self.rewards.feet_air_time.weight = 0.20
        self.rewards.joint_pos.weight = -0.30
        self.rewards.action_rate.weight = -0.04
        self.rewards.termination_penalty.weight = -300.0


@configclass
class RobotCompactForwardBootstrapPlayEnvCfg(RobotCompactForwardBootstrapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactPhaseScratchEnvCfg(RobotOmniPhaseEnvCfg):
    """Train the proven phase task from scratch around the compact neutral pose."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CUSTOM_DOG_COMPACT_HIP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Match the command identifiability that produced the existing stable
        # omni gait, while learning the compact neutral pose from iteration 0.
        self.commands.base_velocity.omni_mode_probabilities = (0.25, 0.30, 0.25, 0.20)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.ang_vel_z = (-0.40, 0.40)

        # Keep the reward shape that produced the existing phase gait, but
        # remove its earlier robot-specific hip/height shaping.  The remaining
        # terms are generic exponential tracking, phase contact, all-joint
        # deviation, orientation, energy and action smoothness.
        self.rewards.speed_adaptive_base_height = None
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.joint_pos.weight = -0.10
        self.rewards.action_rate.weight = -0.03
        self.rewards.gait.weight = 1.0


@configclass
class RobotCompactPhaseScratchPlayEnvCfg(RobotCompactPhaseScratchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactPhasePostureEnvCfg(RobotCompactPhaseScratchEnvCfg):
    """Preserve standing height and the full neutral pose while the gait matures."""

    def __post_init__(self):
        super().__post_init__()

        # These are framework-standard whole-body terms: the same all-joint
        # penalty/weight used by the Go2 task and Isaac Lab's generic height
        # objective.  They do not single out hip joints or prescribe foot width.
        self.rewards.joint_pos.weight = -0.70
        self.rewards.action_rate.weight = -0.10
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-10.0,
            params={"target_height": 0.28},
        )
        self.rewards.gait.weight = 0.60


@configclass
class RobotCompactPhasePosturePlayEnvCfg(RobotCompactPhasePostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotCompactPhaseHeightEnvCfg(RobotCompactPhasePostureEnvCfg):
    """Keep the learned gait near its standing height without joint-specific shaping."""

    def __post_init__(self):
        super().__post_init__()

        # A fixed whole-body target still lets tracking rewards pay for a small
        # stability crouch, while making a persistent deep crouch uneconomical.
        self.rewards.base_height.weight = -60.0
        self.rewards.base_height.params["target_height"] = 0.29
        self.rewards.joint_pos.weight = -1.20
        self.rewards.gait.weight = 0.50


@configclass
class RobotCompactPhaseHeightPlayEnvCfg(RobotCompactPhaseHeightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniPhasePostureEnvCfg(RobotOmniPhaseEnvCfg):
    """Fine-tune the learned omni gait for compact hips and usable body height."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.gait.weight = 0.60
        self.rewards.speed_adaptive_base_height.weight = -60.0
        self.rewards.speed_adaptive_base_height.params.update(
            {
                "standing_height": 0.30,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.35,
                "crouch_full_speed": 1.20,
                "yaw_speed_scale": 0.40,
            }
        )
        self.rewards.hip_nominal.weight = -1.0
        self.rewards.hip_nominal.params["target_positions"] = (-0.05, 0.05, -0.05, 0.05)
        self.rewards.hip_outward_excess.weight = -20.0
        self.rewards.hip_outward_excess.params.update(
            {
                "outward_limit": 0.15,
                "lateral_limit_gain": 0.20,
            }
        )
        self.rewards.joint_pos.weight = -0.10


@configclass
class RobotOmniPhasePosturePlayEnvCfg(RobotOmniPhasePostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLeggedGymStage1EnvCfg(RobotEnvCfg):
    """Use the upstream Go2/legged-gym reward on identifiable pure-axis commands."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = custom_mdp.RecoveryVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.40, 0.30, 0.30, 0.0),
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.45),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.40, 0.40),
            ),
            limit_ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.45),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.40, 0.40),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None

        # Keep the upstream Go2/legged-gym reward terms. The Go2 tracking
        # kernels are too broad for this stage's low commands: at 0.3 m/s a
        # stationary policy otherwise receives about 70% of the maximum
        # tracking reward and converges to standing. Tighten and scale only the
        # two task rewards; all posture and regularization terms remain intact.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25

        # Simplify randomization until the basic flat-ground behaviors have
        # been learned.
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.3)
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.push_robot = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotLeggedGymStage1PlayEnvCfg(RobotLeggedGymStage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLeggedGymStage2EnvCfg(RobotLeggedGymStage1EnvCfg):
    """Adapt a proven forward policy to pure-axis omni commands."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.omni_mode_probabilities = (0.25, 0.40, 0.35, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.70, 0.70)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.80)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.40, 0.40)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.80, 0.80)

        # The exponential rewards retain the legged-gym objective near the
        # target. Non-saturating, command-conditioned errors and progress give
        # PPO a usable gradient when a forward-only checkpoint initially does
        # not respond to lateral or yaw commands.
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_relative_l1,
            weight=-1.5,
            params={"command_name": "base_velocity", "command_min": 0.10},
        )
        self.rewards.track_ang_vel_z_relative_l1 = RewTerm(
            func=custom_mdp.track_ang_vel_z_relative_l1,
            weight=-1.5,
            params={"command_name": "base_velocity", "command_min": 0.15},
        )
        self.rewards.planar_command_progress = RewTerm(
            func=custom_mdp.planar_command_progress,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.10,
                "max_progress": 1.0,
            },
        )
        self.rewards.yaw_command_progress = RewTerm(
            func=custom_mdp.yaw_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 1.2,
            },
        )
        self.rewards.lateral_command_progress = RewTerm(
            func=custom_mdp.lateral_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 0.8,
            },
        )

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.rewards.joint_pos.weight = -0.10
        self.rewards.action_rate.weight = -0.05
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.air_time_variance.weight = -0.10
        self.rewards.feet_slide.weight = -0.08


@configclass
class RobotLeggedGymStage2PlayEnvCfg(RobotLeggedGymStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLeggedGymStage2LateralEnvCfg(RobotLeggedGymStage2EnvCfg):
    """Bootstrap real side stepping before mixing all three command axes."""

    def __post_init__(self):
        super().__post_init__()

        # A forward-only checkpoint can satisfy mixed-command batches by
        # standing and changing its leg pose.  Concentrate the batch on clear
        # pure-y commands until measured lateral velocity becomes non-zero.
        self.commands.base_velocity.omni_mode_probabilities = (0.15, 0.80, 0.05, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.50, 0.50)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.80)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.50, 0.50)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.80, 0.80)

        # These terms depend only on commanded and measured velocity.  There
        # is deliberately no robot-specific hip-angle or body-height target.
        self.rewards.track_lin_vel_xy.weight = 7.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.track_lin_vel_xy_l2.weight = -4.0
        self.rewards.track_lin_vel_xy_relative_l1.weight = -8.0
        self.rewards.planar_command_progress.weight = 2.0
        self.rewards.lateral_command_progress.weight = 10.0
        self.rewards.lateral_command_progress.params["max_progress"] = 0.70

        # Retain generic legged-gym posture and smoothness regularization so
        # the new behavior cannot improve reward merely by holding a splayed
        # static pose.
        self.rewards.joint_pos.weight = -0.20
        self.rewards.action_rate.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -3.0


@configclass
class RobotLeggedGymStage2LateralPlayEnvCfg(RobotLeggedGymStage2LateralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLateralBootstrapEnvCfg(RobotOmniFoundationEnvCfg):
    """Force an identifiable side-step before adding the full command mixture."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.omni_mode_probabilities = (0.20, 0.80, 0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.10, 0.10)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.10, 0.10)

        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy_l2.weight = -2.0
        self.rewards.track_lin_vel_xy_relative_l1.weight = -3.0
        self.rewards.track_ang_vel_z_l2.weight = -2.0
        self.rewards.track_ang_vel_z_relative_l1 = None
        self.rewards.lateral_command_progress = RewTerm(
            func=custom_mdp.lateral_command_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 0.6,
            },
        )
        self.rewards.hip_outward_excess.params["outward_limit"] = 0.25


@configclass
class RobotLateralBootstrapPlayEnvCfg(RobotLateralBootstrapEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotLateralBootstrapV2EnvCfg(RobotLateralBootstrapEnvCfg):
    """Focused lateral bootstrap that rejects the stationary posture shortcut."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.omni_mode_probabilities = (0.10, 0.90, 0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.05

        self.rewards.track_lin_vel_xy.weight = 6.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.20
        self.rewards.track_lin_vel_xy_l2.weight = -4.0
        self.rewards.track_lin_vel_xy_relative_l1.weight = -6.0
        self.rewards.lateral_command_progress.weight = 8.0
        self.rewards.lateral_command_progress.params["max_progress"] = 0.50

        self.rewards.termination_penalty.weight = -300.0
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.action_rate.weight = -0.03
        self.rewards.joint_pos.weight = -0.05
        self.rewards.feet_slide.weight = -0.02
        self.rewards.speed_adaptive_base_height.weight = -80.0
        self.rewards.hip_nominal.weight = -1.0
        self.rewards.hip_outward_excess.weight = -12.0
        self.rewards.hip_outward_excess.params["outward_limit"] = 0.18
        self.rewards.hip_outward_excess.params["lateral_limit_gain"] = 0.25


@configclass
class RobotLateralBootstrapV2PlayEnvCfg(RobotLateralBootstrapV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniPretrainEnvCfg(RobotGaitEnvCfg):
    """Standing-start omni-directional pretraining before prone recovery."""

    def __post_init__(self):
        super().__post_init__()

        # Learn side stepping and turning from a known stable stance first.
        # Pure lateral/yaw samples are deliberately large enough to make a
        # stationary policy visibly worse than moving.
        self.commands.base_velocity = custom_mdp.RecoveryVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=True,
            omni_mode_probabilities=(0.10, 0.50, 0.30, 0.10),
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.8),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.8, 0.8),
            ),
            limit_ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.8),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.8, 0.8),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_relative_l1,
            weight=-1.5,
            params={"command_name": "base_velocity", "command_min": 0.10},
        )
        self.rewards.track_ang_vel_z_relative_l1 = RewTerm(
            func=custom_mdp.track_ang_vel_z_relative_l1,
            weight=-1.5,
            params={"command_name": "base_velocity", "command_min": 0.15},
        )
        self.rewards.planar_command_progress = RewTerm(
            func=custom_mdp.planar_command_progress,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.10,
                "max_progress": 1.0,
            },
        )
        self.rewards.yaw_command_progress = RewTerm(
            func=custom_mdp.yaw_command_progress,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.15,
                "max_progress": 1.2,
            },
        )
        self.rewards.lateral_command_progress = RewTerm(
            func=custom_mdp.lateral_command_progress,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.20,
                "max_progress": 0.8,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.joint_pos.weight = -0.05
        self.rewards.action_rate.weight = -0.03
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.air_time_variance.weight = -0.10
        self.rewards.feet_slide.weight = -0.08
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-8.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.35,
                "crouch_full_speed": 1.0,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = RewTerm(
            func=custom_mdp.hip_nominal_l2,
            weight=-0.20,
            params={"target_positions": (-0.08, 0.08, -0.08, 0.08)},
        )
        self.rewards.hip_outward_excess = RewTerm(
            func=custom_mdp.hip_outward_excess_l2,
            weight=-4.0,
            params={
                "outward_limit": 0.20,
                "lateral_limit_gain": 0.35,
                "command_name": "base_velocity",
            },
        )


@configclass
class RobotOmniPretrainPlayEnvCfg(RobotOmniPretrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotRecoveryOmniEnvCfg(RobotSpeedEnvCfg):
    """Recover from prone, then learn moderate omni-directional commands."""

    def __post_init__(self):
        super().__post_init__()

        # Train the measured folded state directly.  The root starts low enough
        # to contact flat ground, so base contact must not end a recovery
        # episode before the policy has a chance to unfold and lift the trunk.
        self.events.reset_base = EventTerm(
            func=custom_mdp.reset_recovery_or_standing,
            mode="reset",
            params={
                "prone_probability": 0.35,
                "prone_root_height": 0.07,
                "prone_thigh_position": 1.2391837689,  # 71 deg
                "prone_calf_position": -2.8099800957,  # -161 deg
                "prone_joint_noise": 0.02,
                "standing_joint_noise": 0.04,
                "root_xy_range": (-0.25, 0.25),
                "yaw_range": (-3.14, 3.14),
            },
        )
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.events.scale_body_mass = None
        self.events.randomize_actuator_gains = None
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 0.3)

        # Stage one exposes all three deployment commands without combining
        # recovery with the final high-speed range in the same PPO update.
        self.commands.base_velocity = custom_mdp.RecoveryVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(6.0, 6.0),
            recovery_duration_s=1.0,
            omni_mixture=True,
            rel_standing_envs=0.15,
            debug_vis=False,
            ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.5, 0.5),
            ),
            limit_ranges=custom_mdp.RecoveryVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.5, 0.5),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        # Preserve recovery episodes with a level trunk, then keep a compact
        # stance while allowing the body to crouch progressively at speed.
        self.terminations.base_contact = None
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_ang_vel_z.params["std"] = 0.4
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_lin_vel_xy_relative_l1 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.05},
        )
        self.rewards.track_ang_vel_z_relative_l1 = RewTerm(
            func=custom_mdp.track_ang_vel_z_relative_l1,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_min": 0.05},
        )
        self.rewards.flat_orientation_l2.weight = -4.0
        self.rewards.joint_pos.weight = -0.08
        self.rewards.action_rate.weight = -0.06
        self.rewards.dof_pos_limits.weight = -2.0
        self.rewards.recovery_upright_height = RewTerm(
            func=custom_mdp.recovery_upright_height,
            weight=3.0,
            params={"prone_height": 0.07, "standing_height": 0.30, "prone_only": True},
        )
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-15.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.255,
                "crouch_start_speed": 0.8,
                "crouch_full_speed": 2.5,
                "command_name": "base_velocity",
            },
        )
        self.rewards.hip_nominal = RewTerm(
            func=custom_mdp.hip_nominal_l2,
            weight=-0.6,
            params={"target_positions": (-0.08, 0.08, -0.08, 0.08)},
        )
        self.rewards.hip_outward_excess = RewTerm(
            func=custom_mdp.hip_outward_excess_l2,
            weight=-8.0,
            params={
                "outward_limit": 0.20,
                "lateral_limit_gain": 0.35,
                "command_name": "base_velocity",
            },
        )


@configclass
class RobotRecoveryOmniPlayEnvCfg(RobotRecoveryOmniEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.events.reset_base.params["prone_probability"] = 1.0
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotRecoveryOmniFullEnvCfg(RobotRecoveryOmniEnvCfg):
    """Final command-range expansion after recovery and omni tracking converge."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.omni_mixture = False
        self.events.reset_base.params["prone_probability"] = 0.15
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.6, 2.5)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.8, 0.8)


@configclass
class RobotRecoveryOmniFullPlayEnvCfg(RobotRecoveryOmniFullEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.events.reset_base.params["prone_probability"] = 1.0
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
