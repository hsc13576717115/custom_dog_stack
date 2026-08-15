"""Custom-dog task derived from the working Go2 locomotion task."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from custom_dog_rl.assets.custom_dog import (
    CUSTOM_DOG_CFG,
    CUSTOM_DOG_COMPACT_HIP_CFG,
    CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG,
)
from custom_dog_rl.tasks.locomotion import mdp as custom_mdp
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ObservationsCfg as Go2ObservationsCfg,
    RobotEnvCfg as Go2RobotEnvCfg,
)


CUSTOM_DOG_TERRAIN_T0_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.40),
        "mild_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.30,
            noise_range=(0.005, 0.025),
            noise_step=0.005,
            border_width=0.25,
        ),
        "mild_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.12),
            platform_width=2.0,
            border_width=0.25,
        ),
        "mild_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.12),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


CUSTOM_DOG_TERRAIN_T1_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.40),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.05),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.22),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.22),
            platform_width=2.0,
            border_width=0.25,
        ),
        "low_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "low_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class PrivilegedVelocityTeacherObsCfg(Go2ObservationsCfg.PolicyCfg):
    """Teacher contract: deployable 45-D state plus true planar velocity."""

    base_lin_vel_xy = ObsTerm(
        func=custom_mdp.base_lin_vel_xy,
        scale=1.0,
        clip=(-2.0, 2.0),
    )


@configclass
class PrivilegedClosedLoopTeacherObsCfg(Go2ObservationsCfg.PolicyCfg):
    """Exact 51-D closed-loop teacher: base state, trot clock, and true vx/vy."""

    trot_clock = ObsTerm(
        func=custom_mdp.command_trot_clock,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.03,
            "yaw_command_threshold": 0.05,
            "min_frequency": 1.4,
            "max_frequency": 3.2,
            "full_speed": 3.0,
            "yaw_speed_scale": 0.35,
        },
    )
    base_lin_vel_xy = ObsTerm(
        func=custom_mdp.base_lin_vel_xy,
        scale=1.0,
        clip=(-2.0, 2.0),
    )


@configclass
class HistoryDistillationObservationsCfg(Go2ObservationsCfg):
    """Separate deployable student and privileged teacher observation groups."""

    teacher: ObsGroup = PrivilegedVelocityTeacherObsCfg()


@configclass
class ClosedLoopHistoryDistillationObservationsCfg(Go2ObservationsCfg):
    """Deployable 213-D history student and exact 51-D closed-loop teacher."""

    teacher: ObsGroup = PrivilegedClosedLoopTeacherObsCfg()


@configclass
class DeployablePrivilegedDistillationObservationsCfg(Go2ObservationsCfg):
    """Current 45-D policy observations plus a separate 47-D teacher group."""

    teacher: ObsGroup = PrivilegedVelocityTeacherObsCfg()


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
class RobotSpeedOmniAxisQuality45EnvCfg(RobotSpeedOmniAxis45EnvCfg):
    """Polish the accepted Axis-45 gait without changing its policy contract.

    This is intentionally a reward-only continuation of ``axis4698``.  The
    command sampler, 45-D observation and 12-D action remain unchanged.  It
    targets two observed style defects: excessive sustained hip splay at fast
    forward speeds, and feet that drag through pure lateral or yaw commands.
    """

    def __post_init__(self):
        super().__post_init__()

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

        # This remains a soft excess penalty, not a hip=0 target.  Sideways
        # and turning commands expand the allowance, while a sustained wide
        # stance during straight high-speed running is no longer free.
        self.rewards.hip_outward_speed_style.weight = -1.25
        self.rewards.hip_outward_speed_style.params.update(
            {
                "standing_limit": 0.22,
                "walking_limit": 0.27,
                "high_speed_limit": 0.34,
                "walking_speed": 0.35,
                "high_speed": 1.50,
                "lateral_limit_gain": 0.22,
                "yaw_limit_gain": 0.10,
            }
        )

        # The source task accidentally treated a pure yaw command as static
        # for clearance and landing.  Counting yaw as motion gives PPO a
        # direct swing-foot signal for in-place rotations as well as side
        # steps, without adding a gait phase or height target to the policy.
        self.rewards.foot_clearance_style.weight = 0.30
        self.rewards.foot_clearance_style.params.update(
            {
                "target_height": 0.075,
                "std": 0.040,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            }
        )
        self.rewards.foot_soft_landing.weight = -0.05
        self.rewards.foot_soft_landing.params.update(
            {
                "vertical_speed_std": 0.90,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            }
        )
        self.rewards.foot_impact_velocity = RewTerm(
            func=custom_mdp.foot_impact_velocity_l2,
            weight=-0.015,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "impact_speed_std": 1.0,
                "command_threshold": 0.08,
            },
        )
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.16

        # Shape the support polygon rather than commanding a joint angle.  The
        # target shifts with vy and yaw, so lateral movement and turning retain
        # the hip freedom they physically need.
        nominal_feet = (
            0.1661,
            -0.1694,
            -0.2970,
            0.1660,
            0.1696,
            -0.2970,
            -0.2041,
            -0.1696,
            -0.2970,
            -0.2050,
            0.1695,
            -0.2972,
        )
        self.rewards.stance_foot_lateral_placement = RewTerm(
            func=custom_mdp.stance_foot_lateral_placement_l2,
            weight=-0.08,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "nominal_positions": nominal_feet,
                "stance_time": 0.24,
                "position_std": 0.07,
                "max_error": 0.14,
                "command_name": "base_velocity",
            },
        )
        self.rewards.action_smoothness_2 = RewTerm(
            func=custom_mdp.ActionSmoothness2,
            weight=-0.010,
        )


@configclass
class RobotSpeedOmniAxisQuality45PlayEnvCfg(RobotSpeedOmniAxisQuality45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisQualityAware45EnvCfg(RobotSpeedOmniAxisQuality45EnvCfg):
    """Second reward-only polish pass for pure lateral and yaw motion.

    This continues the accepted axis4698 45-D policy.  The additional
    command-aware air-time signal fixes the pure-yaw gate in the stock reward;
    no gait phase, height target, observation, or action dimension is added.
    """

    def __post_init__(self):
        super().__post_init__()

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

        # Compact the high-speed forward stance without fixing hip angles.
        # Side and yaw commands receive extra allowance through the gains.
        self.rewards.hip_outward_speed_style.weight = -3.0
        self.rewards.hip_outward_speed_style.params.update(
            {
                "standing_limit": 0.20,
                "walking_limit": 0.25,
                "high_speed_limit": 0.30,
                "walking_speed": 0.35,
                "high_speed": 1.50,
                "lateral_limit_gain": 0.22,
                "yaw_limit_gain": 0.10,
            }
        )

        # Disable the stock XY-only term and use the command-aware version.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_command_aware = RewTerm(
            func=custom_mdp.feet_air_time_command_aware,
            weight=0.50,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 0.16,
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            },
        )
        self.rewards.foot_clearance_style.weight = 0.40
        self.rewards.foot_clearance_style.params.update(
            {
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.075,
                "std": 0.040,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            }
        )
        self.rewards.foot_soft_landing.weight = -0.05
        self.rewards.foot_soft_landing.params.update(
            {
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "vertical_speed_std": 0.90,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            }
        )
        self.rewards.foot_impact_velocity = RewTerm(
            func=custom_mdp.foot_impact_velocity_l2,
            weight=-0.02,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "impact_speed_std": 1.0,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.35,
            },
        )
        self.rewards.stance_foot_lateral_placement.weight = -0.06
        self.rewards.action_smoothness_2.weight = -0.010


@configclass
class RobotSpeedOmniAxisQualityAware45PlayEnvCfg(RobotSpeedOmniAxisQualityAware45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisSwingDiscovery45EnvCfg(RobotSpeedOmniAxisQualityAware45EnvCfg):
    """Discover actual pure-axis steps without changing the 45-D policy API."""

    def __post_init__(self):
        super().__post_init__()

        contact_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )

        # Give pure vy/wz enough signal to leave the all-feet-planted local
        # optimum, while keeping forward and vx+wz samples as the majority.
        self.commands.base_velocity.omni_mode_probabilities = (0.45, 0.25, 0.25, 0.05)
        self.commands.base_velocity.rel_low_speed_forward = 0.30
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (1.0, 2.0, 2.0),
            }
        )
        self.rewards.track_velocity_components_progress = RewTerm(
            func=custom_mdp.track_velocity_components_progress,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (1.0, 2.0, 2.0),
                "max_progress": (3.2, 0.60, 0.90),
            },
        )
        self.rewards.track_ang_vel_z_l2.weight = -2.0
        self.rewards.lateral_command_progress.weight = 6.0
        self.rewards.lateral_command_progress.params["command_min"] = 0.08
        self.rewards.lateral_command_progress.params["max_progress"] = 0.60
        self.rewards.yaw_command_progress.weight = 4.0
        self.rewards.yaw_command_progress.params["command_min"] = 0.12
        self.rewards.yaw_command_progress.params["max_progress"] = 0.90

        # This term is phase-free and only affects the missing pure-axis
        # behaviours.  Forward and vx+wz retain the learned policy cadence.
        self.rewards.pure_axis_swing_count = RewTerm(
            func=custom_mdp.pure_axis_swing_count,
            weight=0.75,
            params={
                "sensor_cfg": contact_cfg,
                "command_name": "base_velocity",
                "forward_deadband": 0.05,
                "lateral_minimum": 0.08,
                "yaw_minimum": 0.12,
                "target_airborne": 1.5,
                "airborne_std": 0.75,
            },
        )
        self.rewards.feet_air_time_command_aware.weight = 0.75
        self.rewards.foot_clearance_style.weight = 0.55
        self.rewards.hip_outward_speed_style.weight = -1.50
        self.rewards.action_rate.weight = -0.04
        self.rewards.joint_pos.weight = -0.05
        self.rewards.feet_slide.weight = -0.06


@configclass
class RobotSpeedOmniAxisSwingDiscovery45PlayEnvCfg(RobotSpeedOmniAxisSwingDiscovery45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisSwingDirection45EnvCfg(RobotSpeedOmniAxisSwingDiscovery45EnvCfg):
    """Add a geometric placement direction to the pure-axis swing reward."""

    def __post_init__(self):
        super().__post_init__()
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
        self.rewards.pure_axis_swing_direction = RewTerm(
            func=custom_mdp.pure_axis_swing_direction,
            weight=1.00,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "command_name": "base_velocity",
                "forward_deadband": 0.05,
                "lateral_minimum": 0.08,
                "yaw_minimum": 0.12,
                "target_speed": 0.25,
            },
        )


@configclass
class RobotSpeedOmniAxisSwingDirection45PlayEnvCfg(RobotSpeedOmniAxisSwingDirection45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisPhaseBridgeEnvCfg(RobotSpeedOmniAxis45EnvCfg):
    """Add a deployable gait clock while preserving the axis4698 forward expert.

    The first 45 observation dimensions are unchanged.  A two-channel
    sine/cosine clock is appended only to make pure lateral and yaw commands
    phase-identifiable; the actor is initialized with zero weights for those
    columns so its initial forward behaviour is exactly the 45-D expert.
    """

    def __post_init__(self):
        super().__post_init__()

        gait_period = 0.70
        self.observations.policy.gait_phase = ObsTerm(
            func=custom_mdp.command_gait_phase,
            params={
                "period": gait_period,
                "command_name": "base_velocity",
                "command_threshold": 0.08,
            },
        )

        # Keep the user-facing modes identifiable.  The combined bucket is
        # deliberately vx+wz, not three-axis motion, because it is the main
        # turning use case and must not compromise pure-axis learning.
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.35, 0.25, 0.20, 0.20),
            combined_include_lateral=False,
            minimum_command_magnitude=0.10,
            negative_x_probability=0.0,
            rel_standing_envs=0.05,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.10, 3.0),
                lin_vel_y=(-0.30, 0.30),
                ang_vel_z=(-0.60, 0.60),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None

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

        # Pure side/yaw commands need an explicit alternating contact signal.
        # Forward and vx+wz stay unclocked to protect the accepted high-speed
        # gait from an arbitrary fixed cadence.
        self.rewards.pure_axis_gait = RewTerm(
            func=custom_mdp.pure_axis_feet_gait,
            weight=0.75,
            params={
                "period": gait_period,
                "threshold": 0.55,
                "forward_deadband": 0.05,
                "lateral_minimum": 0.10,
                "yaw_minimum": 0.15,
                "command_name": "base_velocity",
                "sensor_cfg": contact_cfg,
            },
        )
        self.rewards.foot_clearance_style.params.update(
            {
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.060,
                "std": 0.045,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.30,
            }
        )
        self.rewards.foot_clearance_style.weight = 0.15
        self.rewards.foot_soft_landing.params.update(
            {
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "command_threshold": 0.08,
                "yaw_speed_scale": 0.30,
            }
        )
        self.rewards.feet_air_time.weight = 0.20
        self.rewards.feet_air_time.params["threshold"] = 0.18

        # Track every active command component but avoid the old very large
        # progress reward, which traded stability for speed during bootstrap.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.30
        self.rewards.track_ang_vel_z.weight = 2.5
        self.rewards.track_ang_vel_z.params["std"] = 0.30
        self.rewards.track_velocity_components_relative_l1.weight = -3.0
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (1.0, 2.0, 1.5),
            }
        )
        self.rewards.track_ang_vel_z_l2.weight = -1.0
        self.rewards.lateral_command_progress.weight = 1.0
        self.rewards.yaw_command_progress.weight = 1.5
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (0.5, 1.5, 1.25),
            },
        )

        # This is a soft excessive-splay cost, not a hip-to-zero target.  Side
        # steps and turns retain extra clearance, while sustained high-speed
        # outward posture becomes less attractive than a compact landing plan.
        self.rewards.hip_outward_speed_style.params.update(
            {
                "standing_limit": 0.20,
                "walking_limit": 0.25,
                "high_speed_limit": 0.32,
                "walking_speed": 0.40,
                "high_speed": 1.75,
                "lateral_limit_gain": 0.16,
                "yaw_limit_gain": 0.06,
            }
        )
        self.rewards.hip_outward_speed_style.weight = -1.0
        self.rewards.action_rate.weight = -0.06
        self.rewards.action_smoothness_2 = RewTerm(
            func=custom_mdp.ActionSmoothness2,
            weight=-0.012,
        )
        self.rewards.feet_slide.weight = -0.08
        self.rewards.base_height = None
        self.rewards.speed_adaptive_base_height = None


@configclass
class RobotSpeedOmniAxisPhaseBridgePlayEnvCfg(RobotSpeedOmniAxisPhaseBridgeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisStylePolish45EnvCfg(RobotSpeedOmniAxis45EnvCfg):
    """Refine the accepted axis4698 posture without fixing hip angles."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve the operating modes that axis4698 already performs.  Pure
        # lateral and reverse discovery are separate stages so their large PPO
        # gradients cannot erase the accepted 0-3 m/s forward gait here.
        self.commands.base_velocity = custom_mdp.MovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.50, 0.10, 0.40, 0.0),
            rel_standing_envs=0.05,
            rel_low_speed_forward=0.35,
            low_speed_range=(0.15, 0.75),
            rel_high_speed_forward=0.45,
            high_speed_range=(1.50, 3.00),
            high_speed_modes=(0, 2),
            lateral_min_fraction=0.40,
            yaw_min_fraction=0.35,
            debug_vis=False,
            ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 3.00),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.60, 0.60),
            ),
            limit_ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 3.00),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
        )

        # Foot geometry, rather than a fixed hip target, defines the preferred
        # stance.  Only lateral foot placement is shaped so high-speed stride
        # length remains free.
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
        nominal_feet = (
            0.1661,
            -0.1694,
            -0.2970,
            0.1660,
            0.1696,
            -0.2970,
            -0.2041,
            -0.1696,
            -0.2970,
            -0.2050,
            0.1695,
            -0.2972,
        )
        self.rewards.hip_outward_speed_style = None
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.stance_foot_lateral_placement = RewTerm(
            func=custom_mdp.stance_foot_lateral_placement_l2,
            weight=-0.18,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "nominal_positions": nominal_feet,
                "stance_time": 0.24,
                "position_std": 0.06,
                "max_error": 0.12,
                "command_name": "base_velocity",
            },
        )
        self.rewards.action_smoothness_2 = RewTerm(
            func=custom_mdp.ActionSmoothness2,
            weight=-0.015,
        )
        self.rewards.foot_impact_velocity = RewTerm(
            func=custom_mdp.foot_impact_velocity_l2,
            weight=-0.02,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "impact_speed_std": 0.90,
                "command_name": "base_velocity",
            },
        )
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.05, 0.10),
                "axis_weights": (0.25, 2.0, 1.0),
            },
        )
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_velocity_components_relative_l1.weight = -2.0
        self.rewards.track_ang_vel_z_l2.weight = -0.75
        self.rewards.lateral_command_progress.weight = 0.5
        self.rewards.yaw_command_progress.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate.weight = -0.06
        self.rewards.joint_pos.weight = -0.06
        self.rewards.feet_slide.weight = -0.08
        self.rewards.base_height = None
        self.rewards.speed_adaptive_base_height = None


@configclass
class RobotSpeedOmniAxisStylePolish45PlayEnvCfg(RobotSpeedOmniAxisStylePolish45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniAxisBootstrap45EnvCfg(RobotSpeedOmniAxis45EnvCfg):
    """Bootstrap missing lateral/yaw behaviors while retaining the 45-D contract."""

    def __post_init__(self):
        super().__post_init__()

        # A straight-running checkpoint can minimize broad omni losses by
        # standing still.  Concentrate updates on unmistakable pure-axis
        # commands first; the play/deploy limits still retain 0-3 m/s vx.
        self.commands.base_velocity.omni_mode_probabilities = (0.15, 0.40, 0.40, 0.05)
        self.commands.base_velocity.rel_standing_envs = 0.03
        self.commands.base_velocity.rel_low_speed_forward = 0.0
        self.commands.base_velocity.lateral_min_fraction = 0.65
        self.commands.base_velocity.yaw_min_fraction = 0.65
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.80)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 3.0)

        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            2.0,
            1.75,
        )
        self.rewards.track_ang_vel_z_l2.weight = -2.0
        self.rewards.lateral_command_progress.weight = 8.0
        self.rewards.lateral_command_progress.params["max_progress"] = 0.65
        self.rewards.yaw_command_progress.weight = 6.0
        self.rewards.yaw_command_progress.params["max_progress"] = 1.0

        # Permit gait discovery without removing the generic smoothness and
        # soft hip-style terms that produced the improved forward posture.
        self.rewards.joint_pos.weight = -0.05
        self.rewards.action_rate.weight = -0.03
        self.rewards.feet_slide.weight = -0.04


@configclass
class RobotSpeedOmniAxisBootstrap45PlayEnvCfg(RobotSpeedOmniAxisBootstrap45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniMoving45EnvCfg(RobotSpeedOmniStyle45EnvCfg):
    """Learn lateral/yaw steering on top of the validated moving forward gait."""

    def __post_init__(self):
        super().__post_init__()

        # A non-zero vx keeps the inherited gait oscillator active while PPO
        # learns how vy and wz should modulate it. Pure-axis commands are added
        # only after this steering stage passes sim2sim validation.
        self.commands.base_velocity.omni_mode_probabilities = (0.45, 0.0, 0.0, 0.55)
        self.commands.base_velocity.rel_standing_envs = 0.03
        self.commands.base_velocity.rel_low_speed_forward = 0.65
        self.commands.base_velocity.low_speed_range = (0.30, 0.80)
        self.commands.base_velocity.lateral_min_fraction = 0.45
        self.commands.base_velocity.yaw_min_fraction = 0.45

        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.5
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.5,
            1.25,
        )
        self.rewards.track_velocity_components_progress = RewTerm(
            func=custom_mdp.track_velocity_components_progress,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.10, 0.15),
                "axis_weights": (1.0, 1.5, 1.25),
                "max_progress": (3.2, 0.65, 1.0),
            },
        )


@configclass
class RobotSpeedOmniMoving45PlayEnvCfg(RobotSpeedOmniMoving45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniMovingDisentangled45EnvCfg(RobotSpeedOmniMoving45EnvCfg):
    """Separate side-steering and yaw learning before exposing pure-axis commands."""

    def __post_init__(self):
        super().__post_init__()
        previous = self.commands.base_velocity
        self.commands.base_velocity = custom_mdp.MovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.35, 0.30, 0.25, 0.10),
            rel_standing_envs=0.03,
            rel_low_speed_forward=0.65,
            low_speed_range=(0.30, 0.80),
            lateral_min_fraction=0.45,
            yaw_min_fraction=0.45,
            debug_vis=False,
            ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.10, 3.0),
                lin_vel_y=(-0.42, 0.42),
                ang_vel_z=(-0.60, 0.60),
            ),
            limit_ranges=previous.limit_ranges,
        )
        self.rewards.track_ang_vel_z_l2.weight = -3.0


@configclass
class RobotSpeedOmniMovingDisentangled45PlayEnvCfg(RobotSpeedOmniMovingDisentangled45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniSteeringPolish45EnvCfg(RobotSpeedOmniMovingDisentangled45EnvCfg):
    """Reduce lateral/yaw cross-coupling without changing the 45-D policy contract."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.steering_mode_probabilities = (0.30, 0.35, 0.30, 0.05)

        # Relative error intentionally ignores zero-command axes. These full
        # errors prevent a lateral command from being implemented as a turn,
        # or a yaw command from being implemented as a curved side-slip.
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-4.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2.weight = -5.0
        self.rewards.track_velocity_components_relative_l1.weight = -3.0
        self.rewards.track_velocity_components_progress.weight = 1.0


@configclass
class RobotSpeedOmniSteeringPolish45PlayEnvCfg(RobotSpeedOmniSteeringPolish45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniBidirectional45EnvCfg(RobotSpeedOmniSteeringPolish45EnvCfg):
    """Conservative backward curriculum with the final joystick limits exposed."""

    def __post_init__(self):
        super().__post_init__()
        # The source checkpoint has only seen positive vx.  Train a narrow
        # negative band first, while keeping the exported contract at the
        # requested vx=[-3, 3], vy=[-.4, .4], wz=[-1, 1].
        self.commands.base_velocity = custom_mdp.MovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.30, 0.35, 0.30, 0.05),
            rel_standing_envs=0.03,
            rel_low_speed_forward=0.65,
            low_speed_range=(0.30, 0.80),
            lateral_min_fraction=0.45,
            yaw_min_fraction=0.45,
            debug_vis=False,
            ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.80, 3.0),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-0.80, 0.80),
            ),
            limit_ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(-3.0, 3.0),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-1.0, 1.0),
            ),
        )


@configclass
class RobotSpeedOmniBidirectional45PlayEnvCfg(RobotSpeedOmniBidirectional45EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotSpeedOmniBackwardDiscovery45EnvCfg(RobotSpeedOmniSteeringPolish45EnvCfg):
    """Discover a backward gait without sacrificing the accepted forward range."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.40, 0.25, 0.25, 0.10),
            rel_standing_envs=0.03,
            rel_low_speed_forward=0.45,
            low_speed_range=(0.30, 0.80),
            rel_backward_envs=0.35,
            backward_speed_range=(0.25, 0.80),
            lateral_min_fraction=0.45,
            yaw_min_fraction=0.45,
            debug_vis=False,
            ranges=custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.10, 3.0),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-0.80, 0.80),
            ),
            limit_ranges=custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(-3.0, 3.0),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
        self.rewards.track_velocity_components_progress.weight = 5.0
        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        # Give the policy room to discover the reversed contact sequence; the
        # style/smoothness terms remain active and are restored in polishing.
        self.rewards.action_rate.weight = -0.04
        self.rewards.joint_pos.weight = -0.05


@configclass
class RobotSpeedOmniBackwardDiscovery45PlayEnvCfg(RobotSpeedOmniBackwardDiscovery45EnvCfg):
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
class RobotSelfRightingR0EnvCfg(RobotEnvCfg):
    """Dedicated zero-command recovery from the measured belly-down fold."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn = CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.copy()
        self.episode_length_s = 5.0

        self.events.reset_base = EventTerm(
            func=custom_mdp.reset_self_righting_states,
            mode="reset",
            params={
                "orientation_probabilities": (1.0, 0.0, 0.0, 0.0),
                "arbitrary_orientation_probability": 0.0,
                "root_height_range": (0.08, 0.10),
                "hip_position_range": (-0.03, 0.03),
                "thigh_position_range": (1.20, 1.28),
                "calf_position_range": (-2.84, -2.76),
                "max_linear_velocity": 0.0,
                "max_angular_velocity": 0.0,
                "root_xy_range": (-0.20, 0.20),
            },
        )
        self.events.reset_robot_joints = None
        self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (0.75, 1.15)
        self.events.physics_material.params["dynamic_friction_range"] = (0.70, 1.10)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.25, 0.25)

        command = self.commands.base_velocity
        command.rel_standing_envs = 1.0
        command.heading_command = False
        for ranges in (command.ranges, command.limit_ranges):
            ranges.lin_vel_x = (0.0, 0.0)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.terrain_levels = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

        feet_contact_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        success_params = {
            "minimum_height": 0.27,
            "maximum_tilt_deg": 15.0,
            "maximum_angular_velocity": 0.50,
            "minimum_contact_feet": 4,
            "contact_force_threshold": 1.0,
            "sensor_cfg": feet_contact_cfg,
        }
        self.terminations.base_contact = None
        self.terminations.bad_orientation = None
        self.terminations.recovery_success = DoneTerm(
            func=custom_mdp.recovery_success_dwell,
            params={"dwell_time_s": 0.40, **success_params},
        )

        self.rewards.track_lin_vel_xy.weight = 0.0
        self.rewards.track_ang_vel_z.weight = 0.0
        self.rewards.base_linear_velocity.weight = 0.0
        self.rewards.base_angular_velocity.weight = 0.0
        self.rewards.joint_vel.weight = -1.0e-4
        self.rewards.joint_acc.weight = -1.0e-7
        self.rewards.joint_torques.weight = -5.0e-5
        self.rewards.action_rate.weight = -0.02
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.energy.weight = -1.0e-5
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.joint_pos.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.recovery_orientation = RewTerm(
            func=custom_mdp.recovery_orientation_progress,
            weight=2.0,
        )
        self.rewards.recovery_upright_height = RewTerm(
            func=custom_mdp.recovery_upright_height,
            weight=3.0,
            params={
                "prone_height": 0.07,
                "standing_height": 0.30,
                "prone_only": False,
            },
        )
        self.rewards.recovery_stable_support = RewTerm(
            func=custom_mdp.recovery_stable_support,
            weight=4.0,
            params={
                "prone_height": 0.07,
                "standing_height": 0.30,
                "angular_velocity_std": 0.75,
                "contact_force_threshold": 1.0,
                "sensor_cfg": feet_contact_cfg,
            },
        )
        self.rewards.recovery_success = RewTerm(
            func=custom_mdp.recovery_success_state,
            weight=10.0,
            params=success_params,
        )
        self.rewards.time_cost = RewTerm(func=mdp.is_alive, weight=-0.05)
        if hasattr(self.rewards, "termination_penalty"):
            self.rewards.termination_penalty = None


@configclass
class RobotSelfRightingR0PlayEnvCfg(RobotSelfRightingR0EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1


@configclass
class RobotSelfRightingR1EnvCfg(RobotSelfRightingR0EnvCfg):
    """Expand recovery to back-down and both side-lying orientations."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.0
        params = self.events.reset_base.params
        params["orientation_probabilities"] = (0.25, 0.25, 0.25, 0.25)
        params["root_height_range"] = (0.11, 0.16)
        params["hip_position_range"] = (-0.20, 0.20)
        params["thigh_position_range"] = (0.90, 1.55)
        params["calf_position_range"] = (-2.85, -1.90)
        params["max_linear_velocity"] = 0.15
        params["max_angular_velocity"] = 0.35


@configclass
class RobotSelfRightingR1PlayEnvCfg(RobotSelfRightingR1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1


@configclass
class RobotSelfRightingR2EnvCfg(RobotSelfRightingR1EnvCfg):
    """Final flat-ground recovery distribution with arbitrary dynamic falls."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 7.0
        params = self.events.reset_base.params
        params["arbitrary_orientation_probability"] = 0.35
        params["root_height_range"] = (0.14, 0.24)
        params["hip_position_range"] = (-0.45, 0.45)
        params["thigh_position_range"] = (0.45, 1.75)
        params["calf_position_range"] = (-2.85, -1.10)
        params["max_linear_velocity"] = 0.50
        params["max_angular_velocity"] = 1.50


@configclass
class RobotSelfRightingR2PlayEnvCfg(RobotSelfRightingR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1


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


@configclass
class RobotOmni45V2EnvCfg(RobotOmniSymmetryEnvCfg):
    """45-D omni locomotion with stratified commands and geometry-based style."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 6.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.35, 0.25, 0.20, 0.20),
            minimum_command_magnitude=0.08,
            rel_standing_envs=0.08,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.50, 0.50),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.40, 0.40),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.omni_velocity_cmd_levels = CurrTerm(
            func=custom_mdp.stratified_omni_cmd_levels,
            params={
                "lin_reward_term": "track_lin_vel_xy",
                "yaw_reward_term": "track_ang_vel_z",
                "increments": (0.25, 0.05, 0.10),
                "lin_success_threshold": 0.70,
                "yaw_success_threshold": 0.70,
            },
        )

        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.35
        self.rewards.track_ang_vel_z.weight = 1.5
        self.rewards.track_ang_vel_z.params["std"] = 0.35
        self.rewards.track_velocity_components_relative_l1 = RewTerm(
            func=custom_mdp.track_velocity_components_relative_l1,
            weight=-0.50,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (1.0, 1.20, 1.0),
            },
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-0.50,
            params={"command_name": "base_velocity"},
        )

        # First- and second-order action shaping. Neither changes the policy
        # observation or deployment contract.
        self.rewards.action_rate.weight = -0.04
        self.rewards.action_smoothness_2 = RewTerm(
            func=custom_mdp.ActionSmoothness2,
            weight=-0.02,
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
        self.rewards.feet_slide.weight = -0.08
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
        self.rewards.foot_impact_velocity = RewTerm(
            func=custom_mdp.foot_impact_velocity_l2,
            weight=-0.02,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "impact_speed_std": 0.90,
                "command_name": "base_velocity",
            },
        )
        self.rewards.stance_foot_placement = RewTerm(
            func=custom_mdp.stance_foot_placement_l2,
            weight=-0.08,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                # FR, FL, RR, RL body-frame home positions measured from the
                # canonical MuJoCo model at the home keyframe.
                "nominal_positions": (
                    0.1661,
                    -0.1694,
                    -0.2970,
                    0.1660,
                    0.1696,
                    -0.2970,
                    -0.2041,
                    -0.1696,
                    -0.2970,
                    -0.2050,
                    0.1695,
                    -0.2972,
                ),
                "stance_time": 0.24,
                "position_std": 0.07,
                "max_error": 0.18,
                "command_name": "base_velocity",
            },
        )

        # Foot geometry replaces explicit hip-angle and height targets.  Hip
        # ab/adduction remains free for turning, lateral motion and recovery.
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.speed_adaptive_base_height = None
        self.rewards.base_height = None
        self.rewards.joint_pos.weight = -0.08
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.20
        self.rewards.air_time_variance.weight = -0.05


@configclass
class RobotOmni45V2PlayEnvCfg(RobotOmni45V2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45HighSpeedEnvCfg(RobotOmni45V2EnvCfg):
    """Expand the converged signed omni policy to the target joystick limits."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        # model_4960 already covers this envelope. Starting here avoids both a
        # needless easy stage and an abrupt jump to out-of-distribution speeds.
        command.ranges = command.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
        )
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.0, 2.0),
        )
        command.bucket_probabilities = (0.30, 0.20, 0.20, 0.30)
        command.rel_standing_envs = 0.05

        curriculum = self.curriculum.omni_velocity_cmd_levels
        curriculum.params["increments"] = (0.25, 0.05, 0.10)
        curriculum.params["lin_success_threshold"] = 0.72
        curriculum.params["yaw_success_threshold"] = 0.72

        # High-speed combinations need stronger tracking without removing the
        # existing posture, contact, and action-smoothness objectives.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_velocity_components_relative_l1.weight = -0.75
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.35,
            1.15,
        )
        self.rewards.track_ang_vel_z_l2.weight = -0.75


@configclass
class RobotOmni45HighSpeedPlayEnvCfg(RobotOmni45HighSpeedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotEnvCfg(RobotOmni45V2EnvCfg):
    """From-scratch all-direction controller with an explicit diagonal trot."""

    def __post_init__(self):
        super().__post_init__()

        gait_params = {
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "min_frequency": 1.4,
            "max_frequency": 3.2,
            "full_speed": 3.0,
            "yaw_speed_scale": 0.35,
        }
        self.observations.policy.trot_clock = ObsTerm(
            func=custom_mdp.command_trot_clock,
            params=gait_params,
        )
        self.observations.critic.trot_clock = ObsTerm(
            func=custom_mdp.command_trot_clock,
            params=gait_params,
        )

        # Start with a learnable signed envelope and expand symmetrically to
        # the joystick contract only after planar and yaw tracking succeed.
        command = self.commands.base_velocity
        command.ranges = command.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.4, 0.4),
        )
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.0, 2.0),
        )
        command.bucket_probabilities = (0.35, 0.20, 0.20, 0.25)
        command.negative_x_probability = 0.50
        command.rel_low_speed_x = 0.25
        command.low_speed_x_range = (0.15, 0.45)
        command.rel_standing_envs = 0.08

        curriculum = self.curriculum.omni_velocity_cmd_levels
        curriculum.params.update(
            {
                "increments": (0.25, 0.05, 0.10),
                "lin_success_threshold": 0.72,
                "yaw_success_threshold": 0.72,
            }
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
        gait_reward_params = {
            **gait_params,
            "duty_factor": 0.52,
            "sensor_cfg": contact_cfg,
        }
        self.rewards.trot_contact_schedule = RewTerm(
            func=custom_mdp.trot_contact_schedule,
            weight=1.5,
            params=gait_reward_params,
        )
        self.rewards.trot_stance_swing_tracking = RewTerm(
            func=custom_mdp.trot_stance_swing_tracking,
            weight=1.0,
            params={
                **gait_reward_params,
                "asset_cfg": feet_cfg,
                "stance_velocity_std": 0.35,
                "swing_force_std": 25.0,
            },
        )

        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_velocity_components_relative_l1.weight = -0.75
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.30,
            1.15,
        )
        self.rewards.track_ang_vel_z_l2.weight = -0.75

        # A soft height target prevents the low-body shortcut while allowing
        # the robot to crouch modestly as planar/yaw demand rises.
        self.rewards.speed_adaptive_base_height = RewTerm(
            func=custom_mdp.speed_adaptive_base_height_l2,
            weight=-4.0,
            params={
                "standing_height": 0.31,
                "crouched_height": 0.26,
                "crouch_start_speed": 0.7,
                "crouch_full_speed": 3.0,
                "yaw_speed_scale": 0.35,
                "command_name": "base_velocity",
            },
        )
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_angular_velocity.weight = -0.08
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.air_time_variance.weight = 0.0


@configclass
class RobotOmniTrotClosedLoopFoundationEnvCfg(RobotOmniTrotEnvCfg):
    """Random-init 51-D trot teacher with explicit planar velocity feedback.

    This is deliberately separate from the 49-D continuation tasks.  The
    actor sees the measured body-frame ``vx/vy`` from iteration zero, while
    the four-value trot clock keeps the desired diagonal gait observable.
    """

    def __post_init__(self):
        super().__post_init__()

        # The actor contract is 45-D base state + 4-D trot clock + 2-D
        # measured planar velocity.  The critic already has privileged base
        # velocity, so only the policy group is extended here.
        self.observations.policy.base_lin_vel_xy = ObsTerm(
            func=custom_mdp.base_lin_vel_xy,
            scale=1.0,
            clip=(-2.0, 2.0),
        )

        command = self.commands.base_velocity
        command.resampling_time_range = (6.0, 8.0)
        command.ranges = command.Ranges(
            lin_vel_x=(-0.45, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )
        # Do not advertise the final joystick envelope in deploy.yaml before
        # this stage has passed the fixed MuJoCo grid.
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-0.45, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )
        command.bucket_probabilities = (0.30, 0.22, 0.28, 0.20)
        command.minimum_command_magnitude = 0.03
        command.rel_standing_envs = 0.10
        command.negative_x_probability = 0.50
        command.combined_include_lateral = True
        command.rel_low_speed_x = 0.65
        command.low_speed_x_range = (0.03, 0.20)
        command.rel_low_speed_y = 0.65
        command.low_speed_y_range = (0.03, 0.10)
        command.rel_low_speed_yaw = 0.70
        command.low_speed_yaw_range = (0.05, 0.20)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.omni_velocity_cmd_levels = None

        planar_deadband = 0.03
        yaw_deadband = 0.05
        for clock in (self.observations.policy.trot_clock, self.observations.critic.trot_clock):
            clock.params.update(
                {
                    "command_threshold": planar_deadband,
                    "yaw_command_threshold": yaw_deadband,
                }
            )
        for reward in (self.rewards.trot_contact_schedule, self.rewards.trot_stance_swing_tracking):
            reward.params.update(
                {
                    "command_threshold": planar_deadband,
                    "yaw_command_threshold": yaw_deadband,
                }
            )
        self.rewards.trot_contact_schedule.weight = 1.75
        self.rewards.trot_stance_swing_tracking.weight = 1.10

        # Tracking is primary during discovery.  The relative and L2 terms
        # keep small commands visible without making standing the cheapest
        # solution.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.30
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_ang_vel_z.params["std"] = 0.28
        self.rewards.track_velocity_components_relative_l1.weight = -1.5
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.03, 0.03, 0.05),
                "axis_weights": (1.0, 1.5, 1.5),
            }
        )
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-1.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2 = RewTerm(
            func=custom_mdp.track_ang_vel_z_l2,
            weight=-1.25,
            params={"command_name": "base_velocity"},
        )
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-2.5,
            params={
                "command_name": "base_velocity",
                "command_min": (0.03, 0.03, 0.05),
                "axis_weights": (1.0, 1.5, 1.5),
            },
        )
        self.rewards.pure_axis_velocity_decoupling = RewTerm(
            func=custom_mdp.pure_axis_velocity_decoupling_l2,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.03, 0.03, 0.05),
                "axis_weights": (1.0, 1.5, 1.5),
            },
        )

        # Keep the trunk level and the body height in the requested band while
        # allowing a smooth 0.33 -> 0.28 m crouch as speed rises.
        self.rewards.speed_adaptive_base_height.weight = -35.0
        self.rewards.speed_adaptive_base_height.params.update(
            {
                "standing_height": 0.33,
                "crouched_height": 0.28,
                "crouch_start_speed": 0.10,
                "crouch_full_speed": 3.0,
                "yaw_speed_scale": 0.35,
                "command_name": "base_velocity",
            }
        )
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_angular_velocity.weight = -0.12
        self.rewards.base_linear_velocity.weight = -2.5
        self.rewards.energy.weight = -3.0e-5
        self.rewards.action_rate.weight = -0.04
        self.rewards.action_smoothness_2.weight = -0.02

        # A symmetric calibrated home pose is active only while standing.
        nominal_feet = (
            0.1661,
            -0.1694,
            -0.2970,
            0.1660,
            0.1696,
            -0.2970,
            -0.2041,
            -0.1696,
            -0.2970,
            -0.2050,
            0.1695,
            -0.2972,
        )
        self.rewards.joint_pos = RewTerm(
            func=custom_mdp.joint_deviation_l2,
            weight=-0.12,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 3.0,
                "velocity_threshold": 0.20,
                "command_name": "base_velocity",
                "planar_command_threshold": planar_deadband,
                "yaw_command_threshold": yaw_deadband,
            },
        )
        self.rewards.standing_foot_placement = RewTerm(
            func=custom_mdp.standing_foot_placement_l2,
            weight=-0.30,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    preserve_order=True,
                ),
                "nominal_positions": nominal_feet,
                "command_name": "base_velocity",
                "planar_command_threshold": planar_deadband,
                "yaw_command_threshold": yaw_deadband,
                "velocity_threshold": 0.20,
                "position_std": 0.045,
                "max_error": 0.12,
            },
        )
        self.rewards.hip_nominal = None
        self.rewards.hip_outward_excess = None
        self.rewards.hip_outward_band = RewTerm(
            func=custom_mdp.hip_outward_band_l2,
            weight=-4.0,
            params={
                "standing_band": (-0.02, 0.12),
                "walking_band": (-0.04, 0.16),
                "high_speed_band": (-0.06, 0.20),
                "walking_speed": 0.35,
                "high_speed": 2.0,
                "lateral_allowance": 0.10,
                "yaw_allowance": 0.04,
                "command_name": "base_velocity",
            },
        )
        self.rewards.calf_pair_separation = RewTerm(
            func=custom_mdp.paired_lateral_separation_l2,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
                    preserve_order=True,
                ),
                "minimum_separation": 0.18,
                "violation_scale": 0.05,
                "lateral_allowance": 0.06,
                "yaw_allowance": 0.02,
                "command_name": "base_velocity",
            },
        )

        contact_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        feet_cfg = SceneEntityCfg(
            "robot",
            body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
            preserve_order=True,
        )
        self.rewards.feet_air_time_command_aware = RewTerm(
            func=custom_mdp.feet_air_time_command_aware,
            weight=0.35,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 0.12,
                "command_name": "base_velocity",
                "command_threshold": planar_deadband,
                "yaw_speed_scale": 1.0,
            },
        )
        self.rewards.foot_clearance_style = RewTerm(
            func=custom_mdp.foot_clearance_speed_style,
            weight=0.25,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.055,
                "std": 0.035,
                "command_name": "base_velocity",
                "command_threshold": planar_deadband,
                "yaw_speed_scale": 1.0,
            },
        )
        self.rewards.all_motion_swing_count = RewTerm(
            func=custom_mdp.motion_swing_count,
            weight=0.40,
            params={
                "sensor_cfg": contact_cfg,
                "command_name": "base_velocity",
                "planar_deadband": planar_deadband,
                "yaw_deadband": yaw_deadband,
                "target_airborne": 2.0,
                "airborne_std": 0.75,
            },
        )


@configclass
class RobotOmniTrotClosedLoopFoundationPlayEnvCfg(RobotOmniTrotClosedLoopFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopPolishA1EnvCfg(RobotOmniTrotClosedLoopFoundationEnvCfg):
    """Correct Stage-A yaw overshoot and low standing height without expanding commands."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.resampling_time_range = (8.0, 12.0)
        command.bucket_probabilities = (0.22, 0.13, 0.45, 0.20)
        command.rel_standing_envs = 0.18
        command.rel_low_speed_yaw = 0.80
        command.low_speed_yaw_range = (0.05, 0.18)

        # Stage A already passed planar tracking and pure-yaw XY drift, but
        # consistently overshot the requested +/-0.25 rad/s yaw rate.
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.20
        self.rewards.track_ang_vel_z_l2.weight = -3.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.5,
            2.5,
        )

        # The zero-command MuJoCo height settled near 0.30 m despite the
        # 0.33 m target. Strengthen height tracking and let the legs extend
        # away from the nominal pose while retaining the foot XY footprint.
        self.rewards.speed_adaptive_base_height.weight = -90.0
        self.rewards.joint_pos.weight = -0.08
        self.rewards.joint_pos.params["stand_still_scale"] = 1.5


@configclass
class RobotOmniTrotClosedLoopPolishA1PlayEnvCfg(RobotOmniTrotClosedLoopPolishA1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopPolishA2EnvCfg(RobotOmniTrotClosedLoopPolishA1EnvCfg):
    """Correct the remaining Stage-A boundary-yaw and standing-band failures."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.bucket_probabilities = (0.18, 0.10, 0.52, 0.20)
        command.rel_low_speed_yaw = 0.50
        command.rel_high_speed_yaw = 0.40
        command.high_speed_yaw_range = (0.20, 0.25)

        # A1 improved its aggregate training yaw metric but still overshot a
        # fixed 0.25 rad/s command in both Isaac and MuJoCo.  Keep symmetric
        # tracking for recovery, then penalize only true pure-yaw overspeed.
        self.rewards.yaw_overspeed_relative = RewTerm(
            func=custom_mdp.yaw_overspeed_relative_l2,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.05,
                "planar_command_threshold": 0.03,
                "max_ratio": 1.0,
            },
        )

        # A normalized band loss supplies a useful gradient at the observed
        # 0.30 m stand, while remaining zero throughout the accepted range.
        self.rewards.standing_base_height_band = RewTerm(
            func=custom_mdp.standing_base_height_band_l2,
            weight=-0.8,
            params={
                "command_name": "base_velocity",
                "lower_height": 0.31,
                "upper_height": 0.33,
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 0.20,
                "error_std": 0.02,
                "max_error": 0.08,
            },
        )
        self.rewards.joint_pos.weight = -0.10
        self.rewards.joint_pos.params["stand_still_scale"] = 2.5
        self.rewards.standing_foot_placement.weight = -0.50
        self.rewards.hip_outward_band.params["standing_band"] = (-0.02, 0.10)


@configclass
class RobotOmniTrotClosedLoopPolishA2PlayEnvCfg(RobotOmniTrotClosedLoopPolishA2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopCrossPhysicsEnvCfg(RobotOmniTrotClosedLoopPolishA2EnvCfg):
    """Adapt Stage A to bounded actuator and rigid-body physics variation."""

    def __post_init__(self):
        super().__post_init__()

        # A2 is accurate to slightly conservative in Isaac; its residual yaw
        # overshoot appears only in MuJoCo. Do not bias the nominal policy
        # further. Instead expose it to a bounded family of physical systems.
        self.rewards.yaw_overspeed_relative = None
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.55, 1.20),
                "dynamic_friction_range": (0.50, 1.15),
                "restitution_range": (0.0, 0.08),
            }
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.scale_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.95, 1.05),
                "operation": "scale",
            },
        )
        self.events.randomize_base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.015, 0.015),
                    "y": (-0.015, 0.015),
                    "z": (-0.010, 0.010),
                },
            },
        )
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.75, 1.35),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )
        self.events.randomize_joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.0, 0.03),
                "operation": "abs",
                "distribution": "uniform",
            },
        )
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 2


@configclass
class RobotOmniTrotClosedLoopCrossPhysicsPlayEnvCfg(
    RobotOmniTrotClosedLoopCrossPhysicsEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopRobustFoundationEnvCfg(
    RobotOmniTrotClosedLoopCrossPhysicsEnvCfg
):
    """From-scratch Stage-A teacher with salient yaw feedback and bounded physics."""

    def __post_init__(self):
        super().__post_init__()

        # The inherited Go2 scale makes a full 0.25 rad/s yaw command only
        # 0.05 in observation space. Keep roll/pitch scaling conservative but
        # expose yaw rate at the same scale as the command and planar velocity.
        self.observations.policy.base_ang_vel.scale = (0.2, 0.2, 1.0)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.10, n_max=0.10)
        self.observations.critic.base_ang_vel.scale = (0.2, 0.2, 1.0)

        command = self.commands.base_velocity
        command.rel_standing_envs = 0.25
        command.bucket_probabilities = (0.18, 0.10, 0.52, 0.20)
        command.rel_low_speed_yaw = 0.50
        command.rel_high_speed_yaw = 0.40
        command.high_speed_yaw_range = (0.20, 0.25)

        # Physics variation should make feedback useful, while this asymmetric
        # term prevents a robust policy from choosing systematic yaw overspeed.
        self.rewards.yaw_overspeed_relative = RewTerm(
            func=custom_mdp.yaw_overspeed_relative_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_min": 0.05,
                "planar_command_threshold": 0.03,
                "max_ratio": 1.0,
            },
        )


@configclass
class RobotOmniTrotClosedLoopRobustFoundationPlayEnvCfg(
    RobotOmniTrotClosedLoopRobustFoundationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotRobustStandFixEnvCfg(
    RobotOmniTrotClosedLoopRobustFoundationEnvCfg
):
    """Repair the robust teacher's zero-command pose without shaping motion hips."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.resampling_time_range = (8.0, 12.0)
        command.rel_standing_envs = 0.45

        # RF_700 passes all moving-command absolute gates. These normalized
        # terms are exactly zero outside the joystick dead-zone so lateral and
        # yaw motion keep their required hip freedom.
        self.rewards.standing_base_height_band.weight = -1.5
        self.rewards.standing_hip_pose = RewTerm(
            func=custom_mdp.standing_joint_deviation_normalized_l2,
            weight=-0.70,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FR_hip_joint",
                        "FL_hip_joint",
                        "RR_hip_joint",
                        "RL_hip_joint",
                    ],
                    preserve_order=True,
                ),
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 0.20,
                "position_std": 0.08,
                "max_error": 0.30,
            },
        )
        self.rewards.standing_orientation = RewTerm(
            func=custom_mdp.standing_orientation_normalized_l2,
            weight=-0.25,
            params={
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 0.20,
                "tilt_std_deg": 3.0,
            },
        )
        self.rewards.standing_foot_placement.weight = -0.70


@configclass
class RobotOmniTrotRobustStandFixPlayEnvCfg(RobotOmniTrotRobustStandFixEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotStandExpertEnvCfg(RobotOmniTrotClosedLoopRobustFoundationEnvCfg):
    """Independent upright expert trained only on exact zero velocity commands."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.rel_standing_envs = 1.0
        command.resampling_time_range = (20.0, 20.0)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.omni_velocity_cmd_levels = None

        # Learn recovery from the small posture and joint errors expected at a
        # locomotion-to-stand handoff, while leaving large falls to Recovery.
        self.events.reset_base.params["pose_range"].update(
            {"roll": (-0.10, 0.10), "pitch": (-0.10, 0.10)}
        )
        self.events.reset_base.params["velocity_range"].update(
            {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.10, 0.10),
                "roll": (-0.20, 0.20),
                "pitch": (-0.20, 0.20),
                "yaw": (-0.15, 0.15),
            }
        )
        self.events.reset_robot_joints.params["position_range"] = (0.85, 1.15)
        self.events.reset_robot_joints.params["velocity_range"] = (-0.30, 0.30)

        # No gait is rewarded in this task. The exported 51-D contract is
        # intentionally retained so the router and locomotion expert share the
        # same sensors, ordering, and deployment code.
        self.rewards.trot_contact_schedule = None
        self.rewards.trot_stance_swing_tracking = None
        self.rewards.feet_air_time_command_aware = None
        self.rewards.foot_clearance_style = None
        self.rewards.all_motion_swing_count = None
        self.rewards.pure_axis_velocity_decoupling = None
        self.rewards.track_velocity_components_relative_l1 = None
        self.rewards.yaw_overspeed_relative = None

        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.12
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.track_ang_vel_z.params["std"] = 0.12
        self.rewards.track_lin_vel_xy_l2.weight = -2.0
        self.rewards.track_ang_vel_z_l2.weight = -2.0
        self.rewards.inactive_velocity_axes_l2.weight = -5.0
        self.rewards.inactive_velocity_axes_l2.params["axis_weights"] = (1.0, 1.0, 1.0)

        self.rewards.speed_adaptive_base_height.weight = -100.0
        self.rewards.standing_base_height_band.weight = -4.0
        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.base_angular_velocity.weight = -0.50
        self.rewards.base_linear_velocity.weight = -4.0
        self.rewards.joint_pos.weight = -0.05
        self.rewards.joint_pos.params["stand_still_scale"] = 1.0
        self.rewards.action_rate.weight = -0.10
        self.rewards.action_smoothness_2.weight = -0.05
        self.rewards.energy.weight = -1.0e-4
        self.rewards.standing_foot_placement.weight = -0.80
        self.rewards.hip_outward_band.weight = -8.0

        self.rewards.standing_hip_pose = RewTerm(
            func=custom_mdp.standing_joint_deviation_normalized_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FR_hip_joint",
                        "FL_hip_joint",
                        "RR_hip_joint",
                        "RL_hip_joint",
                    ],
                    preserve_order=True,
                ),
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 1.0,
                "position_std": 0.08,
                "max_error": 0.30,
            },
        )
        self.rewards.standing_orientation = RewTerm(
            func=custom_mdp.standing_orientation_normalized_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 1.0,
                "tilt_std_deg": 3.0,
            },
        )
        self.rewards.standing_stable_support = RewTerm(
            func=custom_mdp.recovery_stable_support,
            weight=2.0,
            params={
                "prone_height": 0.20,
                "standing_height": 0.31,
                "angular_velocity_std": 0.35,
                "contact_force_threshold": 10.0,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                    preserve_order=True,
                ),
            },
        )


@configclass
class RobotStandExpertPlayEnvCfg(RobotStandExpertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotStandHeightCalibratedEnvCfg(RobotStandExpertEnvCfg):
    """Compensate the measured Isaac-to-MuJoCo standing-height offset."""

    def __post_init__(self):
        super().__post_init__()

        # StandExpert model_200 transfers 15-20 mm lower in MuJoCo while its
        # tilt, drift, and hip limits already pass. Shift only the training
        # target; deployment is still judged against the unchanged 0.31-0.335
        # m MuJoCo gate.
        self.rewards.speed_adaptive_base_height.params["standing_height"] = 0.35
        self.rewards.standing_base_height_band.weight = -6.0
        self.rewards.standing_base_height_band.params.update(
            {
                "lower_height": 0.335,
                "upper_height": 0.355,
                "error_std": 0.015,
            }
        )
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.standing_orientation.weight = -1.25
        self.rewards.hip_outward_band.weight = -10.0
        self.rewards.standing_hip_pose.weight = -1.25


@configclass
class RobotStandHeightCalibratedPlayEnvCfg(RobotStandHeightCalibratedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotStandHeightHipCalibratedEnvCfg(RobotStandHeightCalibratedEnvCfg):
    """Finish the transfer-height correction while restoring compact stand hips."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.speed_adaptive_base_height.params["standing_height"] = 0.365
        self.rewards.standing_base_height_band.weight = -8.0
        self.rewards.standing_base_height_band.params.update(
            {
                "lower_height": 0.350,
                "upper_height": 0.370,
                "error_std": 0.015,
            }
        )
        self.rewards.hip_outward_band.weight = -20.0
        self.rewards.standing_hip_pose.weight = -3.0


@configclass
class RobotStandHeightHipCalibratedPlayEnvCfg(RobotStandHeightHipCalibratedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg(
    RobotOmniTrotClosedLoopRobustFoundationEnvCfg
):
    """Opt-in Stage-A task with cross-leg contact and same-leg filtering."""

    def __post_init__(self):
        super().__post_init__()
        # Preserve the inherited actuator-delay configuration and replace only
        # the spawner contract that controls articulation self-contact.
        self.scene.robot.spawn = CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.copy()


@configclass
class RobotOmniTrotClosedLoopSelectiveCollisionPlayEnvCfg(
    RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopStandFixEnvCfg(RobotOmniTrotClosedLoopPolishA2EnvCfg):
    """Remove the zero-command hip/tilt exploit without changing locomotion."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.35
        # A2 yaw is accurate inside Isaac.  Keep its policy unchanged while
        # the remaining cross-simulator response gap is calibrated separately.
        self.rewards.yaw_overspeed_relative = None
        self.rewards.standing_hip_pose = RewTerm(
            func=custom_mdp.standing_joint_deviation_normalized_l2,
            weight=-0.50,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FR_hip_joint",
                        "FL_hip_joint",
                        "RR_hip_joint",
                        "RL_hip_joint",
                    ],
                    preserve_order=True,
                ),
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 0.20,
                "position_std": 0.06,
                "max_error": 0.30,
            },
        )
        self.rewards.standing_orientation = RewTerm(
            func=custom_mdp.standing_orientation_normalized_l2,
            weight=-0.35,
            params={
                "command_name": "base_velocity",
                "planar_command_threshold": 0.03,
                "yaw_command_threshold": 0.05,
                "velocity_threshold": 0.20,
                "tilt_std_deg": 3.0,
            },
        )


@configclass
class RobotOmniTrotClosedLoopStandFixPlayEnvCfg(RobotOmniTrotClosedLoopStandFixEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopStageBEnvCfg(
    RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg
):
    """Stage B envelope after the foundation grid passes sim2sim."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.ranges = command.Ranges(
            lin_vel_x=(-0.80, 0.80),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.50, 0.50),
        )
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-0.80, 0.80),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.50, 0.50),
        )
        command.low_speed_y_range = (0.03, 0.20)
        command.low_speed_yaw_range = (0.05, 0.30)


@configclass
class RobotOmniTrotClosedLoopStageBPlayEnvCfg(RobotOmniTrotClosedLoopStageBEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopStageCEnvCfg(RobotOmniTrotClosedLoopStageBEnvCfg):
    """Stage C envelope for medium-speed omni tracking."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.ranges = command.Ranges(
            lin_vel_x=(-1.50, 1.50),
            lin_vel_y=(-0.40, 0.40),
            ang_vel_z=(-1.00, 1.00),
        )
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-1.50, 1.50),
            lin_vel_y=(-0.40, 0.40),
            ang_vel_z=(-1.00, 1.00),
        )


@configclass
class RobotOmniTrotClosedLoopStageCPlayEnvCfg(RobotOmniTrotClosedLoopStageCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopStageDEnvCfg(RobotOmniTrotClosedLoopStageCEnvCfg):
    """Final user envelope; only use after A-C pass fixed-command gates."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.ranges = command.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.0, 2.0),
        )
        command.limit_ranges = command.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.0, 2.0),
        )


@configclass
class RobotOmniTrotClosedLoopStageDPlayEnvCfg(RobotOmniTrotClosedLoopStageDEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotClosedLoopGaitRobustEnvCfg(RobotOmniTrotClosedLoopStageDEnvCfg):
    """WTW-style command-adaptive gait shaping without changing the 51-D actor."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)

        # Frequency was already observable through the four-channel trot
        # clock.  Keep the accepted range explicit while adapting clearance
        # and landing width from the same velocity command.
        for clock in (self.observations.policy.trot_clock, self.observations.critic.trot_clock):
            clock.params.update({"min_frequency": 1.4, "max_frequency": 3.2, "full_speed": 3.0})
        for reward in (self.rewards.trot_contact_schedule, self.rewards.trot_stance_swing_tracking):
            reward.params.update({"min_frequency": 1.4, "max_frequency": 3.2, "full_speed": 3.0})

        self.rewards.foot_clearance_style.params.update(
            {
                "target_height": 0.045,
                "target_height_high": 0.080,
                "full_speed": 3.0,
                "std": 0.030,
            }
        )
        self.rewards.foot_clearance_style.weight = 0.30
        self.rewards.stance_foot_placement.params.update(
            {"lateral_gain": 1.15, "yaw_gain": 1.10}
        )
        self.rewards.stance_foot_placement.weight = -0.10


@configclass
class RobotOmniTrotClosedLoopGaitRobustPlayEnvCfg(
    RobotOmniTrotClosedLoopGaitRobustEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotDynamicsTeacherEnvCfg(RobotOmniTrotClosedLoopGaitRobustEnvCfg):
    """RMA-style teacher conditioned on the exact randomized dynamics."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.dynamics_context = ObsTerm(
            func=custom_mdp.privileged_dynamics_context,
            params={
                "startup_context_dim": 10,
                "maximum_delay_steps": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-2.0, 2.0),
        )
        # Event terms execute in configuration order.  This term is appended
        # after all inherited startup randomizers and records their result.
        self.events.record_dynamics_context = EventTerm(
            func=custom_mdp.record_privileged_dynamics_context,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_base_com": (
                    -0.00425258944579,
                    0.00128121327661,
                    -0.00191914629703,
                ),
            },
        )


@configclass
class RobotOmniTrotDynamicsTeacherPlayEnvCfg(RobotOmniTrotDynamicsTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotTerrainT0EnvCfg(RobotOmniTrotDynamicsTeacherEnvCfg):
    """Gated mild-terrain branch retaining 40 percent flat-ground samples."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = CUSTOM_DOG_TERRAIN_T0_CFG.copy()
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # The first 8/20 terrain columns are flat. They continue to rehearse
        # the full Stage-D envelope; only non-flat T0 columns are limited to B.
        command = self.commands.base_velocity
        command.ranges = command.limit_ranges
        command.flat_terrain_type_count = 8
        command.rough_terrain_ranges = command.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.5, 0.5),
        )
        # This reward measures foot z in a flat world frame.  On slopes and
        # stairs it would encode the wrong terrain height; contact-clock and
        # swing-count terms continue to require actual stepping.
        self.rewards.foot_clearance_style = None
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )


@configclass
class RobotOmniTrotTerrainT0PlayEnvCfg(RobotOmniTrotTerrainT0EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4


@configclass
class RobotOmniTrotTerrainT1EnvCfg(RobotOmniTrotTerrainT0EnvCfg):
    """Rougher blind-locomotion branch with 40 percent flat regression samples."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = CUSTOM_DOG_TERRAIN_T1_CFG.copy()
        # T1 has 8/20 flat columns. Flat environments retain D while rough,
        # slope and low-step environments expand only to the Stage-C range.
        command = self.commands.base_velocity
        command.flat_terrain_type_count = 8
        command.rough_terrain_ranges = command.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
        )


@configclass
class RobotOmniTrotTerrainT1PlayEnvCfg(RobotOmniTrotTerrainT1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 4


@configclass
class RobotClosedLoopHistory213DistillationEnvCfg(
    RobotOmniTrotTerrainT1EnvCfg
):
    """Distill the final terrain teacher into deployable proprio history."""

    observations: ClosedLoopHistoryDistillationObservationsCfg = (
        ClosedLoopHistoryDistillationObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.trot_clock = None
        self.observations.policy.base_lin_vel_xy = None
        self.observations.policy.dynamics_context = None
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ):
            term = getattr(self.observations.policy, name)
            term.history_length = 5
            term.flatten_history_dim = True
        self.observations.policy.velocity_commands.history_length = 1
        self.observations.policy.velocity_commands.flatten_history_dim = True

        self.observations.teacher.base_ang_vel.scale = (0.2, 0.2, 1.0)
        self.observations.teacher.base_ang_vel.noise = Unoise(n_min=-0.10, n_max=0.10)
        self.observations.teacher.dynamics_context = ObsTerm(
            func=custom_mdp.privileged_dynamics_context,
            params={
                "startup_context_dim": 10,
                "maximum_delay_steps": 2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-2.0, 2.0),
        )


@configclass
class RobotClosedLoopHistory213DistillationPlayEnvCfg(
    RobotClosedLoopHistory213DistillationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotPlayEnvCfg(RobotOmniTrotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotPostureEnvCfg(RobotOmniTrotEnvCfg):
    """Refine body height, hip width, standing, and low-rate pure-yaw gait."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the full validated command envelope while sampling standing and
        # pure-axis modes often enough to remove their current local optima.
        command = self.commands.base_velocity
        command.ranges = command.limit_ranges
        command.bucket_probabilities = (0.25, 0.20, 0.25, 0.30)
        command.rel_standing_envs = 0.12
        command.rel_low_speed_yaw = 0.60
        command.low_speed_yaw_range = (0.08, 0.40)

        # A small yaw request must still produce the same diagonal stepping
        # contract. With yaw_speed_scale=0.35 this activates near 0.08 rad/s.
        gait_threshold = 0.025
        self.observations.policy.trot_clock.params["command_threshold"] = gait_threshold
        self.observations.critic.trot_clock.params["command_threshold"] = gait_threshold
        self.rewards.trot_contact_schedule.params["command_threshold"] = gait_threshold
        self.rewards.trot_stance_swing_tracking.params["command_threshold"] = gait_threshold

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

        # Bilateral height tracking prevents both the tall low-speed posture
        # and the excessive high-speed crouch observed in MuJoCo.
        self.rewards.speed_adaptive_base_height.weight = -80.0
        self.rewards.speed_adaptive_base_height.params.update(
            {
                "standing_height": 0.33,
                "crouched_height": 0.28,
                "crouch_start_speed": 0.10,
                "crouch_full_speed": 3.0,
                "yaw_speed_scale": 0.35,
            }
        )

        self.rewards.hip_outward_speed_style = RewTerm(
            func=custom_mdp.hip_outward_speed_style_l2,
            weight=-3.0,
            params={
                "standing_limit": 0.16,
                "walking_limit": 0.22,
                "high_speed_limit": 0.30,
                "walking_speed": 0.35,
                "high_speed": 2.0,
                "lateral_limit_gain": 0.18,
                "yaw_limit_gain": 0.08,
                "command_name": "base_velocity",
            },
        )
        self.rewards.joint_pos = RewTerm(
            func=custom_mdp.joint_deviation_l2,
            weight=-0.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 4.0,
                "velocity_threshold": 0.20,
            },
        )
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.08, 0.04, 0.06),
                "axis_weights": (1.5, 2.0, 1.0),
            },
        )

        # Pure-yaw commands, including small ones, must contain an actual
        # swing phase instead of an all-feet-planted torsional shuffle.
        self.rewards.feet_air_time_command_aware = RewTerm(
            func=custom_mdp.feet_air_time_command_aware,
            weight=0.35,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 0.12,
                "command_name": "base_velocity",
                "command_threshold": gait_threshold,
                "yaw_speed_scale": 0.35,
            },
        )
        self.rewards.foot_clearance_style = RewTerm(
            func=custom_mdp.foot_clearance_speed_style,
            weight=0.25,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": feet_cfg,
                "target_height": 0.055,
                "std": 0.035,
                "command_name": "base_velocity",
                "command_threshold": gait_threshold,
                "yaw_speed_scale": 0.35,
            },
        )
        self.rewards.pure_axis_swing_count = RewTerm(
            func=custom_mdp.pure_axis_swing_count,
            weight=0.40,
            params={
                "sensor_cfg": contact_cfg,
                "command_name": "base_velocity",
                "forward_deadband": 0.06,
                "lateral_minimum": 0.08,
                "yaw_minimum": 0.08,
                "target_airborne": 2.0,
                "airborne_std": 0.75,
            },
        )


@configclass
class RobotOmniTrotPosturePlayEnvCfg(RobotOmniTrotPostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmniTrotRefineEnvCfg(RobotOmniTrotPostureEnvCfg):
    """Refine low-speed stepping, bounded hip motion, efficiency, and trunk stability."""

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.base_velocity
        command.minimum_command_magnitude = 0.03
        command.rel_low_speed_x = 0.65
        command.low_speed_x_range = (0.03, 0.20)
        command.rel_low_speed_y = 0.65
        command.low_speed_y_range = (0.03, 0.20)
        command.rel_low_speed_yaw = 0.70
        command.low_speed_yaw_range = (0.05, 0.30)
        command.rel_standing_envs = 0.12

        # Keep the external 49-D contract while making the clock active for
        # every deliberate command outside the joystick noise dead-zone.
        planar_deadband = 0.03
        yaw_deadband = 0.05
        for clock in (self.observations.policy.trot_clock, self.observations.critic.trot_clock):
            clock.params["command_threshold"] = planar_deadband
            clock.params["yaw_command_threshold"] = yaw_deadband
        for reward in (self.rewards.trot_contact_schedule, self.rewards.trot_stance_swing_tracking):
            reward.params["command_threshold"] = planar_deadband
            reward.params["yaw_command_threshold"] = yaw_deadband
        self.rewards.trot_contact_schedule.weight = 2.0
        self.rewards.trot_stance_swing_tracking.weight = 1.25

        # Keep deliberate low-speed commands visible to the tracking loss.
        self.rewards.track_velocity_components_relative_l1.weight = -1.0
        self.rewards.track_velocity_components_relative_l1.params["command_min"] = (
            0.03,
            0.03,
            0.05,
        )
        self.rewards.track_ang_vel_z_l2.weight = -1.0
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-0.75,
            params={"command_name": "base_velocity"},
        )

        # A band permits modest ab/adduction. Pure forward/reverse stays
        # compact, while lateral and yaw commands continuously widen the band.
        self.rewards.hip_outward_speed_style = None
        self.rewards.hip_outward_band = RewTerm(
            func=custom_mdp.hip_outward_band_l2,
            weight=-10.0,
            params={
                "standing_band": (-0.04, 0.16),
                "walking_band": (-0.06, 0.18),
                "high_speed_band": (-0.08, 0.22),
                "walking_speed": 0.35,
                "high_speed": 2.0,
                "lateral_allowance": 0.18,
                "yaw_allowance": 0.05,
                "command_name": "base_velocity",
            },
        )
        self.rewards.calf_pair_separation = RewTerm(
            func=custom_mdp.paired_lateral_separation_l2,
            weight=-3.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_calf", "FL_calf", "RR_calf", "RL_calf"],
                    preserve_order=True,
                ),
                "minimum_separation": 0.18,
                "violation_scale": 0.05,
                "lateral_allowance": 0.05,
                "yaw_allowance": 0.015,
                "command_name": "base_velocity",
            },
        )

        # Energy is an auxiliary style term. Trunk angle, roll/pitch rate and
        # vertical speed directly address body shake without suppressing yaw.
        self.rewards.energy.weight = -8.0e-5
        self.rewards.flat_orientation_l2.weight = -8.0
        self.rewards.base_angular_velocity.weight = -0.18
        self.rewards.base_linear_velocity.weight = -3.5
        self.rewards.action_rate.weight = -0.06
        self.rewards.action_smoothness_2.weight = -0.035
        self.rewards.speed_adaptive_base_height.weight = -100.0
        self.rewards.speed_adaptive_base_height.params["crouched_height"] = 0.29

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
        self.rewards.feet_air_time_command_aware.weight = 0.50
        self.rewards.feet_air_time_command_aware.params.update(
            {"command_threshold": planar_deadband, "yaw_speed_scale": 1.0}
        )
        self.rewards.foot_clearance_style.weight = 0.35
        self.rewards.foot_clearance_style.params.update(
            {"command_threshold": planar_deadband, "yaw_speed_scale": 1.0}
        )
        self.rewards.all_motion_swing_count = RewTerm(
            func=custom_mdp.motion_swing_count,
            weight=0.60,
            params={
                "sensor_cfg": contact_cfg,
                "command_name": "base_velocity",
                "planar_deadband": planar_deadband,
                "yaw_deadband": yaw_deadband,
                "target_airborne": 2.0,
                "airborne_std": 0.75,
            },
        )


@configclass
class RobotOmniTrotRefinePlayEnvCfg(RobotOmniTrotRefineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45V3PolishEnvCfg(RobotOmni45V2EnvCfg):
    """Refine the converged Omni-45 gait without changing its policy contract."""

    def __post_init__(self):
        super().__post_init__()

        # Keep all operator axes visible from the first resumed iteration. The
        # distribution matches real use: pure x, pure y, pure yaw and vx+wz.
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.35, 0.40, 0.15, 0.10),
            minimum_command_magnitude=0.08,
            rel_low_speed_x=0.50,
            low_speed_x_range=(0.15, 0.40),
            negative_x_probability=0.50,
            combined_include_lateral=False,
            rel_standing_envs=0.08,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-1.0, 1.0),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
        self.curriculum.omni_velocity_cmd_levels = None

        # Relative per-axis error prevents the large forward component from
        # hiding weak lateral tracking. Zero-command axes are constrained by a
        # separate drift term rather than by a pose target.
        self.rewards.track_velocity_components_relative_l1.weight = -2.0
        self.rewards.track_velocity_components_relative_l1.params.update(
            {
                "command_min": (0.08, 0.08, 0.12),
                "axis_weights": (1.25, 2.0, 1.0),
            }
        )
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.06, 0.04, 0.08),
                "axis_weights": (1.0, 1.5, 1.0),
            },
        )

        self.rewards.feet_slide.weight = -0.15
        self.rewards.action_smoothness_2.weight = -0.025
        soft_landing_params = dict(self.rewards.foot_soft_landing.params)
        soft_landing_params["yaw_speed_scale"] = 0.25
        self.rewards.foot_soft_landing = RewTerm(
            func=custom_mdp.FootSoftLandingPreviousVelocity,
            weight=-0.05,
            params=soft_landing_params,
        )
        self.rewards.foot_impact_velocity.weight = -0.03


@configclass
class RobotOmni45V3PolishPlayEnvCfg(RobotOmni45V3PolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1


@configclass
class RobotOmni45V3ConservativeEnvCfg(RobotOmni45V3PolishEnvCfg):
    """Lower-risk continuation after the aggressive v3 reward ablation."""

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.ranges.lin_vel_y = (-0.25, 0.25)
        command.limit_ranges.lin_vel_y = (-0.4, 0.4)
        command.bucket_probabilities = (0.45, 0.30, 0.15, 0.10)
        command.rel_low_speed_x = 0.35
        command.low_speed_x_range = (0.15, 0.35)

        self.rewards.track_velocity_components_relative_l1.weight = -0.75
        self.rewards.track_velocity_components_relative_l1.params.update(
            {"command_min": (0.08, 0.08, 0.12), "axis_weights": (1.0, 1.25, 1.0)}
        )
        self.rewards.inactive_velocity_axes_l2.weight = -0.50
        self.rewards.feet_slide.weight = -0.10
        self.rewards.action_rate.weight = -0.03
        self.rewards.action_smoothness_2.weight = -0.02
        self.rewards.foot_soft_landing.weight = -0.03
        self.rewards.foot_impact_velocity.weight = -0.02


@configclass
class RobotOmni45V3ConservativePlayEnvCfg(RobotOmni45V3ConservativeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45V2SteeringEnvCfg(RobotOmni45V2EnvCfg):
    """Teach lateral and yaw modulation while preserving the forward gait."""

    def __post_init__(self):
        super().__post_init__()

        limits = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity = custom_mdp.MovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.35, 0.30, 0.25, 0.10),
            rel_standing_envs=0.05,
            rel_low_speed_forward=0.45,
            low_speed_range=(0.15, 0.40),
            lateral_min_fraction=0.50,
            yaw_min_fraction=0.50,
            debug_vis=False,
            ranges=custom_mdp.MovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.75),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.40, 0.40),
            ),
            limit_ranges=limits,
        )
        # This bridge stage has a fixed envelope. The signed pure-axis
        # curriculum is enabled only after moving steering passes sim2sim.
        self.curriculum.omni_velocity_cmd_levels = None

        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 3.0
        self.rewards.track_ang_vel_z.params["std"] = 0.30
        self.rewards.track_velocity_components_relative_l1.weight = -3.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.5,
            1.25,
        )
        self.rewards.track_velocity_components_progress = RewTerm(
            func=custom_mdp.track_velocity_components_progress,
            weight=2.5,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.08, 0.12),
                "axis_weights": (1.0, 1.5, 1.25),
                "max_progress": (1.0, 0.40, 0.80),
            },
        )
        self.rewards.track_ang_vel_z_l2.weight = -2.0


@configclass
class RobotOmni45V2SteeringPlayEnvCfg(RobotOmni45V2SteeringEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45UsageEnvCfg(RobotOmni45V2SteeringEnvCfg):
    """First-stage 45-D policy for the robot's actual low-speed command usage."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = custom_mdp.UsageWeightedSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.25, 0.15, 0.50, 0.10),
            rel_standing_envs=0.05,
            rel_low_speed_forward=0.45,
            low_speed_range=(0.15, 0.40),
            lateral_min_fraction=0.50,
            yaw_min_fraction=0.50,
            debug_vis=False,
            ranges=custom_mdp.UsageWeightedSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.45),
                lin_vel_y=(-0.05, 0.05),
                ang_vel_z=(-0.15, 0.15),
            ),
            limit_ranges=custom_mdp.UsageWeightedSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.75),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.30, 0.30),
            ),
            speed_bin_edges=(0.35, 0.55),
            axis_success_thresholds=(0.10, 0.07, 0.10),
        )
        self.curriculum.omni_velocity_cmd_levels = None
        self.curriculum.usage_command_window_levels = CurrTerm(
            func=custom_mdp.usage_command_window_levels,
            params={
                "command_name": "base_velocity",
                "min_windows": 50,
                "success_rate_threshold": 0.70,
                "required_consecutive_windows": 3,
                "increments": (0.15, 0.025, 0.075),
            },
        )
        # Small first-stage lateral/yaw commands must participate in the
        # axis-separated rewards.  The inherited steering thresholds (0.08,
        # 0.12) excluded much of the initial +/-0.05 and +/-0.15 envelopes.
        small_command_thresholds = (0.10, 0.025, 0.05)
        self.rewards.track_velocity_components_relative_l1.params["command_min"] = (
            small_command_thresholds
        )
        self.rewards.track_velocity_components_progress.params["command_min"] = (
            small_command_thresholds
        )
        # The first-stage acceptance grid includes a simultaneous positive
        # lateral command.  Give that active axis a modestly stronger signal;
        # this remains a soft tracking term and does not constrain hip pose.
        usage_axis_weights = (1.0, 2.0, 1.5)
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            usage_axis_weights
        )
        self.rewards.track_velocity_components_progress.params["axis_weights"] = (
            usage_axis_weights
        )
        self.rewards.track_ang_vel_z_l2.weight = -3.0


@configclass
class RobotOmni45UsagePlayEnvCfg(RobotOmni45UsageEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni45UsageBidirectionalEnvCfg(RobotOmni45UsageEnvCfg):
    """Stage A continuation for low-speed reverse and vx+wz commands."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = custom_mdp.BidirectionalUsageWeightedSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            # Actual usage for this stage: straight vx and vx+wz only.
            steering_mode_probabilities=(0.40, 0.0, 0.60, 0.0),
            rel_standing_envs=0.08,
            rel_low_speed_forward=0.65,
            low_speed_range=(0.15, 0.40),
            rel_backward_envs=0.40,
            backward_speed_range=(0.15, 0.35),
            lateral_min_fraction=0.50,
            yaw_min_fraction=0.50,
            debug_vis=False,
            ranges=custom_mdp.BidirectionalUsageWeightedSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.45),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.20, 0.20),
            ),
            limit_ranges=custom_mdp.BidirectionalUsageWeightedSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.20, 0.20),
            ),
            speed_bin_edges=(0.20, 0.35),
            axis_success_thresholds=(0.10, 0.07, 0.10),
        )
        # Keep this experiment fixed. Expanding any axis would confound the
        # backward-gait test with a simultaneous command-curriculum change.
        self.curriculum.usage_command_window_levels = None
        self.rewards.track_velocity_components_progress.weight = 3.5
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.025, 0.05),
                "axis_weights": (1.0, 1.0, 1.0),
            },
        )


@configclass
class RobotOmni45UsageBidirectionalPlayEnvCfg(RobotOmni45UsageBidirectionalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45UsageReverseDiscoveryEnvCfg(RobotOmni45UsageBidirectionalEnvCfg):
    """Focused reverse-gait discovery before reintroducing steering commands."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        command_cfg.rel_standing_envs = 0.05
        command_cfg.rel_backward_envs = 0.75
        command_cfg.backward_speed_range = (0.15, 0.35)
        command_cfg.low_speed_range = (0.15, 0.35)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 0.35),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.35),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_velocity_components_progress.weight = 6.0
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        # Temporarily loosen generic style costs enough to discover a reversed
        # contact sequence. They remain active and are restored during polish.
        self.rewards.action_rate.weight = -0.02
        self.rewards.action_smoothness_2.weight = -0.01
        self.rewards.joint_pos.weight = -0.05
        self.rewards.stance_foot_placement.weight = -0.04


@configclass
class RobotOmni45UsageReverseDiscoveryPlayEnvCfg(RobotOmni45UsageReverseDiscoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45SignedFoundationEnvCfg(RobotOmni45UsageBidirectionalEnvCfg):
    """From-scratch 45-D foundation for signed vx and low-speed vx+wz."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (0.55, 0.0, 0.45, 0.0)
        command_cfg.rel_standing_envs = 0.10
        command_cfg.rel_backward_envs = 0.45
        command_cfg.backward_speed_range = (0.15, 0.35)
        command_cfg.low_speed_range = (0.15, 0.40)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 0.45),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.15, 0.15),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.15, 0.15),
        )
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_velocity_components_progress.weight = 3.0


@configclass
class RobotOmni45SignedFoundationPlayEnvCfg(RobotOmni45SignedFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45PriorityFoundationEnvCfg(RobotOmni45SignedFoundationEnvCfg):
    """From-scratch deployable foundation for the real joystick usage modes."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            # Pure vx and vx+wz dominate; lateral and pure yaw remain explicit.
            bucket_probabilities=(0.35, 0.20, 0.20, 0.25),
            combined_include_lateral=False,
            minimum_command_magnitude=0.08,
            negative_x_probability=0.25,
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.20, 0.20),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.20, 0.20),
            ),
        )
        self.curriculum.usage_command_window_levels = None

        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.30
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        for reward in (
            self.rewards.track_velocity_components_relative_l1,
            self.rewards.track_velocity_components_progress,
        ):
            reward.params["command_min"] = (0.08, 0.05, 0.08)
            reward.params["axis_weights"] = (1.0, 2.0, 1.5)
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_velocity_components_progress.weight = 4.0
        self.rewards.track_ang_vel_z_l2.weight = -4.0
        self.rewards.inactive_velocity_axes_l2.weight = -2.0

        # Keep body and contact quality active without prescribing hip angles
        # or body height during the capability-discovery stage.
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.action_rate.weight = -0.03
        self.rewards.action_smoothness_2.weight = -0.015
        self.rewards.feet_slide.weight = -0.08
        self.rewards.stance_foot_placement.weight = -0.06


@configclass
class RobotOmni45PriorityFoundationPlayEnvCfg(RobotOmni45PriorityFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45RoutedDistillationEnvCfg(RobotOmni45SignedFoundationEnvCfg):
    """Stage-A command distribution for merging forward and reverse experts."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            # Stage A only: signed vx and vx+wz. Pure lateral/yaw are added
            # after the merged gait passes its fixed-command acceptance grid.
            bucket_probabilities=(0.65, 0.0, 0.0, 0.35),
            minimum_command_magnitude=0.08,
            negative_x_probability=0.40,
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.20, 0.20),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.20, 0.20),
            ),
        )
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni45RoutedDistillationPlayEnvCfg(RobotOmni45RoutedDistillationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotHistory213SignedFoundationEnvCfg(RobotOmni45SignedFoundationEnvCfg):
    """Stage-A signed locomotion with deployable 100 ms proprioceptive history."""

    def __post_init__(self):
        super().__post_init__()

        # Five 50 Hz samples expose short-term motion without requiring a base
        # linear-velocity estimator. Commands are current-only. Isaac Lab
        # flattens each term oldest-to-newest before concatenating the terms.
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ):
            term = getattr(self.observations.policy, name)
            term.history_length = 5
            term.flatten_history_dim = True
        self.observations.policy.velocity_commands.history_length = 1
        self.observations.policy.velocity_commands.flatten_history_dim = True


@configclass
class RobotHistory213SignedFoundationPlayEnvCfg(RobotHistory213SignedFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45SignedPolishEnvCfg(RobotOmni45SignedFoundationEnvCfg):
    """Preserve signed vx while reducing drift, overshoot and rough motion."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (0.65, 0.0, 0.35, 0.0)

        # Discovery rewards make motion emerge but also reward overshoot. Make
        # command error and inactive-axis suppression dominant during polish.
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_velocity_components_progress.weight = 0.50
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.track_lin_vel_xy_l2 = RewTerm(
            func=custom_mdp.track_lin_vel_xy_l2,
            weight=-3.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_l2.weight = -6.0
        self.rewards.inactive_velocity_axes_l2.weight = -4.0
        self.rewards.inactive_velocity_axes_l2.params["axis_weights"] = (1.0, 2.0, 1.5)

        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.joint_pos.weight = -0.12
        self.rewards.action_rate.weight = -0.06
        self.rewards.action_smoothness_2.weight = -0.03
        self.rewards.feet_slide.weight = -0.12
        self.rewards.stance_foot_placement.weight = -0.12


@configclass
class RobotOmni45SignedPolishPlayEnvCfg(RobotOmni45SignedPolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45RoutedBalancePolishEnvCfg(RobotOmni45SignedPolishEnvCfg):
    """Reduce the routed policy's sustained reverse pitch without losing signed vx."""

    def __post_init__(self):
        super().__post_init__()

        # The routed student tracks reverse commands but inherits a nearly
        # constant 14 degree nose-down pitch from its reverse expert. Make
        # upright whole-body motion explicit while keeping velocity tracking
        # strong enough that standing still is not a cheaper solution.
        self.rewards.track_lin_vel_xy.weight = 8.0
        self.rewards.track_velocity_components_relative_l1.weight = -8.0
        self.rewards.track_lin_vel_xy_l2.weight = -5.0
        self.rewards.flat_orientation_l2.weight = -12.0
        self.rewards.base_angular_velocity.weight = -0.15

        # Preserve deployable, geometry-based style objectives. No hip-angle
        # or body-height target is introduced.
        self.rewards.joint_pos.weight = -0.08
        self.rewards.action_rate.weight = -0.08
        self.rewards.action_smoothness_2.weight = -0.04
        self.rewards.feet_slide.weight = -0.12
        self.rewards.stance_foot_placement.weight = -0.10


@configclass
class RobotOmni45RoutedBalancePolishPlayEnvCfg(RobotOmni45RoutedBalancePolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45SignedVxPolishEnvCfg(RobotOmni45SignedPolishEnvCfg):
    """Pure signed vx calibration before reintroducing vx+wz steering."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        command_cfg.rel_standing_envs = 0.10
        command_cfg.rel_backward_envs = 0.50
        command_cfg.backward_speed_range = (0.15, 0.35)
        command_cfg.low_speed_range = (0.15, 0.40)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 0.45), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        )
        self.rewards.track_lin_vel_xy.weight = 5.0
        self.rewards.track_velocity_components_progress.weight = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.inactive_velocity_axes_l2.weight = -8.0
        self.rewards.inactive_velocity_axes_l2.params["axis_weights"] = (1.0, 1.0, 1.0)
        self.rewards.action_rate.weight = -0.05
        self.rewards.action_smoothness_2.weight = -0.025


@configclass
class RobotOmni45SignedVxPolishPlayEnvCfg(RobotOmni45SignedVxPolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45UsageStage2EnvCfg(RobotOmni45UsageEnvCfg):
    """Retain the accepted low-speed policy while adding forward speed to 1.5 m/s."""

    def __post_init__(self):
        super().__post_init__()

        command_cfg = self.commands.base_velocity
        command_cfg.rel_low_speed_forward = 0.35
        command_cfg.low_speed_range = (0.15, 0.75)
        command_cfg.rel_high_speed_forward = 0.45
        command_cfg.high_speed_range = (0.90, 1.50)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 1.50),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.30, 0.30),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 1.50),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.30, 0.30),
        )
        command_cfg.speed_bin_edges = (0.35, 0.55, 0.75, 1.00, 1.25)

        # Active component tracking does not constrain an axis commanded at
        # zero. At high vx this allowed vx+wz samples to acquire lateral drift.
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-6.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.10, 0.025, 0.05),
                "axis_weights": (1.0, 2.0, 1.5),
            },
        )


@configclass
class RobotOmni45UsageStage2PlayEnvCfg(RobotOmni45UsageStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni45UsageStage2TurnEnvCfg(RobotOmni45UsageStage2EnvCfg):
    """Target high-speed vx+wz steering without introducing high-speed lateral combinations."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (0.20, 0.10, 0.65, 0.05)
        command_cfg.rel_low_speed_forward = 0.25
        command_cfg.rel_high_speed_forward = 0.55
        command_cfg.high_speed_modes = (0, 2)

        self.rewards.inactive_velocity_axes_l2.weight = -2.0
        self.rewards.track_ang_vel_z_l2.weight = -5.0
        self.rewards.high_speed_turn_lateral_drift_l2 = RewTerm(
            func=custom_mdp.high_speed_turn_lateral_drift_l2,
            weight=-12.0,
            params={
                "command_name": "base_velocity",
                "min_forward_speed": 0.75,
                "min_yaw_rate": 0.10,
                "lateral_command_deadband": 0.025,
            },
        )


@configclass
class RobotOmni45UsageStage2TurnPlayEnvCfg(RobotOmni45UsageStage2TurnEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni47VelocityFeedbackEnvCfg(RobotOmni45UsageStage2TurnEnvCfg):
    """Stage-2 ablation with measured body-frame x/y velocity appended to the actor."""

    def __post_init__(self):
        super().__post_init__()
        # Append after the frozen 45-D contract so a zero-padded first actor
        # layer reproduces model_4840 exactly before fine-tuning.
        self.observations.policy.base_lin_vel_xy = ObsTerm(
            func=custom_mdp.base_lin_vel_xy,
            scale=1.0,
            clip=(-4.0, 4.0),
        )


@configclass
class RobotOmni47VelocityFeedbackPlayEnvCfg(RobotOmni47VelocityFeedbackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni47FoundationEnvCfg(RobotOmni45UsageEnvCfg):
    """From-scratch low-speed foundation with measured planar velocity feedback."""

    def __post_init__(self):
        super().__post_init__()

        # The transfer experiment showed that appending velocity after a 45-D
        # policy has converged is too late: PPO changes the old gait before it
        # learns to use the new feedback.  This task learns the 47-D contract
        # from iteration zero in a fixed Stage-A command envelope.
        self.observations.policy.base_lin_vel_xy = ObsTerm(
            func=custom_mdp.base_lin_vel_xy,
            scale=1.0,
            clip=(-2.0, 2.0),
        )

        command_cfg = self.commands.base_velocity
        command_cfg.rel_standing_envs = 0.10
        command_cfg.steering_mode_probabilities = (0.35, 0.15, 0.45, 0.05)
        command_cfg.rel_low_speed_forward = 0.60
        command_cfg.low_speed_range = (0.15, 0.35)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(0.15, 0.45),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.15, 0.15),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(0.0, 0.45),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.15, 0.15),
        )
        command_cfg.speed_bin_edges = (0.30,)

        # Expansion is deliberately disabled for this controlled ablation.
        # A later task may widen one axis only after the fixed MuJoCo grid
        # proves that this policy has learned a genuine feedback controller.
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni47FoundationPlayEnvCfg(RobotOmni47FoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47FoundationYawEnvCfg(RobotOmni47FoundationEnvCfg):
    """Conservative Stage-A polish for small forward-plus-yaw commands."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.steering_mode_probabilities = (0.30, 0.10, 0.55, 0.05)
        command_cfg.rel_low_speed_forward = 0.60

        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_ang_vel_z_l2.weight = -5.0
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            2.0,
            2.0,
        )
        self.rewards.track_velocity_components_progress.params["axis_weights"] = (
            1.0,
            2.0,
            2.0,
        )


@configclass
class RobotOmni47FoundationYawPlayEnvCfg(RobotOmni47FoundationYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47BackwardEnvCfg(RobotOmni47FoundationEnvCfg):
    """Introduce only a bounded reverse band while preserving Stage-A usage."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            steering_mode_probabilities=(0.55, 0.10, 0.30, 0.05),
            rel_standing_envs=0.10,
            rel_low_speed_forward=0.60,
            low_speed_range=(0.15, 0.35),
            rel_backward_envs=0.25,
            backward_speed_range=(0.15, 0.30),
            lateral_min_fraction=0.50,
            yaw_min_fraction=0.50,
            debug_vis=False,
            ranges=custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.45),
                lin_vel_y=(-0.05, 0.05),
                ang_vel_z=(-0.15, 0.15),
            ),
            limit_ranges=custom_mdp.BidirectionalMovingSteeringVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.30, 0.45),
                lin_vel_y=(-0.05, 0.05),
                ang_vel_z=(-0.15, 0.15),
            ),
        )
        self.curriculum.usage_command_window_levels = None


@configclass
class RobotOmni47BackwardPlayEnvCfg(RobotOmni47BackwardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotHistory213DistillationEnvCfg(RobotOmni47BackwardEnvCfg):
    """Distill a velocity-aware 47-D teacher into deployable history."""

    observations: HistoryDistillationObservationsCfg = HistoryDistillationObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # RobotOmni47BackwardEnvCfg appends true planar velocity to the policy
        # group. Keep it only in the teacher and expose sensor history to the
        # deployable student.
        self.observations.policy.base_lin_vel_xy = None
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos_rel",
            "joint_vel_rel",
            "last_action",
        ):
            term = getattr(self.observations.policy, name)
            term.history_length = 5
            term.flatten_history_dim = True
        self.observations.policy.velocity_commands.history_length = 1
        self.observations.policy.velocity_commands.flatten_history_dim = True


@configclass
class RobotHistory213DistillationPlayEnvCfg(RobotHistory213DistillationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotHistory213OmniBidirectionalEnvCfg(RobotHistory213DistillationEnvCfg):
    """Deployable-history PPO task with the teacher's bidirectional envelope."""


@configclass
class RobotHistory213OmniBidirectionalPlayEnvCfg(RobotHistory213OmniBidirectionalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47AxisEnvCfg(RobotOmni47FoundationEnvCfg):
    """Stage-B 47-D continuation for signed axes and pure lateral/yaw motion."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.45, 0.20, 0.30, 0.05),
            minimum_command_magnitude=0.08,
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.60, 0.75),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.50, 0.50),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.60, 0.75),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.50, 0.50),
            ),
        )
        self.curriculum.omni_velocity_cmd_levels = None
        self.curriculum.usage_command_window_levels = None

        # Pure-axis commands need a sharper, non-saturating signal than the
        # broad Go2 exponential terms.  These weights remain soft and do not
        # prescribe a hip angle or a fixed body height.
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        self.rewards.track_velocity_components_relative_l1.params["command_min"] = (
            0.08,
            0.05,
            0.08,
        )
        self.rewards.track_velocity_components_progress.weight = 4.0
        self.rewards.track_velocity_components_progress.params["command_min"] = (
            0.08,
            0.05,
            0.08,
        )
        self.rewards.track_ang_vel_z_l2.weight = -4.0


@configclass
class RobotOmni47AxisPlayEnvCfg(RobotOmni47AxisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47SignedFoundationEnvCfg(RobotOmni47AxisEnvCfg):
    """From-scratch signed three-axis foundation in a deliberately small envelope."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.bucket_probabilities = (0.50, 0.15, 0.30, 0.05)
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )

        # Keep early exploration useful at the small signed limits.  The
        # actual final command envelope is introduced only after this grid is
        # stable in both Isaac and MuJoCo.
        self.rewards.track_velocity_components_relative_l1.weight = -3.0
        self.rewards.track_velocity_components_progress.weight = 3.0
        self.rewards.track_ang_vel_z_l2.weight = -3.0


@configclass
class RobotOmni47SignedFoundationPlayEnvCfg(RobotOmni47SignedFoundationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45PrivilegedRoutedDistillationEnvCfg(RobotOmni47SignedFoundationEnvCfg):
    """Teach deployable Omni-45 from forward and privileged omni experts."""

    observations: DeployablePrivilegedDistillationObservationsCfg = (
        DeployablePrivilegedDistillationObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.base_lin_vel_xy = None

        command_cfg = self.commands.base_velocity
        command_cfg.bucket_probabilities = (0.30, 0.20, 0.20, 0.30)
        command_cfg.combined_include_lateral = False
        command_cfg.negative_x_probability = 0.25
        command_cfg.rel_standing_envs = 0.10
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.45),
            lin_vel_y=(-0.10, 0.10),
            ang_vel_z=(-0.25, 0.25),
        )


@configclass
class RobotOmni45PrivilegedRoutedDistillationPlayEnvCfg(
    RobotOmni45PrivilegedRoutedDistillationEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47ReverseOmniTeacherEnvCfg(RobotOmni47FoundationYawEnvCfg):
    """True-velocity teacher stage for reverse plus pure-axis omni motion.

    This is intentionally separate from the earlier signed-from-scratch task:
    it starts from the accepted forward/yaw 47-D gait and introduces reverse
    and lateral commands with an explicit, reproducible mixture.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.45, 0.20, 0.30, 0.05),
            minimum_command_magnitude=0.08,
            negative_x_probability=0.40,
            rel_standing_envs=0.10,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.25, 0.25),
            ),
            limit_ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.45),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.25, 0.25),
            ),
        )
        self.curriculum.omni_velocity_cmd_levels = None
        self.curriculum.usage_command_window_levels = None

        # Keep the already learned forward/yaw behavior useful while making
        # reverse and lateral tracking non-saturating at this small envelope.
        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -4.0
        self.rewards.track_velocity_components_relative_l1.params["command_min"] = (
            0.08,
            0.05,
            0.08,
        )
        self.rewards.track_velocity_components_relative_l1.params["axis_weights"] = (
            1.0,
            1.5,
            1.5,
        )
        self.rewards.track_velocity_components_progress.weight = 2.0
        self.rewards.track_velocity_components_progress.params["command_min"] = (
            0.08,
            0.05,
            0.08,
        )
        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_ang_vel_z_l2.weight = -5.0
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.08, 0.05, 0.08),
                "axis_weights": (1.0, 1.0, 1.0),
            },
        )
        self.rewards.action_rate.weight = -0.04
        self.rewards.action_smoothness_2.weight = -0.02


@configclass
class RobotOmni47ReverseOmniTeacherPlayEnvCfg(RobotOmni47ReverseOmniTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47ReverseDiscoveryTeacherEnvCfg(RobotOmni47ReverseOmniTeacherEnvCfg):
    """Short reverse-only teacher stage before reintroducing omni buckets."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.bucket_probabilities = (1.0, 0.0, 0.0, 0.0)
        command_cfg.negative_x_probability = 0.65
        command_cfg.rel_standing_envs = 0.08
        command_cfg.ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.35),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        command_cfg.limit_ranges = command_cfg.Ranges(
            lin_vel_x=(-0.35, 0.35),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )
        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        self.rewards.track_velocity_components_progress.weight = 3.0
        self.rewards.track_velocity_components_progress.params["command_min"] = (0.08, 0.05, 0.08)
        self.rewards.track_ang_vel_z = None
        self.rewards.track_ang_vel_z_l2 = None
        self.rewards.inactive_velocity_axes_l2.weight = -2.0


@configclass
class RobotOmni47ReverseDiscoveryTeacherPlayEnvCfg(RobotOmni47ReverseDiscoveryTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni47SignedPolishEnvCfg(RobotOmni47SignedFoundationEnvCfg):
    """Convert signed-axis discovery into accurate, smooth and transferable motion."""

    def __post_init__(self):
        super().__post_init__()
        command_cfg = self.commands.base_velocity
        command_cfg.bucket_probabilities = (0.35, 0.25, 0.35, 0.05)

        # Discovery progress made signed behaviors emerge, but it also rewards
        # overshoot.  Retain a small directional term while making tracking
        # error the dominant objective for this polish stage.
        self.rewards.track_velocity_components_progress.weight = 0.50
        self.rewards.track_velocity_components_relative_l1.weight = -6.0
        self.rewards.track_ang_vel_z.weight = 5.0
        self.rewards.track_ang_vel_z.params["std"] = 0.22
        self.rewards.track_ang_vel_z_l2.weight = -8.0
        self.rewards.inactive_velocity_axes_l2 = RewTerm(
            func=custom_mdp.inactive_velocity_axes_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_min": (0.08, 0.05, 0.08),
                "axis_weights": (1.0, 1.5, 1.5),
            },
        )

        # Generic physical style terms only: no explicit hip or body-height
        # target.  The feet and whole-body posture are constrained instead.
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.joint_pos.weight = -0.12
        self.rewards.action_rate.weight = -0.06
        self.rewards.action_smoothness_2.weight = -0.03
        self.rewards.feet_slide.weight = -0.12
        self.rewards.stance_foot_placement.weight = -0.12


@configclass
class RobotOmni47SignedPolishPlayEnvCfg(RobotOmni47SignedPolishEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotOmni45V2PureAxisEnvCfg(RobotOmni45V2SteeringEnvCfg):
    """Introduce pure lateral and yaw buckets after moving steering works."""

    def __post_init__(self):
        super().__post_init__()

        limits = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity = custom_mdp.StratifiedOmniVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 5.0),
            recovery_duration_s=0.0,
            omni_mixture=False,
            bucket_probabilities=(0.25, 0.30, 0.30, 0.15),
            minimum_command_magnitude=0.08,
            rel_standing_envs=0.05,
            debug_vis=False,
            ranges=custom_mdp.StratifiedOmniVelocityCommandCfg.Ranges(
                lin_vel_x=(0.15, 0.75),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.40, 0.40),
            ),
            limit_ranges=limits,
        )
        self.curriculum.omni_velocity_cmd_levels = None

        self.rewards.track_lin_vel_xy.params["std"] = 0.25
        self.rewards.track_ang_vel_z.weight = 4.0
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        self.rewards.track_velocity_components_relative_l1.weight = -5.0
        self.rewards.track_velocity_components_progress.weight = 4.0
        self.rewards.track_ang_vel_z_l2.weight = -3.0


@configclass
class RobotOmni45V2PureAxisPlayEnvCfg(RobotOmni45V2PureAxisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
