"""Asset configuration for the custom quadruped."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg

from .collision_contract import filtered_body_pairs


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
CUSTOM_DOG_DESCRIPTION_DIR = Path(
    os.environ.get(
        "CUSTOM_DOG_DESCRIPTION_DIR",
        _PROJECT_ROOT / "ros2" / "src" / "custom_dog_description",
    )
).resolve()


@configclass
class GoM8010DirectCfg(unitree_actuators.UnitreeActuatorCfg_Go2HV):
    """Conservative GO-M8010-6 output-side torque-speed envelope."""

    # Unitree specifies 30 rad/s and 23.7 Nm at the built-in 6.33:1
    # output. The existing Unitree RL curve uses 23.4 Nm as its limit.
    X1 = 13.5
    X2 = 30.0
    Y1 = 20.2
    Y2 = 23.4


@configclass
class GoM8010Calf2To1Cfg(GoM8010DirectCfg):
    """GO-M8010-6 curve reflected through the calf's 1:2 transmission."""

    # These are the Go2HV envelope values reflected through a 2:1
    # output-speed reduction.  Configclass fields are instance values,
    # not attributes available on the parent class.
    X1 = 6.75
    X2 = 15.0
    Y1 = 40.4
    Y2 = 46.8


CUSTOM_DOG_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        # Isaac uses calibrated primitive leg proxies. The canonical URDF keeps
        # the complete engine-neutral collision model for ROS and Pinocchio.
        asset_path=str(
            CUSTOM_DOG_DESCRIPTION_DIR
            / "urdf"
            / "custom_dog_selective_collision.urdf"
        ),
        collider_type="convex_hull",
        # The current CAD visual meshes overlap around adjacent joint housings.
        # Keep illegal ground-contact checks, but disable articulation self-contact.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Computed from this robot's thigh/calf lengths and foot-frame sphere
        # centers; Go2's 0.4 m base height leaves this model floating.
        pos=(0.0, 0.0, 0.324),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO_M8010_6_direct": GoM8010DirectCfg(
            joint_names_expr=[".*_(hip|thigh)_joint"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
        "GO_M8010_6_calf_2_to_1": GoM8010Calf2To1Cfg(
            joint_names_expr=[".*_calf_joint"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
    },
    joint_sdk_names=[
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
    ],
)


@sim_utils.clone
def _spawn_custom_dog_with_selective_self_collisions(
    prim_path: str,
    cfg: UnitreeUrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn one source robot and author same-leg collision filters before cloning."""

    from pxr import Sdf, UsdPhysics

    prim = sim_utils.spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    stage = prim.GetStage()
    missing_bodies = []
    for body_a, body_b in filtered_body_pairs():
        prim_a = stage.GetPrimAtPath(f"{prim_path}/{body_a}")
        prim_b = stage.GetPrimAtPath(f"{prim_path}/{body_b}")
        if not prim_a.IsValid():
            missing_bodies.append(body_a)
            continue
        if not prim_b.IsValid():
            missing_bodies.append(body_b)
            continue
        # PhysX's own addPairFilter helper authors the relationship on both
        # endpoints.  A single directed USD target is visible to inspection
        # but does not reliably suppress articulation-link contact.
        for source, target in ((prim_a, prim_b), (prim_b, prim_a)):
            filter_api = UsdPhysics.FilteredPairsAPI.Apply(source)
            filter_api.CreateFilteredPairsRel().AddTarget(Sdf.Path(target.GetPath()))
    if missing_bodies:
        names = ", ".join(sorted(set(missing_bodies)))
        raise ValueError(f"Selective collision asset is missing rigid bodies: {names}")
    return prim


# Opt-in self-collision variant. Both Isaac configurations use the same calibrated
# proxy URDF; this one additionally enables cross-leg contact and authors filters
# for base-to-leg and same-leg pairs.
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG = CUSTOM_DOG_CFG.copy()
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn = CUSTOM_DOG_CFG.spawn.copy()
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.asset_path = str(
    CUSTOM_DOG_DESCRIPTION_DIR / "urdf" / "custom_dog_selective_collision.urdf"
)
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.func = (
    _spawn_custom_dog_with_selective_self_collisions
)
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.self_collision = True
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.articulation_props = (
    CUSTOM_DOG_CFG.spawn.articulation_props.copy()
)
CUSTOM_DOG_SELECTIVE_SELF_COLLISION_CFG.spawn.articulation_props.enabled_self_collisions = True


# Keep historical checkpoints on their original +/-0.1 rad hip offsets.  The
# compact variant changes only the nominal standing/action offset; motor gains,
# limits and all physical parameters remain identical to CUSTOM_DOG_CFG.
CUSTOM_DOG_COMPACT_HIP_CFG = CUSTOM_DOG_CFG.copy()
CUSTOM_DOG_COMPACT_HIP_CFG.init_state = CUSTOM_DOG_CFG.init_state.copy()
CUSTOM_DOG_COMPACT_HIP_CFG.init_state.joint_pos = dict(CUSTOM_DOG_CFG.init_state.joint_pos)
CUSTOM_DOG_COMPACT_HIP_CFG.init_state.joint_pos[".*R_hip_joint"] = -0.05
CUSTOM_DOG_COMPACT_HIP_CFG.init_state.joint_pos[".*L_hip_joint"] = 0.05
