from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/velocity_env_cfg.py"
EVENTS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/events.py"
OBSERVATIONS = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/mdp/observations.py"
PPO = ROOT / "rl/src/custom_dog_rl/agents/ppo_cfg.py"
REGISTRY = ROOT / "rl/src/custom_dog_rl/tasks/locomotion/__init__.py"
SIM2SIM = ROOT / "sim2sim/custom_dog/run_sim2sim.py"


def _source(path: Path, name: str, kind: type[ast.AST]) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in module.body if isinstance(node, kind) and node.name == name)
    return ast.unparse(node)


def test_context_is_recorded_after_randomizers_and_has_live_delay() -> None:
    event = _source(EVENTS, "record_privileged_dynamics_context", ast.FunctionDef)
    observation = _source(OBSERVATIONS, "privileged_dynamics_context", ast.FunctionDef)
    assert "get_masses" in event
    assert "get_coms" in event
    assert "actuator.stiffness" in event and "actuator.damping" in event
    assert "joint_friction_coeff" in event
    assert "get_material_properties" in event
    assert "positions_delay_buffer" in observation
    assert "startup_context_dim" in observation


def test_dynamics_teacher_expands_51d_policy_to_62d_without_symmetry() -> None:
    env = _source(ENV, "RobotOmniTrotDynamicsTeacherEnvCfg", ast.ClassDef)
    runner = _source(PPO, "CustomDogDynamicsTeacherPPORunnerCfg", ast.ClassDef)
    script = (ROOT / "scripts/train_dynamics_teacher_stage.sh").read_text(encoding="utf-8")
    assert "RobotOmniTrotClosedLoopGaitRobustEnvCfg" in env
    assert "self.observations.policy.dynamics_context" in env
    assert env.index("super().__post_init__()") < env.index("record_dynamics_context")
    assert "self.algorithm.symmetry_cfg = None" in runner
    assert "--source-dim 51 --target-dim 62" in script
    assert "CUSTOM_DOG_LOAD_OPTIMIZER=0" in script


def test_terrain_uses_privileged_teacher_but_student_remains_213d() -> None:
    terrain = _source(ENV, "RobotOmniTrotTerrainT0EnvCfg", ast.ClassDef)
    distill = _source(ENV, "RobotClosedLoopHistory213DistillationEnvCfg", ast.ClassDef)
    assert "RobotOmniTrotDynamicsTeacherEnvCfg" in terrain
    assert "self.observations.policy.dynamics_context = None" in distill
    assert "self.observations.teacher.dynamics_context" in distill
    assert "term.history_length = 5" in distill


def test_registry_sim2sim_and_pipeline_include_dynamics_stage() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    sim2sim = SIM2SIM.read_text(encoding="utf-8")
    queue = (ROOT / "scripts/queue_gated_training_pipeline.sh").read_text(encoding="utf-8")
    terrain_gate = (ROOT / "scripts/train_next_terrain_stage_gated.sh").read_text(
        encoding="utf-8"
    )
    assert "CustomDog-Velocity-OmniTrot-DynamicsTeacher-v1" in registry
    assert '"trot_clock", "base_lin_vel_xy", "dynamics_context"' in sim2sim
    assert "dynamics_teacher" in queue
    assert "dynamics_teacher_selection.json" in terrain_gate


def test_recorded_context_has_normalized_values_and_exact_width() -> None:
    torch = pytest.importorskip("torch")
    module = ast.parse(EVENTS.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "record_privileged_dynamics_context"
    )
    function.decorator_list = []

    class SceneEntityCfg:
        def __init__(self, name: str):
            self.name = name

    namespace = {
        "torch": torch,
        "ManagerBasedEnv": object,
        "SceneEntityCfg": SceneEntityCfg,
        "Articulation": object,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(EVENTS), "exec"), namespace)

    nominal_com = (-0.00425258944579, 0.00128121327661, -0.00191914629703)
    default_mass = torch.tensor([[6.0, 4.0], [6.0, 4.0]])

    class RootView:
        def get_masses(self):
            return torch.tensor([[6.6, 4.4], [5.4, 3.6]])

        def get_coms(self):
            values = torch.zeros((2, 2, 7))
            values[:, 0, :3] = torch.tensor(nominal_com) + torch.tensor(
                [[0.015, -0.015, 0.010], [-0.015, 0.015, -0.010]]
            )
            return values

        def get_material_properties(self):
            centers = torch.tensor([0.875, 0.825, 0.04])
            return centers.view(1, 1, 3).repeat(2, 3, 1)

    class Data:
        pass

    Data.default_mass = default_mass
    Data.default_joint_stiffness = torch.full((2, 2), 25.0)
    Data.default_joint_damping = torch.full((2, 2), 0.5)
    Data.joint_friction_coeff = torch.full((2, 2), 0.015)

    class Actuator:
        joint_indices = slice(None)
        stiffness = torch.tensor([[28.75, 28.75], [21.25, 21.25]])
        damping = torch.tensor([[0.675, 0.675], [0.325, 0.325]])

    class Asset:
        device = "cpu"
        root_physx_view = RootView()
        data = Data()
        actuators = {"legs": Actuator()}

        @staticmethod
        def find_bodies(name: str):
            assert name == "base"
            return [0], ["base"]

    class Scene(dict):
        pass

    class Env:
        num_envs = 2
        scene = Scene(robot=Asset())

    env = Env()
    namespace["record_privileged_dynamics_context"](
        env, None, nominal_base_com=nominal_com, asset_cfg=SceneEntityCfg("robot")
    )
    context = env._custom_dog_startup_dynamics_context
    assert context.shape == (2, 10)
    assert torch.allclose(context[:, 0], torch.tensor([1.0, -1.0]), atol=1.0e-5)
    assert torch.allclose(context[:, 1:4], torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]))
    assert torch.allclose(context[:, 4:6], torch.tensor([[1.0, 1.0], [-1.0, -1.0]]), atol=1.0e-5)
    assert torch.allclose(context[:, 6], torch.full((2,), 0.5))
    assert torch.allclose(context[:, 7:], torch.zeros((2, 3)), atol=1.0e-5)
