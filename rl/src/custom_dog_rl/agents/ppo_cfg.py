"""RSL-RL PPO configuration for velocity tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from custom_dog_rl.tasks.locomotion.mdp.symmetry import compute_symmetric_states


@configclass
class CustomDogPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "custom_dog_velocity"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CustomDogFineTunePPORunnerCfg(CustomDogPPORunnerCfg):
    """Conservative optimizer and checkpoint cadence for short fine-tunes."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 5
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogRecoveryPPORunnerCfg(CustomDogPPORunnerCfg):
    """Moderate-rate PPO settings for recovery plus omni-directional adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01


@configclass
class CustomDogOmniPPORunnerCfg(CustomDogPPORunnerCfg):
    """Exploratory adaptation settings for standing-start omni pretraining."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01


@configclass
class CustomDogLateralPPORunnerCfg(CustomDogPPORunnerCfg):
    """Low-rate adaptation for learning lateral motion without gait collapse."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogOmniSymmetryPPORunnerCfg(CustomDogPPORunnerCfg):
    """From-scratch PPO with left-right augmentation for omni locomotion."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 25
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_symmetric_states,
        )


@configclass
class CustomDogOmni45V2PPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """From-scratch PPO settings for the 45-D omni reward migration."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.num_learning_epochs = 5
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01


@configclass
class CustomDogOmni45V3PolishPPORunnerCfg(CustomDogOmni45V2PPORunnerCfg):
    """Low-rate mirror-regularized refinement of a converged 45-D policy."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.004
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.05


@configclass
class CustomDogOmniPhaseFineTunePPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """Conservative symmetric fine-tune after an omni gait has emerged."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogControlFineTunePPORunnerCfg(CustomDogOmniPhaseFineTunePPORunnerCfg):
    """Low-rate policy refinement after a stable three-axis gait exists."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class CustomDogPolishFineTunePPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Very small symmetric updates for refining an accepted controller."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogNaturalGaitPPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Low-rate model800 continuation with immutable 20-iteration checkpoints."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 2.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45StylePPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Conservative 45-D continuation of the validated 0-3 m/s policy."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 1.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45AxisPPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Axis-discovery continuation after the 45-D forward gait is preserved."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class CustomDogAxisPhaseBridgePPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Conservative phase-column adaptation of the validated 45-D forward actor."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 1.0e-3
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45NoSymmetryPPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """45-D omni fine-tune for CADs whose left/right motor axes are not canonical."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = None
        self.save_interval = 20
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class CustomDogOmni45UsagePPORunnerCfg(CustomDogOmni45NoSymmetryPPORunnerCfg):
    """Conservative first-stage continuation for the real usage distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class CustomDogOmni45UsageStage2PPORunnerCfg(CustomDogOmni45UsagePPORunnerCfg):
    """Low-rate continuation for the accepted Usage-v1 policy at higher vx."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.0025


@configclass
class CustomDogOmni45UsageBidirectionalPPORunnerCfg(CustomDogOmni45UsagePPORunnerCfg):
    """Conservative low-speed reverse continuation of the accepted model4840."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.0025


@configclass
class CustomDogOmni45UsageReverseDiscoveryPPORunnerCfg(CustomDogOmni45UsagePPORunnerCfg):
    """Exploratory optimizer for a bounded reverse-gait discovery stage."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.entropy_coef = 1.0e-3
        self.algorithm.desired_kl = 0.004


@configclass
class CustomDogOmni45SignedFoundationPPORunnerCfg(CustomDogPPORunnerCfg):
    """From-scratch PPO settings for the deployable signed 45-D foundation."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 50
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.num_learning_epochs = 5
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01
        self.algorithm.symmetry_cfg = None


@configclass
class CustomDogHistory213SignedFoundationPPORunnerCfg(
    CustomDogOmni45SignedFoundationPPORunnerCfg
):
    """From-scratch PPO for the 5-frame deployable proprioceptive contract."""


@configclass
class CustomDogHistory213TransferPPORunnerCfg(CustomDogOmni45NoSymmetryPPORunnerCfg):
    """Conservative optimizer after mapping a stable 45-D actor into history."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogHistory213DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Behavior-clone a privileged 47-D teacher into the 213-D student."""

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 20
    experiment_name = "custom_dog_velocity"
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.05,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="huber",
        optimizer="adam",
    )


@configclass
class CustomDogOmni45RoutedDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Merge frozen forward and reverse experts into one deployable 45-D actor."""

    num_steps_per_env = 24
    max_iterations = 100
    save_interval = 20
    experiment_name = "custom_dog_velocity"
    obs_groups = {"policy": ["policy"], "teacher": ["policy"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.03,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-5,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="huber",
        optimizer="adam",
    )


@configclass
class CustomDogOmni45PrivilegedRoutedDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill routed 45-D forward and 47-D omni experts into a 45-D actor."""

    num_steps_per_env = 24
    max_iterations = 120
    save_interval = 20
    experiment_name = "custom_dog_velocity"
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.03,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-5,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="huber",
        optimizer="adam",
    )


@configclass
class CustomDogHistory213OmniBidirectionalPPORunnerCfg(CustomDogHistory213TransferPPORunnerCfg):
    """Conservative PPO continuation after privileged-history distillation."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45SignedPolishPPORunnerCfg(CustomDogOmni45NoSymmetryPPORunnerCfg):
    """Conservative optimizer for retaining the discovered signed gait."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45SignedVxPolishPPORunnerCfg(CustomDogOmni45NoSymmetryPPORunnerCfg):
    """Low-rate optimizer for pure signed forward/backward drift calibration."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 2.0e-4
        self.algorithm.desired_kl = 0.0015


@configclass
class CustomDogOmni47VelocityFeedbackPPORunnerCfg(CustomDogOmni45UsageStage2PPORunnerCfg):
    """Controlled 47-D continuation with only actor velocity feedback changed."""


@configclass
class CustomDogOmni47FoundationPPORunnerCfg(CustomDogPPORunnerCfg):
    """From-scratch PPO settings for the low-speed 47-D feedback foundation."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 50
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.num_learning_epochs = 5
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01
        self.algorithm.symmetry_cfg = None


@configclass
class CustomDogOmni47SignedTransferPPORunnerCfg(CustomDogOmni47FoundationPPORunnerCfg):
    """Low-rate continuation of a signed 45-D policy after velocity expansion."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.policy.init_noise_std = 0.1
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni47AxisPPORunnerCfg(CustomDogOmni47FoundationPPORunnerCfg):
    """Low-rate Stage-B continuation after the 47-D foundation gait."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 25
        self.policy.init_noise_std = 0.30
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogOmni47FoundationYawPPORunnerCfg(CustomDogOmni47AxisPPORunnerCfg):
    """Very small Stage-A updates that preserve the accepted foundation gait."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni47ReverseOmniTeacherPPORunnerCfg(CustomDogOmni47FoundationYawPPORunnerCfg):
    """Conservative teacher continuation for explicit reverse and pure axes."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.policy.init_noise_std = 0.12
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class CustomDogOmni47BackwardPPORunnerCfg(CustomDogOmni47AxisPPORunnerCfg):
    """Single-capability continuation for the first bounded reverse band."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.004


@configclass
class CustomDogOmni47SignedPolishPPORunnerCfg(CustomDogOmni47AxisPPORunnerCfg):
    """Conservative optimizer for preserving signed behaviors during polish."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 3.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni47SignedSymmetryPPORunnerCfg(CustomDogOmni47FoundationPPORunnerCfg):
    """From-scratch signed 47-D PPO with corrected left-right augmentation."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_symmetric_states,
        )


@configclass
class CustomDogOmni45NoSymmetryPolishPPORunnerCfg(CustomDogOmni45NoSymmetryPPORunnerCfg):
    """Low-rate steering decoupling after lateral/yaw responses emerge."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.entropy_coef = 1.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class CustomDogOmni45AxisBootstrapPPORunnerCfg(CustomDogControlFineTunePPORunnerCfg):
    """Exploratory 45-D continuation for discovering lateral and yaw motion."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.num_learning_epochs = 4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.desired_kl = 0.008


@configclass
class CustomDogOmni45AxisDirectionPPORunnerCfg(CustomDogOmni45AxisPPORunnerCfg):
    """Moderate exploration for pure-axis foot-placement discovery."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 20
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.num_learning_epochs = 3
        self.algorithm.entropy_coef = 2.0e-3
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogCompactFoundationPPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """Low-noise from-scratch PPO for the compact foundation stage."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 25
        self.policy.init_noise_std = 0.2
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.desired_kl = 0.01


@configclass
class CustomDogCompactMotionPPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """Conservative symmetric continuation from a stable compact foundation."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.002
        self.algorithm.desired_kl = 0.005


@configclass
class CustomDogLeggedGymScratchPPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """From-scratch PPO for the staged legged-gym reward baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 25
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.005


@configclass
class CustomDogAxisBootstrapPPORunnerCfg(CustomDogOmniSymmetryPPORunnerCfg):
    """Symmetric adaptation settings for discovering a missing command axis."""

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 10
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.desired_kl = 0.01
