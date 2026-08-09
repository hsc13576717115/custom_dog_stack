"""RSL-RL PPO configuration for velocity tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
