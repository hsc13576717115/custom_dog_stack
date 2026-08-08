"""Velocity-tracking task registration for the custom quadruped."""

import gymnasium as gym


gym.register(
    id="CustomDog-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)
