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

gym.register(
    id="CustomDog-Velocity-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRobustEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRobustPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Gait-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotGaitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotGaitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Speed-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedHigh-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedHighEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedHighPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedFull-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedFullEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedFullPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)
