"""Velocity-tracking task registration for the custom quadruped."""

import gymnasium as gym


gym.register(
    id="CustomDog-Go2Reference-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Go2ReferenceEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Go2ReferencePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)


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

gym.register(
    id="CustomDog-Velocity-SpeedStraight-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedStraightEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedStraightPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedBalancedTune-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedBalancedTuneEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedBalancedTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniStyle45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniStyle45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniStyle45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45StylePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxis45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxis45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxis45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45AxisPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisQuality45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisQuality45EnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisQuality45PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisQualityAware45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisQualityAware45EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisQualityAware45PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisSwingDiscovery45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisSwingDiscovery45EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisSwingDiscovery45PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45AxisBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisSwingDirection45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisSwingDirection45EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisSwingDirection45PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45AxisDirectionPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-AxisPhaseBridge-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisPhaseBridgeEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisPhaseBridgePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogAxisPhaseBridgePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisStylePolish45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisStylePolish45EnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisStylePolish45PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniAxisBootstrap45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisBootstrap45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniAxisBootstrap45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45AxisBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniMoving45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniMoving45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniMoving45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45AxisPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniMovingDisentangled45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniMovingDisentangled45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniMovingDisentangled45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniSteeringPolish45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniSteeringPolish45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniSteeringPolish45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPolishPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniBidirectional45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniBidirectional45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniBidirectional45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPolishPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-SpeedOmniBackwardDiscovery45-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniBackwardDiscovery45EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSpeedOmniBackwardDiscovery45PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniCurriculum-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniCurriculumEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniCurriculumPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniPretrain-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPretrainEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPretrainPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniFoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniFoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniSymmetry-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniSymmetryEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniSymmetryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniPhase-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPhaseEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPhasePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LeggedGymPhaseTune-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymPhaseTuneEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymPhaseTunePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPhaseFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniRefine-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniRefineEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniRefinePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPhaseFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniRefineCompactHip-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniRefineCompactHipEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniRefineCompactHipPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactHeightPolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactHeightPolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactHeightPolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactOmniAdaptiveHeight-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactOmniAdaptiveHeightEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactOmniAdaptiveHeightPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniStabilityStage1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniStabilityStage2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniStabilityStage3-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage3EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-NaturalGait-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotNaturalGaitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotNaturalGaitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogNaturalGaitPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniStabilityStage4-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage4EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniStabilityStage5-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage5EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniStabilityStage5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniPolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactFoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactFoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogCompactFoundationPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactMotion-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactMotionEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactMotionPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogCompactMotionPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactForwardBootstrap-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactForwardBootstrapEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactForwardBootstrapPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogPolishFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactPhaseScratch-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhaseScratchEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhaseScratchPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactPhasePosture-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhasePostureEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhasePosturePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPhaseFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-CompactPhaseHeight-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhaseHeightEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotCompactPhaseHeightPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogControlFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniPhasePosture-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPhasePostureEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniPhasePosturePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPhaseFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LeggedGymStage1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogLeggedGymScratchPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LeggedGymStage2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniPhaseFineTunePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LeggedGymStage2Lateral-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage2LateralEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLeggedGymStage2LateralPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogAxisBootstrapPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LateralBootstrap-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLateralBootstrapEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLateralBootstrapPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogLateralPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-LateralBootstrap-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLateralBootstrapV2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotLateralBootstrapV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogLateralPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-RecoveryOmni-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRecoveryOmniEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRecoveryOmniPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogRecoveryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-RecoveryOmniFull-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRecoveryOmniFullEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotRecoveryOmniFullPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogRecoveryPPORunnerCfg",
    },
)


for stage, env_cfg, play_cfg in (
    ("R0", "RobotSelfRightingR0EnvCfg", "RobotSelfRightingR0PlayEnvCfg"),
    ("R1", "RobotSelfRightingR1EnvCfg", "RobotSelfRightingR1PlayEnvCfg"),
    ("R2", "RobotSelfRightingR2EnvCfg", "RobotSelfRightingR2PlayEnvCfg"),
):
    gym.register(
        id=f"CustomDog-SelfRighting-{stage}-v2",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{env_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{play_cfg}",
            "rsl_rl_cfg_entry_point": (
                "custom_dog_rl.agents.ppo_cfg:CustomDogSelfRightingPPORunnerCfg"
            ),
        },
    )


gym.register(
    id="CustomDog-Velocity-ClosedLoop-History213-Distill-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotClosedLoopHistory213DistillationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotClosedLoopHistory213DistillationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:"
            "CustomDogClosedLoopHistory213DistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45V2PPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-HighSpeed-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45HighSpeedEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45HighSpeedPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45HighSpeedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniTrotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniTrotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-Posture-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniTrotPostureEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotPosturePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotPosturePPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-Refine-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmniTrotRefineEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotRefinePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotRefinePPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopPolishA1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopPolishA1EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopPolishA1PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopPolishPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopPolishA2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopPolishA2EnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopPolishA2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopPolishPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopStandFix-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStandFixEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStandFixPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopPolishPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopCrossPhysics-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopCrossPhysicsEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopCrossPhysicsPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopCrossPhysicsPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopRobustFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopRobustFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopRobustFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopFoundationPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-RobustStandFix-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotRobustStandFixEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotRobustStandFixPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotRobustStandFixPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Stand-ClosedLoop-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotStandExpertEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotStandExpertPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogStandExpertPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Stand-HeightCalibrated-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotStandHeightCalibratedEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotStandHeightCalibratedPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogStandHeightCalibratedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Stand-HeightHipCalibrated-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotStandHeightHipCalibratedEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotStandHeightHipCalibratedPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogStandHeightCalibratedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopSelectiveCollision-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopSelectiveCollisionEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopSelectiveCollisionPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopCrossPhysicsPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopStageB-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageBEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageBPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopStageC-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageCEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageCPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopStageD-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageDEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopStageDPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-ClosedLoopGaitRobust-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopGaitRobustEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotClosedLoopGaitRobustPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmniTrotClosedLoopExpansionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="CustomDog-Velocity-OmniTrot-DynamicsTeacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotDynamicsTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmniTrotDynamicsTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogDynamicsTeacherPPORunnerCfg"
        ),
    },
)


for stage, env_cfg, play_cfg in (
    ("T0", "RobotOmniTrotTerrainT0EnvCfg", "RobotOmniTrotTerrainT0PlayEnvCfg"),
    ("T1", "RobotOmniTrotTerrainT1EnvCfg", "RobotOmniTrotTerrainT1PlayEnvCfg"),
):
    gym.register(
        id=f"CustomDog-Velocity-OmniTrot-Terrain{stage}-v1",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{env_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{play_cfg}",
            "rsl_rl_cfg_entry_point": (
                "custom_dog_rl.agents.ppo_cfg:"
                "CustomDogDynamicsTeacherPPORunnerCfg"
            ),
        },
    )


gym.register(
    id="CustomDog-Velocity-Omni45-Polish-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V3PolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V3PolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45V3PolishPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-Conservative-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V3ConservativeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V3ConservativePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45V3PolishPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-Steering-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2SteeringEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2SteeringPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-PureAxis-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2PureAxisEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45V2PureAxisPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45NoSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-Usage-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsagePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45UsagePPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-UsageStage2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageStage2EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageStage2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45UsageStage2PPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-UsageBidirectional-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageBidirectionalEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageBidirectionalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45UsageBidirectionalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-UsageReverseDiscovery-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45UsageReverseDiscoveryEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45UsageReverseDiscoveryPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45UsageReverseDiscoveryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-SignedFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45SignedFoundationEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45SignedFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45SignedFoundationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-PriorityFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45PriorityFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45PriorityFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45SignedFoundationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-History213-SignedFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213SignedFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213SignedFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogHistory213SignedFoundationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-History213-SignedTransfer-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213SignedFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213SignedFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogHistory213TransferPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-History213-Distill-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213DistillationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213DistillationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogHistory213DistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-History213-OmniBidirectional-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213OmniBidirectionalEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotHistory213OmniBidirectionalPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogHistory213OmniBidirectionalPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-SignedPolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45SignedPolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45SignedPolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45SignedPolishPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-SignedVxPolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45SignedVxPolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45SignedVxPolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45SignedVxPolishPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-UsageStage2Turn-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageStage2TurnEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45UsageStage2TurnPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45UsageStage2PPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-VelocityFeedback-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47VelocityFeedbackEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47VelocityFeedbackPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47VelocityFeedbackPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-Foundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47FoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47FoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47FoundationPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-Axis-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47AxisEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47AxisPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47AxisPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-FoundationYaw-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47FoundationYawEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47FoundationYawPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47FoundationYawPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-Backward-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47BackwardEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47BackwardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47BackwardPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-SignedFoundation-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47FoundationPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-SignedTransfer-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47SignedTransferPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-SignedPolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedPolishEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedPolishPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47SignedPolishPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-SignedSymmetry-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni47SignedFoundationPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47SignedSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-ReverseOmniTeacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47ReverseOmniTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47ReverseOmniTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47ReverseOmniTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni47-ReverseDiscoveryTeacher-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47ReverseDiscoveryTeacherEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni47ReverseDiscoveryTeacherPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni47ReverseOmniTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-RoutedDistill-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotOmni45RoutedDistillationEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45RoutedDistillationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45RoutedDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-RoutedBalancePolish-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45RoutedBalancePolishEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45RoutedBalancePolishPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:CustomDogOmni45SignedPolishPPORunnerCfg"
        ),
    },
)

gym.register(
    id="CustomDog-Velocity-Omni45-PrivilegedRoutedDistill-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45PrivilegedRoutedDistillationEnvCfg"
        ),
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotOmni45PrivilegedRoutedDistillationPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "custom_dog_rl.agents.ppo_cfg:"
            "CustomDogOmni45PrivilegedRoutedDistillationRunnerCfg"
        ),
    },
)
