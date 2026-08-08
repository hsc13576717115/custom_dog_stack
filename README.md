# Custom Dog Stack

自制四足机器人的统一工程仓库，覆盖机器人描述、Isaac Lab 强化学习、
MuJoCo sim2sim、ROS 2 Humble 硬件控制和 Orin NX 策略部署。

## 当前状态

- URDF：17 links、16 joints、12 个可驱动关节。
- 电机：GO-M8010-6，hip/thigh 直接输出，calf 当前按额外 2:1 传动建模。
- 策略接口：45 维 observation，12 维关节位置 action，50 Hz。
- 训练方式：WSL CPU PhysX + RTX 4060 CUDA PPO。
- 已验证：64 个并行环境、1 轮 PPO、1536 timesteps、约 1199 steps/s。
- 尚未完成：RS485 实机驱动、编码器零点标定、急停链路和实机策略验收。

## 目录职责

```text
rl/                 自有 Isaac Lab 扩展和训练入口
ros2/src/           ROS 2 description、hardware、controller、bringup
deploy/             部署契约与审核后的策略发布包
docs/               架构、流程、安全与接口说明
scripts/            开发、训练、导出和验证命令
tests/              不依赖 Isaac Sim 的快速契约测试
third_party/         外部依赖版本锁定，不复制上游源码
```

## 首次初始化

```bash
cd /home/hsc/custom_dog_stack
./scripts/bootstrap.sh
./scripts/validate.sh
./scripts/check_system.sh
```

## 训练

接受 NVIDIA Omniverse EULA 后执行：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/train_smoke.sh
```

100 轮预训练检查：

```bash
CUSTOM_DOG_NUM_ENVS=128 CUSTOM_DOG_MAX_ITERATIONS=100 ./scripts/train.sh
```

正式训练：

```bash
./scripts/train.sh
```

训练日志位于 `logs/rsl_rl/custom_dog_velocity/`。查看曲线：

```bash
./scripts/tensorboard.sh
```

## 导出

```bash
./scripts/play_export.sh /absolute/path/to/model_N.pt
```

脚本会在 checkpoint 所在 run 的 `exported/` 下生成 `policy.onnx` 和
`policy.pt`。只有完成 sim2sim 和安全审查的 ONNX 才能进入
`deploy/releases/`。

## 开发原则

- `main` 分支始终保持可训练、可验证。
- 上游框架只作为锁定版本的外部依赖，不在本仓库复制维护。
- observation、action、关节顺序或缩放有任何变化，都必须提升部署契约版本。
- 每次正式训练必须保留 Git commit、seed、URDF 和导出的 `deploy.yaml`。
- 未完成编码器零点、方向、限位、急停和悬空测试前，不允许实机落地运行策略。

详细说明见 [架构文档](docs/architecture.md) 和
[开发流程](docs/development_workflow.md)，部署验收见
[部署流程](docs/deployment.md)。
