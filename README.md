# Custom Dog Stack

自制四足机器人的统一工程仓库，覆盖机器人描述、Isaac Lab 强化学习、MuJoCo
sim2sim、ROS 2 Humble 硬件控制和 Orin NX 策略部署。

本 README 是从训练到实机的执行手册。第一次使用时按顺序执行，不要直接跳过
sim2sim 和悬空测试把 ONNX 接到电机。

## 当前状态

- URDF：17 个 link、16 个 joint、12 个可驱动关节。
- 电机：GO-M8010-6；hip/thigh 直接输出，calf 当前按额外 2:1 传动建模。
- 策略接口：45 维基础 observation，或 47 维 phase/机身 `vx/vy` 反馈 observation；
  12 维关节位置 action、50 Hz。47 维扩展类型必须读取候选 `deploy.yaml`，不能只看维度。
- WSL 训练：CPU PhysX + RTX 4060 CUDA PPO。当前会话
  `torch.cuda.is_available()` 为 `True`；Isaac Sim 的 Vulkan/RTX 渲染设备仍不可用。
- 已验证：速度课程、45/12 策略契约、ONNX 导出、标准 MuJoCo 长时速度回放、
  Unitree MuJoCo + SDK2 bridge 接口闭环、ROS 2 Humble 四包编译。
- 当前前进高速候选：`deploy/candidates/model_4500_yaw_straight`；候选 YAML 显式包含低速
  command calibration 和高速目标校准，Python MuJoCo 已通过请求速度 `0~3 m/s` 的
  长时验收，但尚未进行真实机器人落地验收。
- 当前三轴 sim2sim 候选：`deploy/candidates/model_800_omni_stability_calibrated`；支持
  `vx/vy/wz` 和 47 维 phase observation，关节侧 `Kp=25/Kd=0.5`。训练 reward 不包含
  显式机身高度目标，策略可以在较快运动时自然压低机身；20 秒固定命令矩阵和 56 秒
  动态切换已通过 MuJoCo 验收。外部安全范围暂定为 `vx 0~0.6 m/s`、
  `vy +/-0.17 m/s`、`wz +/-0.6 rad/s`。纯侧移和原地偏航仍有约 `0.07~0.13 m/s`
  的非指令前向耦合，因此这是当前可视化和继续训练的基线，不是实机发布结论。
  `model_710_compact_height_polish` 和 `model_700_compact_omni_balanced` 保留为回退。
  所有候选均未进行真实机器人落地验收。
- 已拒绝候选：同一微调 run 的 `model_4695` 和后续长训 checkpoint；它们在高速
  sim2sim 中偏航恶化或翻倒，不能按“最后一个 checkpoint”直接部署。
- 已拒绝 47 维迁移：`omni47_velocity_feedback_from4840_100` 和
  `omni47_input_adapter_from4840_100` 均未降低高速转弯横向串扰，没有 checkpoint 进入
  `deploy/candidates/`；详见对应消融报告。
- 未完成：真实 GOM-8010-6 RS485 协议、编码器零点、方向标定、急停链路和落地验收。

Isaac Sim 在当前 WSL 中不能创建 Vulkan/RTX 设备，因此日志中可能出现
`No device could be created` 或 `CUDA libs are present, but no suitable CUDA GPU was found`。
只有 `torch.cuda.is_available()` 为 `True` 时才能让 PPO 使用 CUDA。Isaac Sim 的 Vulkan
报错和 PyTorch CUDA 是两个不同检查项；当前配置明确让 PhysX 使用 CPU、PPO 使用
`cuda:0`，因此 Vulkan 报错不会把策略网络训练自动降到 CPU。

## 目录职责

```text
rl/                 自有 Isaac Lab 任务、资产、PPO 和训练/导出入口
ros2/src/           description、hardware、controller、bringup
sim2sim/            自制狗 MuJoCo MJCF、凸包网格和 ONNX 控制器
deploy/             部署契约与审核后的策略发布包
docs/               架构、策略接口、流程、安全和部署说明
scripts/            初始化、验证、训练、导出、TensorBoard、ROS 2 构建
tests/              不依赖 Isaac Sim 的 URDF 契约测试
third_party/        外部依赖提交号，不复制上游源码
```

## 0. 环境准备

迁移到原生 Ubuntu 22.04 时，优先按照
[`docs/ubuntu2204_migration.md`](docs/ubuntu2204_migration.md) 执行。仓库已把
Isaac Lab、Unitree RL Lab 和 Unitree MuJoCo 固定为 Git submodule，并保存当前 WSL
使用的 Unitree 本地补丁。推荐克隆方式：

```bash
git clone --recurse-submodules git@github.com:hsc13576717115/custom_dog_stack.git
cd custom_dog_stack
git lfs pull
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/setup_ubuntu2204.sh --install-system-deps
```

新安装默认使用仓库内固定版本的 submodule：

```text
custom_dog_stack/third_party/IsaacLab
custom_dog_stack/third_party/unitree_rl_lab
custom_dog_stack/third_party/unitree_mujoco
custom_dog_stack/third_party/unitree_sdk2
```

仍可用已有的外部工作区，只需在执行脚本前覆盖环境变量：

```bash
export ISAACLAB_ROOT=/path/to/IsaacLab
export UNITREE_RL_LAB_ROOT=/path/to/unitree_rl_lab
export UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
```

初始化 Python 可编辑安装、检查依赖和 Git LFS：

```bash
cd /home/hsc/custom_dog_stack
./scripts/bootstrap.sh
./scripts/check_dependency_versions.sh
./scripts/check_system.sh
```

`check_system.sh` 应显示 RTX 4060，并且 `CUDA available: True`。

NVIDIA Omniverse EULA 已在本机持久化到 `/home/hsc/.bashrc`，仓库脚本也会自动设置：

```bash
echo "$OMNI_KIT_ACCEPT_EULA"
# YES
```

其他机器必须在接受 EULA 后单独配置该变量。

## 1. 修改 URDF 后先验证

训练、ROS 2 和后续 MuJoCo 都依赖同一份机器人描述：

[`ros2/src/custom_dog_description/urdf/custom_dog.urdf`](ros2/src/custom_dog_description/urdf/custom_dog.urdf)

每次修改 URDF、STL、质量、惯量、关节轴或限位后执行：

```bash
./scripts/validate.sh
```

验证内容包括：

```text
17 个 link、16 个 joint
关节树无环、关节类型正确
mesh 引用可以解析
质量和惯量为正
旋转关节具有轴、限位和合法范围
```

Isaac Sim 曾提示 thigh/calf 关节轴不是严格的 X/Y/Z 主轴。这个警告不一定阻止训练，
但说明 CAD 导出的轴带有微小偏转。进入 sim2sim 前必须确认这些偏转是真实机械轴，不能
让不同仿真器分别自动修正出不同结果。

## 2. 训练烟雾测试

先用 64 个环境跑 1 轮，只验证程序闭环：

```bash
./scripts/train_smoke.sh
```

等价的显式配置为：

```bash
CUSTOM_DOG_SIM_DEVICE=cpu \
CUSTOM_DOG_RL_DEVICE=cuda:0 \
CUSTOM_DOG_NUM_ENVS=64 \
CUSTOM_DOG_MAX_ITERATIONS=1 \
./scripts/train_smoke.sh
```

成功标准：

```text
程序退出码为 0
日志中显示 policy observation shape 为 45
日志中显示 action shape 为 12
生成 model_0.pt 和 events.out.tfevents...
```

当前烟雾测试的 checkpoint 只能证明训练链路可用，不能用于实机。

## 3. TensorBoard 和短训练

另开一个终端查看曲线：

```bash
cd /home/hsc/custom_dog_stack
./scripts/tensorboard.sh
```

浏览器访问：

```text
http://localhost:6006
```

在当前 RTX 4060 Laptop GPU（8 GiB）的 Ubuntu 22.04 环境中，`./scripts/train.sh`
默认同时使用 CUDA 物理仿真和 CUDA PPO，并启动 4096 个并行环境。该配置已经完成
多轮采样和 PPO 更新验证，实测约为 36,000--43,000 steps/s。6144 个环境虽然可以
创建场景，但在 8 GiB 显存上会在首次 PPO 更新时发生 CUDA OOM，因此不应作为默认值。

先进行 100 轮短训练：

```bash
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=100 \
./scripts/train.sh
```

每个 run 会创建在：

```text
logs/rsl_rl/custom_dog_velocity/<YYYY-MM-DD_HH-MM-SS>/
```

重点观察：

```text
Train/mean_reward
Train/mean_episode_length
Episode_Termination/bad_orientation
Metrics/base_velocity/error_vel_xy
Metrics/base_velocity/error_vel_yaw
Policy/mean_noise_std
```

100 轮内 reward 仍然可能为负，这不代表训练程序错误。此阶段主要检查环境不崩溃、
观测和动作维度正确、checkpoint 持续保存。

## 4. 正式长训练

当前推荐的显式对角 Trot 主线使用 49 维策略观测（45 维基础观测加 FR/FL/RR/RL
四腿时钟），从低速全向范围自动扩展到 `vx +/-3.0 m/s`、`vy +/-0.6 m/s`、
`wz +/-2.0 rad/s`：

```bash
./scripts/train_omni_trot.sh
```

该任务从随机初始化训练，不兼容 45/47 维旧 checkpoint。四腿顺序显式固定为
`FR, FL, RR, RL`，其中 `FR+RL` 和 `FL+RR` 为两组交替对角腿。

短训练不再出现 Python、URDF、CUDA/PyTorch 或环境创建错误后，再启动长训练：

```bash
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=5000 \
./scripts/train.sh
```

当前脚本默认值就是：

```text
simulation device: cpu
RL device: cuda:0
num envs: 128
max iterations: 5000
seed: 42
```

如果显存或系统内存不足，降低并行环境数量：

```bash
CUSTOM_DOG_NUM_ENVS=64 \
CUSTOM_DOG_MAX_ITERATIONS=5000 \
./scripts/train.sh
```

如果训练被中断，先保留已有 run 和 checkpoint。正式恢复训练时使用训练入口的
`--resume`、`--load_run` 和 `--checkpoint` 参数，不要把新 run 的配置文件覆盖到旧 run：

```bash
source scripts/activate_env.sh
python rl/scripts/train.py \
  --headless \
  --device cpu \
  --rl_device cuda:0 \
  --task CustomDog-Velocity-v0 \
  --num_envs 128 \
  --max_iterations 5000 \
  --seed 42 \
  --resume \
  --load_run 2026-08-08_15-42-17 \
  --checkpoint model_99.pt \
  --kit_args '--/renderer/enabled=pxr --/renderer/active=pxr --/renderer/multiGpu/enabled=false'
```

`model_N.pt` 就是 checkpoint。选择部署模型时，应根据 TensorBoard 和仿真表现选择，
不要机械地认为最后一个 checkpoint 一定最好。

当前用于解决 SDK2 接管姿态偏差的鲁棒微调命令为：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-Robust-v0 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=500 \
./scripts/train_smoke.sh \
  --run_name robust_finetune \
  --resume \
  --load_run 2026-08-08_15-57-41 \
  --checkpoint model_4999.pt
```

该任务把训练速度限制为 `vx ±0.3`、`vy ±0.15`、`yaw ±0.5`，并加入初始姿态和
关节偏差。结果见 [`docs/robust_finetune_report_2026-08-08.md`](docs/robust_finetune_report_2026-08-08.md)。

### 自然步态发现阶段

现有 `Robust-v0` 和 `model_5498_robust` 保留为稳定站立基线。不要在该配置上原样
追加长训练。自然步态实验使用独立任务 `CustomDog-Velocity-Gait-v1`：

- actor observation 仍为 45 维，不增加 gait phase；
- 不使用 `feet_gait` 显式相位奖励；
- 5% 环境训练零速站立，其余环境使用 `vx 0.2~0.5 m/s`、`vy ±0.05 m/s`、
  `yaw ±0.15 rad/s`；
- 暂时关闭周期推扰，收窄摩擦、质量和 reset 随机化；
- 增加存活奖励和非超时失败惩罚，降低姿态回归与 action-rate 惩罚；
- 步态涌现前暂时关闭会持续产生负值的 air-time 风格项，避免策略通过主动倾倒
  提前结束负回报 episode。

先做 100 轮健康检查：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-Gait-v1 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=100 \
./scripts/train.sh --run_name gait_v1_smoke
```

确认无 NaN、checkpoint 正常保存后，从随机初始化训练 2000 轮。这里不要使用
`--resume` 加载站立策略，因为站立策略和原 optimizer 可能强化当前局部最优：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-Gait-v1 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=2000 \
./scripts/train.sh --run_name gait_v1_natural
```

每 100 轮回放一次 checkpoint，固定测试 `vx=0.2` 和 `vx=0.5`。验收依据是实际
前进速度、线速度误差、四足接触周期和动作周期性，而不是只看 total reward。
若 1000~2000 轮后仍然只站立，再建立带弱 `feet_gait` 奖励的第二个对照任务，
不要直接修改本任务以免失去实验基线。

### 0~3 m/s 速度课程与 sim2sim 验收

自然步态阶段稳定后使用 `CustomDog-Velocity-Speed-v1`。它仍保持 actor observation
45 维、12 维位置动作；初始前向速度范围为 `0~0.75 m/s`，每个完整回合只有在速度
跟踪奖励达到门槛时才增加 `0.25 m/s`，上限为 `3.0 m/s`。该任务增加了小范围质量、
摩擦、PD 增益随机化，用于缩小 Isaac Sim、MuJoCo 和实机之间的差异。

先从 Gait-v1 的稳定 checkpoint 正确恢复（`train.sh` 会转发参数，`--kit_args` 放在
最末尾，不会吞掉 `--resume`）：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-Speed-v1 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=1500 \
./scripts/train.sh --run_name speed_v1_curriculum \
  --resume --load_run 2026-08-09_00-01-34 --checkpoint model_999.pt
```

需要继续巩固 `0~2 m/s` 后再扩速时，使用高速度续训任务：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-SpeedHigh-v1 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=1200 \
./scripts/train.sh --run_name speed_high_v1 \
  --resume --load_run <speed_v1_run> --checkpoint model_1499.pt
```

恢复测试必须在输出中看到 `Loading model checkpoint from:`、正确的 run 名和非 `1.0`
的 `Mean action noise std`。否则说明 checkpoint 没有加载，禁止继续长训。

导出并逐点验收：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-SpeedHigh-v1 CUSTOM_DOG_PLAY_STEPS=2 \
./scripts/play_export.sh logs/rsl_rl/custom_dog_velocity/<run>/model_XXXX.pt

for vx in 0 0.25 0.5 1.0 1.5 2.0 2.5 3.0; do
  ./scripts/run_sim2sim.sh \
    --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
    --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
    --command "$vx" 0 0 --duration 15 --warmup 3
done
```

`run_sim2sim.py` 会输出外部请求的 `command_vx`、实际机身坐标系 `vx/vy`、用于
observation 的 `policy_vx`、速度绝对误差、偏航角速度、最小机身高度、最大倾角以及
足端接触 duty/transitions。当前候选的仿真门槛是：不摔倒，`max_tilt <= 15 deg`、
`min_height >= 0.22 m`；`0.1~0.5 m/s` 使用 `abs_error <= max(0.05, 0.25*vx)`，
`0.5~3 m/s` 使用 `abs_error <= 0.15 m/s`。任一点不满足就保留 checkpoint 做诊断，
不部署到 Orin NX。

2026-08-09 的实验记录：早期 `speed_v1` 和 `speed_high_v1` 分别停在 `2.0 m/s`。
随后从正确恢复的 checkpoint 继续运行 `speed_full_v1_extend`，最终模型位于
`logs/rsl_rl/custom_dog_velocity/2026-08-09_01-49-22_speed_full_v1_extend`。MuJoCo
逐点测试 `0~3 m/s` 的速度误差为 `0.001~0.143 m/s`，最大倾角小于 `6 deg`，最小
机身高度 `0.236 m`；`3.0 m/s` 延长到 30 秒仍未摔倒。当前策略通过了“前向速度和
姿态”的初步 sim2sim 验收，但零偏航命令下平均偏航角速度约 `0.093 rad/s`。独立核对
MuJoCo gyro、四元数航向积分和关节映射后，确认这是策略/动力学偏置，不是传感器符号错误。

为此新增 `CustomDog-Velocity-SpeedStraight-v1`：保持 observation 为 45 维，把侧向和
偏航指令固定为零，并加入按指令计算的 yaw-rate L2 奖励。短微调 run
`2026-08-09_02-26-29_yaw_straight_finetune` 中，最早的 `model_4500` 通过了 sim2sim；
后续 `model_4695` 和长续训 checkpoint 反而偏航恶化或翻倒，因此不采用。

当前冻结候选位于 `deploy/candidates/model_4500_yaw_straight`。其 YAML 在
`0.1~0.6 m/s` 显式映射外部请求以跨过低速 gait dead zone，并在 `2.6~3.0 m/s`
线性加入左右 thigh `+/-0.02 rad` 的目标位置校准。请求速度 15 秒网格全部未摔倒，
误差不超过 `0.067 m/s`；0.1、0.25、0.5、2.5、2.75、3.0 m/s 的 60 秒测试也全部
未摔倒，误差不超过 `0.022 m/s`。完整数据和复现命令见
[`docs/speed_straight_report_2026-08-09.md`](docs/speed_straight_report_2026-08-09.md)。

这两项校准目前由本仓库 Python sim2sim 控制器读取。Unitree C++/ROS 2 action 后处理也
必须实现相同公式，并在真实机器人上重新标定；否则不能把该候选标记为硬件可用。

若要让 policy 本身逐步消除低速校准，可使用实验任务
`CustomDog-Velocity-SpeedBalancedTune-v1`。它保持 `0~3 m/s` 全范围样本，同时让约一半
非站立环境从 `0.1~0.5 m/s` 采样，并只在该段增加相对速度误差项。该任务使用固定
`1e-5` 学习率和每 5 轮 checkpoint；必须从稳定候选短微调并逐点 sim2sim，不能默认采用
最后一轮。当前实验 checkpoint 尚未整体优于已冻结候选，因此没有替换发布包。

### 开源全向训练与恢复参考

`unitree_rl_lab` 的四足任务类别主要是 velocity tracking，但它并非只有一个奖励：当前
自制狗基础任务已经包含速度/偏航跟踪、姿态、关节位姿、关节限位、动作变化、力矩、能耗、
足端滞空、足端滑动、非法接触和终止等项。不同开源工程最值得迁移的是任务结构，而不是把
大量 reward 名称原样相加：

- [`robot_lab`](https://github.com/fan-ziqi/robot_lab)：同为 Isaac Lab ManagerBased 环境，Go2 全向速度任务和 RSL-RL 左右镜像
  数据增强可直接参考；本仓库已据此加入 `CustomDog-Velocity-OmniSymmetry-v1`。
- [`basic-locomotion-isaaclab`](https://github.com/iit-DLSLab/basic-locomotion-isaaclab)：公开 Go2 `concurrent_symm` 的训练配置和测试权重，同时包含
  morphological symmetry、状态估计、MuJoCo sim2sim 和 ROS 2 sim2real。其 observation
  带并发状态估计器，必须先做契约映射，不能直接替换本仓库 45 维 ONNX。
- [`MuJoCo Playground`](https://github.com/google-deepmind/mujoco_playground)：`Go1Joystick` 是完整 `vx/vy/yaw` 任务；`Go1Getup` 是独立恢复
  任务。Getup 使用相对当前关节位置的动作目标以及朝向、高度、站姿门控奖励，适合用于
  趴姿恢复阶段，不应直接和早期全向步态一起训练。
- [`Walk These Ways`](https://github.com/Improbable-AI/walk-these-ways)：适合后续增加频率、占空比、腿相位、摆脚高度和站距等步态风格命令；
  它会改变 observation 和部署接口，不属于第一阶段闭环。
- [`rl_sar`](https://github.com/fan-ziqi/rl_sar)：可参考 ROS 2、ONNX 和 MuJoCo 部署框架，但其 Unitree 通信后端不能替代本项目
  自己的 USB-RS485 驱动、安全状态机和电机协议。

### Omni-45 v2 主线（当前推荐）

三套上游的奖励思想已选择性迁移到 `CustomDog-Velocity-Omni45-v2`：

- [`legged_gym`](https://github.com/leggedrobotics/legged_gym) 的线速度/偏航跟踪、姿态、能耗、
  关节速度/加速度、动作变化和碰撞等基础速度与稳定性项；
- [`HIMLoco`](https://github.com/InternRobotics/HIMLoco) 的二阶动作平滑；
- [`Walk These Ways`](https://github.com/Improbable-AI/walk-these-ways) 的接触足滑移、落地冲击和
  Raibert 足端落点思想。

这些项按 Isaac Lab Manager API 重新实现或复用 Isaac Lab 等价项，没有直接搬运旧 Isaac Gym
环境。没有迁移 gait phase、历史观测或 AMP，因此导出契约仍是 `45 -> 12`，可以沿用当前
ROS 2/Orin observation 代码。

这个任务从随机初始化开始，不要从 `model_4500` 续训。命令使用四个分桶：前进/后退、纯侧向、
纯偏航、三轴组合；每个分桶都对左右符号对称采样。初始范围为
`vx=[-0.5,0.5]`、`vy=[-0.15,0.15]`、`wz=[-0.4,0.4]`，只有线速度和偏航跟踪同时达到
`0.70` 才扩展，最终上限为 `vx=[-1,1]`、`vy=[-0.4,0.4]`、`wz=[-1,1]`。

奖励保留速度/偏航跟踪作为主项，并加入：

- legged_gym 风格的基础速度、偏航、姿态、能耗、关节与碰撞奖励；
- 一阶 `action_rate` 和 HIMLoco 风格的二阶 `action_smoothness_2`；
- Walk These Ways 风格的 `feet_slide`，以及首次接触的软落地和冲击速度；
- 从 MuJoCo home 姿态测得的四个足端 xyz 落点，加入速度和偏航预测后的软 Raibert 站立落点约束；
- RSL-RL 左右镜像数据增强。没有固定 hip=0 或固定机身高度目标，侧移和转向时 hip 可以自然变化。

先跑 1 轮 smoke：

```bash
cd /home/hsc/custom_dog_stack
OMNI_KIT_ACCEPT_EULA=YES \
CUSTOM_DOG_TASK=CustomDog-Velocity-Omni45-v2 \
CUSTOM_DOG_NUM_ENVS=32 \
CUSTOM_DOG_MAX_ITERATIONS=1 \
./scripts/train.sh --run_name omni45_v2_smoke
```

本机 WSL 的 Isaac Sim 仍会打印 Vulkan/RTX 不可用并使用 CPU PhysX；这不影响 PPO 使用
`cuda:0`。smoke 成功标准是日志显示 policy `(45,)`、action `12`、命令类型
`StratifiedOmniVelocityCommand`，并保存 `model_0.pt`。

通过 smoke 后启动正式主线（每 20 轮保存 checkpoint）：

```bash
cd /home/hsc/custom_dog_stack
OMNI_KIT_ACCEPT_EULA=YES \
CUSTOM_DOG_TASK=CustomDog-Velocity-Omni45-v2 \
CUSTOM_DOG_NUM_ENVS=4096 \
CUSTOM_DOG_MAX_ITERATIONS=5000 \
./scripts/train.sh --run_name omni45_v2_main
```

训练期间用 `./scripts/tensorboard.sh` 查看 `Metrics/base_velocity/error_vel_xy`、
`Metrics/base_velocity/error_vel_yaw`、`Episode_Termination/bad_orientation`、
`Episode_Reward/feet_slide`、`Episode_Reward/stance_foot_placement` 和
`Loss/symmetry`。每个 checkpoint 必须导出到独立目录，然后在 MuJoCo 逐点测试
`vx={-1,-0.5,0,0.5,1}`、`vy={-0.4,0,0.4}`、`wz={-1,0,1}`；验收记录速度误差、hip 外展、
动作一阶/二阶变化率、最低机身高度、最大倾角、足端滑移和是否摔倒。只有通过这组
sim2sim 网格的 checkpoint 才进入 ONNX/ROS 2 部署候选。

### Omni-45 v3 精修

`model_4999.pt` 已经在 v2 中覆盖完整 `[-1,1] / [-0.4,0.4] / [-1,1]` 命令范围，
但固定点测试表明低速前进、纯侧移和零指令轴漂移仍是局部最优。不要重新随机训练十万轮；
先使用独立的 `CustomDog-Velocity-Omni45-Polish-v3` 做低学习率续训。v3 保持 45→12
契约，固定使用完整命令范围，并提高纯侧移和 `0.15~0.40 m/s` 双向低速样本比例；组合样本
只使用常见的 `vx+wz`。奖励增加分轴相对误差、非指令轴漂移和触地前一步的 soft landing，
同时启用弱 mirror loss。它仍不限制 hip 角度，也不设置机身高度目标。

```bash
cd /home/hsc/custom_dog_stack
OMNI_KIT_ACCEPT_EULA=YES \
CUSTOM_DOG_TASK=CustomDog-Velocity-Omni45-Polish-v3 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=1000 \
CUSTOM_DOG_LOAD_OPTIMIZER=0 \
./scripts/train.sh --run_name omni45_v3_polish_from4999 \
  --resume \
  --load_run 2026-08-11_11-31-14_omni45_v2_vx1_omni_main \
  --checkpoint model_4999.pt
```

`CUSTOM_DOG_LOAD_OPTIMIZER=0` 是必要条件：它保留 actor/critic 权重，但用 v3 的
`5e-5` 固定学习率重新建立优化器。训练每 20 轮保存一次。新增的 signed error 与 signed
fraction 在 TensorBoard 中成对出现；某个方向的条件平均误差等于对应
`signed_error / signed_fraction`，这样正负方向的误差不会在总平均中互相掩盖。

如果 v3 的前 200~400 轮出现纯侧移过冲，不要继续使用该支线的最后 checkpoint。可从
`model_5200.pt` 启动保守支线：纯侧移训练带先收窄到 `vy=[-0.25,0.25]`，但
`deploy.yaml` 的外部上限仍保留 `[-0.4,0.4]`；同时降低分轴误差和足端风格项的权重，先恢复
稳定步态，再进行第二阶段的 `vy=[-0.4,0.4]` 扩展。

```bash
OMNI_KIT_ACCEPT_EULA=YES \
CUSTOM_DOG_TASK=CustomDog-Velocity-Omni45-Conservative-v3 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=500 \
CUSTOM_DOG_LOAD_OPTIMIZER=0 \
./scripts/train.sh --run_name omni45_v3_conservative_from5200 \
  --resume \
  --load_run 2026-08-11_13-31-38_omni45_v3_polish_from4999 \
  --checkpoint model_5200.pt
```

#### HIMLoco 与 AMP 的采用边界

当前主线是 Isaac Lab ManagerBased 环境加 RSL-RL PPO，未创建 AMP discriminator，
也未加载参考动作数据。因此当前 `amp_reward_coef` 等价于 `0`，不是一个遗漏的 reward
参数。HIMLoco 的核心是利用历史观测学习速度和环境隐变量估计，也不是 AMP。

现阶段不把 HIMLoco 或 AMP 训练器整体嫁接到主线：它会同时改变 observation、runner、
ONNX 导出和 Orin 部署契约，却不能自动修正 CAD/动力学左右不对称、横移串轴或后退局部
最优。推荐在平地三轴策略验收后建立两个独立版本：

- `Robust-v2` 参考 HIMLoco 的 history encoder/estimator，用于扰动和复杂地形；
- `Style-v2` 仅在获得与本机 12 关节拓扑一致、物理可执行的参考步态轨迹后加入 AMP。

最终手柄安全范围定义为 `vx=[-3,3] m/s`、`vy=[-0.4,0.4] m/s`、
`wz=[-1,1] rad/s`。范围是验收目标，不表示现有 checkpoint 已经覆盖它。2026-08-10
从前进 checkpoint 做的两次负向速度短微调中，稀疏负向采样得到的实际后退速度接近零；
强制 35% 负向样本又使 `bad_orientation` 超过 50%。这些模型均拒绝。完整双向策略应从
随机初始化的小速度双向课程开始，再逐级扩展到上述范围，不能通过修改 `deploy.yaml`
把未训练命令伪装成已支持。

镜像任务的训练入口保持 actor observation 45 维、action 12 维：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-OmniSymmetry-v1 \
CUSTOM_DOG_NUM_ENVS=128 \
CUSTOM_DOG_MAX_ITERATIONS=2000 \
./scripts/train.sh --run_name omni_symmetry_scratch
```

左右镜像同时变换机身角速度、重力投影、`vx/vy/yaw` 指令、policy/critic 关节量和 action。
第一阶段命令按纯前进、纯横移、纯偏航和组合四种模式采样，避免随机初始化策略一开始
同时面对三个不可辨识目标；待这四种模式分别通过 sim2sim 后再扩展倒退和 `3 m/s`。
数学 involution 测试与 1 iteration PPO smoke 已通过，PPO 日志中可见 `Loss/symmetry`。
从纯前进 `model_4500` 做的 4510/4520/4530 短迁移在 MuJoCo 中对 `vy=+/-0.25 m/s`、
`yaw=+/-0.5 rad/s` 仍基本保持四脚着地，实际横移/偏航接近零，因此这些 checkpoint 已
拒绝，不能部署。下一轮应从随机初始化训练全向任务，或先验证
`basic-locomotion-isaaclab/tested_policies/go2/concurrent_symm` 的教师契约后做迁移；不要再
从前进局部最优直接加大奖励微调。

## 5. 导出 ONNX

假设选择的 checkpoint 是：

```text
logs/rsl_rl/custom_dog_velocity/2026-08-08_15-42-17/model_99.pt
```

执行：

```bash
./scripts/play_export.sh \
  logs/rsl_rl/custom_dog_velocity/2026-08-08_15-42-17/model_99.pt
```

鲁棒任务的 checkpoint 必须用同一个任务配置导出：

```bash
CUSTOM_DOG_TASK=CustomDog-Velocity-Robust-v0 \
./scripts/play_export.sh \
  logs/rsl_rl/custom_dog_velocity/2026-08-08_21-31-25_robust_finetune/model_5498.pt
```

导出文件位于 checkpoint 同一个 run 的 `exported/` 目录：

```text
exported/policy.onnx
exported/policy.pt
```

同时保留：

```text
params/deploy.yaml
params/env.yaml
params/agent.yaml
events.out.tfevents...
训练时的 Git commit
```

ONNX 只代表策略网络，不包含 URDF、MJCF、RS485 协议、零点和安全逻辑。

## 6. 策略接口

当前策略契约见 [`docs/policy_contract.md`](docs/policy_contract.md)。基础策略使用 45 维
observation；带步态时钟的 phase 候选在末尾增加 `sin/cos`，使用 47 维：

```text
control period: 0.02 s
observation: 45 or 47 floats (由 deploy.yaml 声明)
action: 12 floats
```

Observation 顺序为：

```text
base angular velocity       3
projected gravity           3
velocity command            3
joint position - default  12
joint velocity             12
previous action            12
optional gait phase         2
```

Action 后处理为（先 clip，再加入可选 bias）：

```text
target_base = clip(default_joint_pos + 0.25 * policy_action, exported_action_clip)
target_position = target_base + speed_blend * optional_joint_target_bias
```

外部速度指令若候选 YAML 声明 `command_calibration`，分别对 `vx/vy/wz` 插值为 policy
command 写入 observation；`external_ranges` 是外部安全范围，`policy_ranges` 是校准后
允许写入网络的范围。日志和安全逻辑仍保留外部请求。当前策略是关节位置目标策略，不是直接
力矩策略。Kp/Kd 在低层执行器控制中生效。
当前候选固定使用关节侧 `Kp=25`、`Kd=0.5`，与 Isaac 训练执行器一致。
可选 `joint_target_bias` 只改变下发目标，45 维 observation 中的 `last_action` 仍然是
原始 ONNX action，不能回填处理后的目标位置。

`params/deploy.yaml` 中的 `joint_ids_map` 是动作和 SDK 电机编号的唯一权威映射，
不能通过名称排序自行推断。每次 observation、action、关节顺序或缩放变化，都必须
提升 policy contract 版本。

## 7. 建立自制狗 MuJoCo MJCF

完整说明见 [`sim2sim/custom_dog/README.md`](sim2sim/custom_dog/README.md)。第一次运行先建立
独立的 MuJoCo Conda 环境：

```bash
cd /home/hsc/custom_dog_stack
./scripts/setup_mujoco.sh
```

从主 URDF 生成并验证 MJCF：

```bash
./scripts/generate_mjcf.sh
```

该命令自动完成：

```text
URDF -> MJCF 和凸包网格
恢复 URDF 的完整质量和惯量张量
建立 floating_base 和 12 个位置执行器
建立 base IMU、四个足端接触传感器和 home keyframe
校验 nq=19、nv=18、nu=12、关节顺序、总质量和传感器契约
执行 1 秒无界面站立冒烟测试

同时生成官方 SDK2 bridge 使用的 `sim2sim/custom_dog/custom_dog_unitree.xml`。该版本
使用 12 个 torque motor，并将传感器按位置、速度、执行器力矩、IMU 的原始地址排列；
不依赖足端接触传感器。训练用的 `custom_dog.xml` 仍保留 position actuator 和仿真
接触传感器，两者用途不同。
```

每次修改 URDF、STL、惯量、关节轴、限位或传动参数后，都必须重新执行生成命令并提交
新的 `custom_dog.xml` 和转换网格。

无 ONNX 的 10 秒位置保持测试：

```bash
./scripts/run_sim2sim.sh --duration 10
```

打开 MuJoCo 窗口：

```bash
./scripts/run_sim2sim.sh --duration 30 --viewer
```

键盘交互查看当前冻结 ONNX 候选：

```bash
./scripts/teleop_mujoco_policy.sh
```

观看 `趴姿 -> 2 秒 FixStand -> 1 秒 PolicyHold -> Velocity` 全流程：

```bash
CUSTOM_DOG_INITIAL_STATE=prone ./scripts/teleop_mujoco_policy.sh
```

窗口获得焦点后，`1`/`P`/空格为 Passive，`2`/`R` 为 FixStand，`3`/`V` 为 Velocity；
`W`/`S` 改前进速度，`A`/`D` 改侧向速度，`Q`/`E` 改偏航速度，`X` 清零。当前默认候选
训练了 `vx/vy/yaw` 三轴命令。FixStand 使用默认 2 秒五次曲线；切换 Velocity 后先保持
默认 1 秒零速度，再释放键盘命令。该入口只控制 MuJoCo，不会发送实机命令。

当前 10 秒位置保持测试数值稳定，但机体会有轻微滑移；在把 sim2sim 结果作为
sim2real 依据前，还要继续优化足端简化碰撞体、摩擦以及 CAD 导出的关节偏轴。

## 8. sim2sim 控制器

使用同一个训练 run 的 `policy.onnx` 和 `params/deploy.yaml`：

```bash
./scripts/run_sim2sim.sh \
  --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
  --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
  --command 0.3 0.0 0.0 \
  --duration 30 \
  --viewer
```

MuJoCo 控制器每 0.02 秒执行一次：

```text
读取 base angular velocity
读取 projected gravity
读取速度指令
读取 12 个关节位置和速度
按 deploy.yaml 拼成 45 或 47 维 observation
phase 候选追加 sin/cos 步态时钟
调用 policy.onnx
得到 12 维 action
转换成目标关节位置并应用候选 YAML 的可选校准
```

必须使用和训练端完全一致的：

```text
observation 顺序
observation scale
default joint position
action scale
joint_ids_map
关节方向
关节限位
控制周期
```

sim2sim 至少验收：

```text
站立 30 秒
前进、后退、左右横移
原地转向
随机外力扰动恢复
关节限位不越界
policy 输入异常时停止
```

sim2sim 失败时先修模型和接口，不要直接修改实机参数来掩盖问题。

`/home/hsc/unitree_mujoco` 是宇树官方 MuJoCo + Unitree SDK2 桥接示例，本项目已经为
自制模型准备了相同的 SDK2 数据格式。官方闭环命令为：

```bash
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_5498_robust
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_4500_yaw_straight
```

执行前需按 [`docs/unitree_mujoco_startup_patch.md`](docs/unitree_mujoco_startup_patch.md)
在两个上游工作区应用启动时序补丁并重新构建。该命令验证的是 MJCF、DDS、SDK2
`LowState/LowCmd` 和 ONNX controller 的接口闭环；默认零速度指令。当前上游 C++
controller 尚未读取 `model_4500_yaw_straight` 的 command/target calibration，因此该命令不能
替代 Python MuJoCo 的 0~3 m/s 验收，也不能直接证明 Orin 部署完成。

## 9. 查看可视化效果

TensorBoard 曲线：

```bash
./scripts/tensorboard.sh
# 浏览器打开 http://localhost:6006
```

Isaac Lab 同域 playback（需要 Isaac Sim 能创建 Vulkan/RTX 图形设备）：

```bash
./scripts/view_isaac_policy.sh \
  logs/rsl_rl/custom_dog_velocity/2026-08-08_15-57-41/model_4999.pt
```

标准 MuJoCo policy playback（WSL 当前可用）：

```bash
./scripts/view_mujoco_policy.sh deploy/candidates/model_4500_yaw_straight 1.0 0.0 0.0
./scripts/view_mujoco_policy.sh deploy/candidates/model_4500_yaw_straight 3.0 0.0 0.0
```

官方 Unitree MuJoCo + SDK2 bridge playback：

```bash
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_4500_yaw_straight
```

历史基线 `model_5498_robust` 在标准 MuJoCo 和官方 SDK2 bridge 中均能零速站立
30 秒，两个控制器的 observation/action trace 逐项误差小于 `1e-5`。标准 MuJoCo
`0.2 m/s` 指令下 30 秒只移动约 `0.009 m`，因此它是稳定站立候选，不是正常行走策略。
当前行走候选的标准 MuJoCo 结果见
[`docs/speed_straight_report_2026-08-09.md`](docs/speed_straight_report_2026-08-09.md)。

当前 WSL 的 Isaac GUI 实测失败于 `No device could be created`、`Graphics plugins not
available` 和 `nvidia-smi not found in /usr/bin`。PyTorch CUDA 和 CPU PhysX 训练不受
这个图形问题影响；要看 Isaac 画面，请在原生 Ubuntu 22.04 或 Windows 11 的 Isaac Sim
工作站环境运行同一个 `view_isaac_policy.sh`，或修复 WSL 的 Vulkan 驱动后重新测试。NVIDIA
当前安装文档列出的工作站系统是 Ubuntu 22.04/24.04 或 Windows 11，并要求兼容的 RTX
GPU/驱动：[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。

## 10. ROS 2 Humble 构建

当前仓库已经提供四个 ROS 2 包：

```text
custom_dog_description
custom_dog_hardware
custom_dog_controller
custom_dog_bringup
```

编译：

```bash
cd /home/hsc/custom_dog_stack
./scripts/build_ros2.sh
```

查看 URDF：

```bash
source /opt/ros/humble/setup.bash
source install/ros2/setup.bash
ros2 launch custom_dog_description display.launch.py
```

`custom_dog_controller` 已提供与仿真一致、经过 C++ 单元测试的
`Passive -> FixStand -> PolicyHold -> Velocity` 状态机核心；它从实测关节角开始 2 秒
五次插值，并在放行 `vx/vy/wz` 前让 policy 零速接管 1 秒。`custom_dog_hardware` 仍是接口
骨架，尚未完成真实 GOM-8010-6 协议。`config/hardware.yaml` 中的端口、ID、方向和零点
都是待标定值，`validated: false` 时禁止启动实机策略。

`qr_ws` 驱动已经确认其上层 `Kp/Kd` 是关节侧参数，写入 GOM-8010-6 前按传动比
平方换算。当前 RL 的固定值为：

```text
joint side: Kp=25, Kd=0.5
hip/thigh motor side (6.33): Kp≈0.624, Kd≈0.0125
calf motor side (12.66):    Kp≈0.156, Kd≈0.00312
```

这些电机侧数值必须由驱动计算，不应在策略层硬编码。`qr_ws` 小跑中更高且随相位变化
的增益还配合 VMC 前馈力矩，不能直接替换本 RL 策略的固定 PD。

## 11. Orin NX Super 部署

目标设备需要准备：

```text
Ubuntu 22.04
ROS 2 Humble
USB 四路 RS485 适配器
ONNX Runtime 或 TensorRT
实时串口调度和物理急停
```

部署包至少包含：

```text
policy.onnx
deploy.yaml
metadata.yaml
sha256sums.txt
```

Orin NX 上的控制链路和状态机：

```text
Prone/Passive
-> 从实测关节角到 HOME 的 2 秒五次曲线 FixStand
-> policy 接管，保持 1 秒零速度
-> Velocity 接收 vx/vy/wz
-> RS485 读取 12 个电机状态和 IMU
-> 按 deploy.yaml 构造 45 或 47 维 observation
-> ONNX/TensorRT 推理
-> default_position + action_scale * action
-> 关节限位、速度限制、温度检查
-> 四路 RS485 发送目标位置、Kp、Kd
```

RL policy 只负责站稳后的速度控制，不使用趴姿 observation，也不输出起身动作。47 维
phase policy 的最后两个输入仅在 Velocity 状态中提供步态周期，不改变上述恢复状态机。

实机开放顺序固定为：

```text
单电机无负载
-> 单腿悬空
-> 四腿悬空
-> 低 Kp/Kd 固定站立
-> 安全绳落地
-> 小速度策略
-> 正常速度策略
```

通信超时、CRC 错误、过温、越限、急停或 observation 异常时必须进入零输出或阻尼状态。

## 12. 香橙派 5 Plus

香橙派 5 Plus 可以作为策略推理主机，但不能替代 RTX 4060 训练机，也不能运行 Isaac Sim。

建议先使用：

```text
ARM64 ONNX Runtime + CPU 推理
```

如果 50 Hz 延迟或抖动不满足要求，再考虑将 ONNX 转换为 RKNN 使用 RK3588 NPU。
香橙派必须重新测量 ONNX 延迟、RS485 调度抖动、内存和温度，不能直接假设与 Orin NX
性能相同。实机首版建议先在 Orin NX 完成闭环，再迁移到香橙派做对比。

## 13. 故障排查

WSL 中看到：

```text
No device could be created
CUDA libs are present, but no suitable CUDA GPU was found
```

先检查：

```bash
./scripts/check_system.sh
```

如果显示：

```text
CUDA available: True
CUDA device: RTX 4060
```

并且训练目录生成了 `model_N.pt`，则说明 PPO CUDA 正常，Isaac Sim 只是使用 CPU
PhysX。不要把 Isaac Sim 的 Vulkan 图形错误误判为 PyTorch CUDA 训练失败。

如果显存或系统内存不足：

```bash
CUSTOM_DOG_NUM_ENVS=64 \
CUSTOM_DOG_MAX_ITERATIONS=5000 \
./scripts/train.sh
```

## 开发规则

- `main` 分支始终保持可训练、可验证。
- 上游框架只作为锁定版本的外部依赖，不在本仓库复制维护。
- 修改 URDF 后必须重新运行 `./scripts/validate.sh` 和训练烟雾测试。
- 每次正式训练保留 Git commit、seed、URDF、`params/*`、checkpoint 和 TensorBoard 日志。
- 未完成编码器零点、方向、限位、急停和悬空测试前，不允许实机落地运行策略。
- 原始 checkpoint、训练日志、ROS 构建目录和 Isaac 缓存不提交到 Git；正式模型发布到
  `deploy/releases/` 前必须经过 sim2sim 和安全审查。

进一步说明：

- [架构文档](docs/architecture.md)
- [开发流程](docs/development_workflow.md)
- [部署流程](docs/deployment.md)
- [策略接口契约](docs/policy_contract.md)
- [鲁棒微调与 sim2sim 报告](docs/robust_finetune_report_2026-08-08.md)
- [0~3 m/s 速度与偏航 sim2sim 报告](docs/speed_straight_report_2026-08-09.md)
- [Omni45 Usage-v1 第一阶段训练报告](reports/omni45_usage_v1_training_report.md)
- [Omni45 Usage-v1 冻结候选](deploy/candidates/omni45_usage_v1_model_4840/README.md)
- [Omni45 Usage Stage-2 试验报告](reports/omni45_usage_stage2_report.md)
- [Omni47 线速度反馈消融报告](reports/omni47_velocity_feedback_ablation.md)
