# Custom Dog Stack

自制四足机器人的统一工程仓库，覆盖机器人描述、Isaac Lab 强化学习、MuJoCo
sim2sim、ROS 2 Humble 硬件控制和 Orin NX 策略部署。

本 README 是从训练到实机的执行手册。第一次使用时按顺序执行，不要直接跳过
sim2sim 和悬空测试把 ONNX 接到电机。

## 当前状态

- URDF：17 个 link、16 个 joint、12 个可驱动关节。
- 电机：GO-M8010-6；hip/thigh 直接输出，calf 当前按额外 2:1 传动建模。
- 策略接口：45 维 observation、12 维关节位置 action、50 Hz。
- WSL 训练：CPU PhysX + RTX 4060 CUDA PPO。
- 已验证：5000 轮基础训练、500 轮鲁棒微调、ONNX 导出、标准 MuJoCo 30 秒站立、
  Unitree MuJoCo + SDK2 bridge 30 秒站立、ROS 2 Humble 四包编译。
- 当前候选：`deploy/candidates/model_5498_robust`；已通过站立，尚未学会有效速度跟踪。
- 未完成：真实 GOM-8010-6 RS485 协议、编码器零点、方向标定、急停链路和落地验收。

Isaac Sim 在当前 WSL 中不能创建 Vulkan/RTX 设备，因此日志中可能出现
`No device could be created` 或 `CUDA libs are present, but no suitable CUDA GPU was found`。
只要 PyTorch CUDA 可用、训练脚本返回成功并保存 checkpoint，当前配置仍然可以训练：
Isaac Sim/PhysX 使用 CPU，PPO 网络使用 CUDA。

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

当前 WSL 工作区假设以下目录存在：

```text
/home/hsc/IsaacLab
/home/hsc/unitree_rl_lab
/home/hsc/unitree_mujoco
/home/hsc/custom_dog_stack
```

如果目录不同，可以在执行脚本前覆盖环境变量：

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

`run_sim2sim.py` 会输出机身坐标系的平均 `vx/vy`、速度绝对误差、偏航角速度、最小
机身高度、最大倾角以及足端接触 duty/transitions。当前建议的实机候选门槛是：每个
速度点不摔倒，`abs_error <= 0.3 m/s`，`max_tilt <= 15 deg`，`min_height >= 0.22 m`；
任一点不满足就保留 checkpoint 做诊断，不部署到 Orin NX。

2026-08-09 的实验记录：早期 `speed_v1` 和 `speed_high_v1` 分别停在 `2.0 m/s`。
随后从正确恢复的 checkpoint 继续运行 `speed_full_v1_extend`，最终模型位于
`logs/rsl_rl/custom_dog_velocity/2026-08-09_01-49-22_speed_full_v1_extend`。MuJoCo
逐点测试 `0~3 m/s` 的速度误差为 `0.001~0.143 m/s`，最大倾角小于 `6 deg`，最小
机身高度 `0.236 m`；`3.0 m/s` 延长到 30 秒仍未摔倒。当前策略通过了“前向速度和
姿态”的 sim2sim 验收，但零偏航命令下平均偏航角速度约 `0.093 rad/s`，世界坐标
轨迹会转弯；这需要在实机低速测试中继续校准 yaw/IMU 符号，不能直接按最高速部署。

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

当前策略契约见 [`docs/policy_contract.md`](docs/policy_contract.md)：

```text
control period: 0.02 s
observation: 45 floats
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
```

Action 后处理为：

```text
target_position = default_joint_pos + 0.25 * policy_action
```

当前策略是关节位置目标策略，不是直接力矩策略。Kp/Kd 在低层执行器控制中生效。
当前候选固定使用关节侧 `Kp=25`、`Kd=0.5`，与 Isaac 训练执行器一致。

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
拼成 45 维 observation
调用 policy.onnx
得到 12 维 action
转换成目标关节位置
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
```

执行前需按 [`docs/unitree_mujoco_startup_patch.md`](docs/unitree_mujoco_startup_patch.md)
在两个上游工作区应用启动时序补丁并重新构建。该命令验证的是 MJCF、DDS、SDK2
`LowState/LowCmd` 和 ONNX controller 的接口闭环；默认零速度指令，不代表当前策略已经
学会稳定行走。策略效果以 [`docs/training_report_2026-08-08_15-57-41.md`](docs/training_report_2026-08-08_15-57-41.md)
中的 5000 轮分析和后续 sim2sim 结果为准。

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
./scripts/view_mujoco_policy.sh deploy/candidates/model_5498_robust 0.0 0.0 0.0
```

官方 Unitree MuJoCo + SDK2 bridge playback：

```bash
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_5498_robust
```

本机实测 `model_5498_robust` 在标准 MuJoCo 和官方 SDK2 bridge 中均能零速站立
30 秒，两个控制器的 observation/action trace 逐项误差小于 `1e-5`。标准 MuJoCo
`0.2 m/s` 指令下 30 秒只移动约 `0.009 m`，因此它是稳定站立候选，不是正常行走策略。

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

当前 `custom_dog_hardware` 和 `custom_dog_controller` 是接口骨架，尚未完成真实
GOM-8010-6 协议。`config/hardware.yaml` 中的端口、ID、方向和零点都是待标定值，
`validated: false` 时禁止启动实机策略。

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

Orin NX 上的控制链路：

```text
RS485 读取 12 个电机状态和 IMU
-> 构造 45 维 observation
-> ONNX/TensorRT 推理
-> default_position + action_scale * action
-> 关节限位、速度限制、温度检查
-> 四路 RS485 发送目标位置、Kp、Kd
```

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
