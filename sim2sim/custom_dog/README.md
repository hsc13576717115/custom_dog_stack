# Custom Dog MuJoCo

这里保存由主 URDF 生成的 MJCF、凸包网格、传感器定义和 ONNX sim2sim 控制器。
URDF 仍然是机器人几何、质量、惯量、关节轴和限位的唯一源文件。

## 生成并验证 MJCF

```bash
cd /home/hsc/custom_dog_stack
./scripts/setup_mujoco.sh
./scripts/generate_mjcf.sh
```

生成脚本会转换 URDF、恢复完整惯量张量、加入浮动基座、12 个位置执行器、IMU、
四个足端接触传感器和 `home` keyframe，然后执行 1 秒无界面站立测试。

同一条命令还会生成 `custom_dog_unitree.xml`。这是给宇树官方
`/home/hsc/unitree_mujoco` 使用的 bridge 版本：12 个执行器是 SDK2 顺序的 torque
motor，传感器地址按 12 个位置、12 个速度、12 个执行器力矩、IMU 的顺序排列，
因此 bridge 不需要足端接触传感器。训练 MJCF 和官方 bridge MJCF 要分开保存，不能
把 position actuator 版本直接当作 SDK2 torque motor 版本。

验证官方格式：

```bash
cd /home/hsc/custom_dog_stack
./scripts/generate_mjcf.sh
```

成功时会检查 `nq=19`、`nv=18`、`nu=12`、执行器顺序、原始 sensor address 和
`nsensordata=52`，并运行 SDK2 等价的 10 秒 PD 冒烟测试。

## 仅验证模型站立

```bash
./scripts/run_sim2sim.sh --duration 10
```

带 MuJoCo 窗口运行：

```bash
./scripts/run_sim2sim.sh --duration 30 --viewer
```

## 键盘交互查看

加载当前冻结候选并一直运行到关闭窗口：

```bash
./scripts/teleop_mujoco_policy.sh
```

从实测趴姿启动并观看完整状态机：

```bash
CUSTOM_DOG_INITIAL_STATE=prone ./scripts/teleop_mujoco_policy.sh
```

窗口获得焦点后：`P` 进入 Passive（零力矩）；`R` 从当前关节角按默认 2 秒五次曲线
插值到 home，随后自动交给零速度 ONNX PolicyHold；确认站稳后按 `V` 释放键盘速度指令。
`W`/`S` 以每次
`0.1 m/s` 调整前进速度，`A`/`D` 调整侧向速度，`Q`/`E` 调整偏航角速度，`X` 清零。
默认加载 `model_800_omni_stability_calibrated`，其已验证外部范围为
`vx=0~0.6 m/s`、`vy=+/-0.17 m/s`、`yaw=+/-0.6 rad/s`。

窗口默认使用 MuJoCo tracking camera，持续以 `base` 为中心跟随机器人平移；鼠标仍可
调整距离、方位角和俯仰角。需要完全自由视角时运行：

```bash
CUSTOM_DOG_CAMERA_MODE=free ./scripts/teleop_mujoco_policy.sh
```

不要用 `Space` 或数字键切换机器人状态：MuJoCo 自己将 `Space` 用作播放/暂停，将
`0..5` 用作几何显示分组。终端出现 `interactive mode=policy_hold` 才表示起身插值已经
完成并由零速度策略接管；出现 `interactive mode=velocity` 后速度按键才会真正生效。
进入 `P` 或 `R` 会自动清零旧的三轴命令，防止再次进入 Velocity 时突然起步。
在 Passive、FixStand 或 PolicyHold 中按速度键会被忽略。

这是 Python MuJoCo 的仿真调试入口，不会通过 ROS 2 或 RS485 下发任何实机指令。

恢复策略验收时，使用训练中相同的折叠初态：hip=`0 deg`、thigh=`71 deg`、
calf=`-161 deg`。

```bash
./scripts/run_sim2sim.sh \
  --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
  --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
  --command 0.0 0.0 0.0 \
  --initial-state prone --recovery-ramp 2.0 --recovery-hold 1.0 \
  --duration 15 --warmup 3 --viewer
```

控制器先用五次平滑曲线在 `--recovery-ramp 2.0` 秒内从实测趴姿移动到当前候选
`deploy.yaml` 的 `default_joint_pos`（映射为 SDK 电机顺序），
随后让 policy 接管并保持零速度 `--recovery-hold 1.0` 秒，最后释放 `--command` 指定的
`vx/vy/yaw`。这样可以避免从趴姿直接交给 ONNX 时出现大幅关节目标跳变。

这是固定的职责边界：趴下到站立完全属于部署状态机，RL policy 不负责起身。phase
候选额外使用的 `sin/cos` 只是在 Velocity 状态中提供步态时钟，不参与五次曲线站立。

输出的 `recovery` 行要求至少出现一次 `height >= 0.25 m` 且 `tilt <= 15 deg`，并会
报告过渡期间的最大高度、倾角、实际执行器力和关节速度。这只是仿真验收门槛；实机状态机
必须实现同一条平滑曲线，并从悬空、低增益、急停可用的条件开始测试。

## 加载 ONNX 策略

`policy.onnx` 和 `deploy.yaml` 必须来自同一个训练 run 或同一个冻结候选：

```bash
./scripts/run_sim2sim.sh \
  --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
  --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
  --command 0.3 0.0 0.0 \
  --duration 30 \
  --viewer
```

控制器严格使用 `deploy.yaml` 中的 observation 顺序、缩放、动作 offset/scale、
`joint_ids_map`、每关节 Kp/Kd 和 `step_dt`。runner 会把 YAML 增益写入 MuJoCo
position actuator，并用同一组增益计算 torque-speed 限幅。当前模型的 MuJoCo 步长为 0.005 秒，策略每四个仿真步
执行一次，即 50 Hz。

Runner 同时支持 45 维基础 observation，以及在末尾追加 `gait_phase` 或
`base_lin_vel_xy` 的两种 47 维契约。具体扩展必须由 YAML 中的 term 名称区分。
`base_lin_vel_xy` 在 MuJoCo 中由浮动基座世界速度旋转到机身坐标系得到。

若 YAML 声明 `command_calibration`，`--command` 是外部 requested command，控制器会
分别插值 `lin_vel_x`、`lin_vel_y` 和 `ang_vel_z`，再写入 observation。输出指标仍与
requested command 比较，并同时打印三轴 `policy_command`。`external_ranges` 限制外部请求，
`policy_ranges` 限制校准后网络输入；移植到 C++/ROS 2 时必须实现相同逻辑。

使用一条 MuJoCo 进程验证动态指令切换：

```bash
./scripts/run_sim2sim.sh \
  --policy deploy/candidates/model_800_omni_stability_calibrated/exported/policy.onnx \
  --deploy-yaml deploy/candidates/model_800_omni_stability_calibrated/params/deploy.yaml \
  --duration 32 --warmup 0.5 \
  --command-step 0 0 0 0 \
  --command-step 5 0.3 0 0 \
  --command-step 12 0 0.15 0 \
  --command-step 19 0 0 0.4 \
  --command-step 25 0.5 0.15 0.4
```

每个 `--command-step` 是 `TIME VX VY YAW`，首项必须从 0 秒开始。输出会分别报告各段
requested command、policy command、实测速度和误差。

当前速度候选的复现命令：

```bash
./scripts/run_sim2sim.sh \
  --policy deploy/candidates/model_4500_yaw_straight/exported/policy.onnx \
  --deploy-yaml deploy/candidates/model_4500_yaw_straight/params/deploy.yaml \
  --command 3.0 0.0 0.0 --duration 60 --warmup 10
```

候选 YAML 的可选 `joint_target_bias` 在 action clip 后、SDK 关节映射前加入目标关节位置；
下一帧 `last_action` 仍使用原始 ONNX action。完整公式和验收结果见
[`../../docs/speed_straight_report_2026-08-09.md`](../../docs/speed_straight_report_2026-08-09.md)。

当前仓库的 Python runner 是自制 RS485 控制栈的主 sim2sim 路径。
官方 Unitree MuJoCo 可以直接作为本项目的 SDK2 sim2sim 端：

```bash
cd /home/hsc/custom_dog_stack
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_4500_yaw_straight
```

该脚本启动官方 `unitree_mujoco` 和官方 `go2_ctrl`，通过本机 DDS `domain=1`、网卡
`lo` 闭环。控制器会自动进入 `Passive -> FixStand -> Velocity`；默认没有手柄速度
指令，因此只用于启动、接口和安全状态验证。真实运动指令需要在 `lo` 对应的 SDK2
控制器中接入 joystick/command 源。

当前上游 C++ action 后处理不会读取 `command_calibration` 或 `joint_target_bias`。在同步实现这些字段并重建前，
官方 bridge 命令不等价于本仓库 Python runner 的高速校准结果，也不能作为 Orin NX
行走验收。

官方二进制的启动补丁位于上游工作区 `/home/hsc/unitree_mujoco`，包括 home keyframe
初始化、bridge 就绪前暂停物理步进和初始 home PD。`/home/hsc/unitree_rl_lab` 的
`State_FixStand` 还需从实测 `lowstate` 开始插值，`JointAction::reset()` 还需先输出
策略 offset。上游仓库不是本项目的子模块，换机
时应按 `docs/unitree_mujoco_startup_patch.md` 应用并重新构建；不要把训练用
`custom_dog.xml` 直接传给官方 binary。

官方 bridge 只读取关节位置、速度、执行器力矩和 IMU，不要求足端接触传感器。没有足端
接触传感器不会阻止接口闭环，但会减少可用于奖励、接触状态估计和安全判断的信息；训练端
若使用接触相关奖励，必须改成仿真接触或力矩/运动学估计，并在真机上重新验证。
