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

## 加载 ONNX 策略

`policy.onnx` 和 `deploy.yaml` 必须来自同一个训练 run：

```bash
./scripts/run_sim2sim.sh \
  --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
  --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
  --command 0.3 0.0 0.0 \
  --duration 30 \
  --viewer
```

控制器严格使用 `deploy.yaml` 中的 observation 顺序、缩放、动作 offset/scale、
`joint_ids_map` 和 `step_dt`。当前模型的 MuJoCo 步长为 0.005 秒，策略每四个仿真步
执行一次，即 50 Hz。

当前仓库的 Python runner 是自制 RS485 控制栈的主 sim2sim 路径。
官方 Unitree MuJoCo 可以直接作为本项目的 SDK2 sim2sim 端：

```bash
cd /home/hsc/custom_dog_stack
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_4999
```

该脚本启动官方 `unitree_mujoco` 和官方 `go2_ctrl`，通过本机 DDS `domain=1`、网卡
`lo` 闭环。控制器会自动进入 `Passive -> FixStand -> Velocity`；默认没有手柄速度
指令，因此只用于启动、接口和安全状态验证。真实运动指令需要在 `lo` 对应的 SDK2
控制器中接入 joystick/command 源。

官方二进制的启动补丁位于上游工作区 `/home/hsc/unitree_mujoco`，包括 home keyframe
初始化、bridge 就绪前暂停物理步进和初始 home PD。`/home/hsc/unitree_rl_lab` 的
`State_FixStand` 还需从实测 `lowstate` 开始插值，`JointAction::reset()` 还需先输出
策略 offset。上游仓库不是本项目的子模块，换机
时应按 `docs/unitree_mujoco_startup_patch.md` 应用并重新构建；不要把训练用
`custom_dog.xml` 直接传给官方 binary。

官方 bridge 只读取关节位置、速度、执行器力矩和 IMU，不要求足端接触传感器。没有足端
接触传感器不会阻止接口闭环，但会减少可用于奖励、接触状态估计和安全判断的信息；训练端
若使用接触相关奖励，必须改成仿真接触或力矩/运动学估计，并在真机上重新验证。
