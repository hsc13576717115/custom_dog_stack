# 官方 Unitree MuJoCo 启动补丁

本项目使用宇树官方 `/home/hsc/unitree_mujoco` 的 MuJoCo viewer 和 SDK2 bridge，
但自制模型是 torque motor。官方示例默认模型和启动时序，不能保证自制模型在 DDS
连接前保持 home 姿态，因此需要以下本地修改。

## 应用位置

在 `/home/hsc/unitree_mujoco` 的 `simulate/src/main.cc`：

- 先在局部 `mjData* dnew` 上执行 `sim->Load`、`mj_resetDataKeyframe(home)`、
  `mj_forward`，最后才发布全局 `d`，避免 bridge 读到零初始化状态。
- 启动时设置 `sim->run = 0`，创建 bridge 并启动其 1 kHz 线程后再恢复 `sim->run = 1`。

在 `simulate/src/unitree_sdk2_bridge.h` 的 `RobotBridge` 构造函数中：

- 若存在 `home` keyframe，把初始关节位置写入 `LowCmd`，并设置 `kp=25`、`kd=0.5`，
  直到外部控制器发布第一条命令。

在 `/home/hsc/unitree_rl_lab/deploy/include/FSM/State_FixStand.h`：

- `FixStand::enter()` 的插值起点必须读取 `lowstate->msg_.motor_state()[i].q()`，
  不能读取刚建立 DDS 时尚未初始化的 `lowcmd`。

## 重新构建

```bash
cmake --build /home/hsc/unitree_mujoco/simulate/build -j4
cmake --build /home/hsc/unitree_rl_lab/deploy/robots/go2/build -j4
```

然后回到本仓库执行：

```bash
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_4999
```

这些修改只解决启动时序和数据初始化，不会提高策略本身的 locomotion 能力；策略仍需
通过 TensorBoard 和 sim2sim 评估后才能进入硬件测试。
