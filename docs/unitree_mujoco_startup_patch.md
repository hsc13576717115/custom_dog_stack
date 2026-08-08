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
- bridge 线程读写 `mjData` 时使用 `sim->mtx`，避免 1 kHz SDK2 线程和 MuJoCo
  物理线程并发访问同一组状态和控制量。
- 设置 `UNITREE_MUJOCO_BRIDGE_TRACE=/path/bridge.csv` 时记录关节反馈、命令、增益和
  力矩，最多 10000 行，用于定位交接冲击。

在 `/home/hsc/unitree_rl_lab/deploy/include/FSM/State_FixStand.h`：

- `FixStand::enter()` 的插值起点必须读取 `lowstate->msg_.motor_state()[i].q()`，
  不能读取刚建立 DDS 时尚未初始化的 `lowcmd`。

在 `/home/hsc/unitree_rl_lab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`：

- `JointAction::reset()` 必须把 `_processed_actions` 初始化为 YAML 中的 `offset`。
  否则刚从 `FixStand` 进入 `Velocity` 时，策略线程完成第一次推理前会短暂发送全零
  关节目标，造成自制模型的突然大幅运动。

在 `/home/hsc/unitree_rl_lab/deploy/include/isaaclab/manager/action_manager.h` 和
`deploy/include/FSM/State_RLBase.h`：

- 为 action 的 reset、推理结果和 1 kHz 控制线程读取增加互斥保护，避免数据竞争。
- 从 `FixStand` 进入 `Velocity` 时，以当前实测关节位置为起点，在
  `transition_duration` 内平滑接管策略目标。
- 设置 `UNITREE_RL_POLICY_TRACE=/path/policy.csv` 时，同一行记录 45 维 observation、
  12 维原始 action、目标关节位置和实测状态。

本项目的 `sim2sim/unitree_deploy/config.template.yaml` 使用：

```yaml
FSM:
  FixStand:
    ts: [0.0, 2.0]
  Velocity:
    transition_duration: 1.0
```

`0.2 s` 起身会造成很大的接触冲击；`2.0 s` 起身和 `1.0 s` 策略接管是当前自制模型
通过官方 bridge 站立测试的一部分，不能在部署时省略。

## 重新构建

```bash
cmake --build /home/hsc/unitree_mujoco/simulate/build -j4
cmake --build /home/hsc/unitree_rl_lab/deploy/robots/go2/build -j4
```

然后回到本仓库执行：

```bash
./scripts/run_unitree_sim2sim.sh deploy/candidates/model_5498_robust
```

这些修改只解决启动时序和数据初始化，不会提高策略本身的 locomotion 能力；策略仍需
通过 TensorBoard 和 sim2sim 评估后才能进入硬件测试。
