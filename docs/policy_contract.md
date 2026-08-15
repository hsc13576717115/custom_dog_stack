# Policy Contract v2.4

控制周期为 0.02 s，即 50 Hz。现有 Actor 输入可能是 45、47、49、51、62 或 213 个浮点数，
输出始终是 12 个浮点数。
准确的 observation 列表必须从候选自己的 `deploy.yaml` 读取，不能根据模型文件名猜测。

## Observation

| 索引 | 内容 | 数量 | 缩放 |
|---:|---|---:|---:|
| 0..2 | base angular velocity | 3 | 0.2 |
| 3..5 | projected gravity | 3 | 1.0 |
| 6..8 | commanded x/y/yaw velocity | 3 | 1.0 |
| 9..20 | joint position minus default position | 12 | 1.0 |
| 21..32 | joint velocity | 12 | 0.05 |
| 33..44 | previous policy action | 12 | 1.0 |
| 45..46 | optional gait phase: sin/cos | 2 | 1.0 |
| 45..46 | optional base-frame linear velocity: vx/vy | 2 | 1.0 |
| 45..48 | optional trot clock: FR/FL/RR/RL | 4 | 1.0 |
| 49..50 | optional base-frame linear velocity with trot clock: vx/vy | 2 | 1.0 |
| 51..61 | privileged dynamics context (teacher only) | 11 | normalized |

基础策略到索引 44 结束，共 45 维。47 维策略只能在末尾选择一种扩展，具体类型由候选
`deploy.yaml` 中的 observation 名称决定，不能仅根据输入维度判断：

- `gait_phase` 是站稳后周期步态的 `sin/cos` 时钟；零命令时为 `[0, 0]`；
- `base_lin_vel_xy` 是机身坐标系实际 `vx/vy`，用于速度闭环反馈。

`base_lin_vel_xy` 在 Isaac 和 MuJoCo 中可直接取得；实机 IMU 不能直接测量平移速度，
ROS 2 部署必须接入经过时间同步和滤波的 IMU + 腿部运动学速度估计器。估计器完成前，
这种 47 维策略只能作为 sim/sim2sim 实验候选，不能标记为实机就绪。

51 维策略在基础 45 维之后依次追加 `trot_clock` 四维和 `base_lin_vel_xy` 两维。
它保持显式对角 Trot，同时使用仿真中的真实平移速度闭环；和 47 维策略一样，在真实
速度估计器完成前只能用于 sim/sim2sim。

62 维策略在 51 维后追加 11 维特权动力学上下文：总质量、基座质心 xyz、执行器刚度与
阻尼、关节摩擦、静/动摩擦材料、恢复系数以及当前执行器延迟。前 10 维在 startup
随机化完成后记录，延迟因每次 reset 都会重采样而实时读取。这是 HIM/RMA 风格教师的
训练接口，只用于仿真和蒸馏；硬件部署不得伪造这些量。MuJoCo 平地回归使用归一化的
名义零上下文，最终必须由 213 维历史学生替代该教师。

### 213 维短历史策略

213 维策略不直接使用机身平移速度。它只使用实机现有 IMU、编码器、速度指令和上一动作，
因此不依赖尚未完成的平移速度估计器。除速度指令外，每个 observation term 保留最近 5 帧，
覆盖 100 ms。布局是 term-major；每个 term 内按最旧帧到最新帧排列：

| Index | Observation | Frames x width | Scale |
| --- | --- | ---: | ---: |
| 0..14 | base angular velocity history | 5 x 3 | 0.2 |
| 15..29 | projected gravity history | 5 x 3 | 1.0 |
| 30..32 | current velocity command | 1 x 3 | 1.0 |
| 33..92 | relative joint position history | 5 x 12 | 1.0 |
| 93..152 | joint velocity history | 5 x 12 | 0.05 |
| 153..212 | previous policy action history | 5 x 12 | 1.0 |

reset、FixStand 到 PolicyHold 的接管以及策略重新加载时必须清空历史。清空后的第一次采样用
当前帧填满该 term 的全部 5 个位置；之后每 20 ms 丢弃最旧帧并追加当前帧。这与 Isaac Lab
`history_length=5`、`flatten_history_dim=true` 的行为一致，避免启动阶段人为注入四帧零值。

`deploy.yaml` 中每个 term 的 `scale` 仍只描述单帧宽度，Actor 总输入维数必须按
`sum(len(scale) * history_length)` 计算。Python MuJoCo、trace 分析器和 C++
`HistoryObservationBuilder` 都必须使用这一规则。

最终 history student 的训练分布继承 T1：平地样本保留完整 Stage-D 指令范围，崎岖地形
限制在 Stage-C 范围；教师的地形高度扫描只进入 critic，不进入 teacher actor 或 student。
蒸馏时 62-D teacher actor 能看到特权动力学上下文，213-D student 只能从 100 ms 本体历史
中推断其影响。当前蒸馏只替换 locomotion actor，零速仍由独立 stand actor 路由。因此“213 维输入可由
现有传感器构建”不等同于整个 routed 候选已可实机部署；stand bias、路由器及历史清空合同
还必须在 C++/ROS 2 中实现并单独验收。

## Deployment state machine

当前已验收的部署合同中，趴下到站立不属于 locomotion RL policy。部署端固定执行：

```text
Prone/Passive
-> FixStand: 从当前实测关节角到 HOME，2 秒五次平滑曲线
-> PolicyHold: policy 接管，1 秒 command=[0, 0, 0]
-> Velocity: 释放外部 vx/vy/wz
```

五次平滑系数为 `s(u)=10u^3-15u^4+6u^5`，`u` 限制在 `[0,1]`。FixStand 的起点
必须取切换瞬间的 12 个实测关节角，不能假设机器人恰好位于标称趴姿。进入 PolicyHold
时清空 `previous policy action` 和策略内部的 phase 计数。通信异常、越限或急停直接退出
Velocity，进入硬件定义的 Passive/阻尼安全状态。

仓库中的 `CustomDog-SelfRighting-R0/R1/R2-v2` 是独立的 recovery actor 实验合同，不能
覆盖上述已验收流程。只有它依次通过 Isaac 选择性自碰撞、MuJoCo 腹/背/左右侧躺、扰动后
连续稳定门槛，并完成安全绳实机试验后，部署状态机才允许扩展为：

```text
Fallen -> RecoveryPolicy -> UprightDwell(0.4 s) -> LocomotionPolicyHold -> Velocity
```

RecoveryPolicy 期间速度指令必须为零；成功切换要求机身高度至少 0.27 m、倾角不超过
15 deg、角速度不超过 0.50 rad/s，并且四足连续接触 0.40 s。超时、关节/力矩越限或重复
失败必须回到 Passive，不能自动循环尝试。

ROS 2 `DeploymentStateMachine` 也提供同一状态接口：`begin_recovery()` 后，调用带
`RecoveryTelemetry` 的 `update()`，输出通过 `use_recovery_policy` 选择恢复 ONNX。恢复成功
进入 `UprightDwell`，任一接触/姿态条件丢失就退回恢复；超时进入 Passive。该接口只定义
状态、安全门槛和策略历史重置，具体 ONNX Runtime 节点仍需在驱动层接入。

MuJoCo 的完整交接验收使用 `scripts/evaluate_recovery_handoff.py`。它同时加载独立的
RecoveryPolicy、stand/locomotion routed candidate，在 belly/back/left/right 四种初态逐一
确认 `self_righting` 成功、RecoveryPolicy 到 locomotion 的 0.30 s 平滑交接、1 s 零速保持，
以及最后才出现的 locomotion command release。单独的 `evaluate_self_righting_mujoco.py`
只证明恢复策略本身，不能替代这个交接门槛。

### Stand/locomotion routed candidate

`closed_loop_stage_a_routed_seed42` 是两个 Actor 组成的 sim2sim 候选：零速附近使用独立
stand policy，其余命令使用 locomotion policy。路由阈值从候选自己的 `routing.json` 读取：

```text
stand -> locomotion: planar speed >= 0.025 m/s or abs(wz) >= 0.040 rad/s
locomotion -> stand: planar speed <= 0.015 m/s and abs(wz) <= 0.025 rad/s
transition: 0.30 s quintic blend between the two raw policy actions
```

因此 `vx=0.03 m/s` 和 `abs(wz)=0.05 rad/s` 都会进入 locomotion policy 并抬脚；滞回区
避免键盘指令或估计噪声导致两个 Actor 来回切换。切换时两个 Actor 各自保留独立的
`previous action` 和 observation state，不能共享上一动作。

stand 候选还可以声明 12 维 `constant_joint_target_bias`。它使用 policy joint order，在
action scale/offset 后、`joint_ids_map` 前无条件加到目标关节角。当前验收值仅在 stand
policy 中将四个 hip 对称地向内偏置 `0.04 rad`；locomotion policy 不使用该偏置。这个字段
目前只在 Python MuJoCo runner 中实现，C++/ROS 2 还没有同合同实现，因此该 routed 候选
只能标记为 sim/sim2sim 已验收，不能标记为实机就绪。最终 213-D student 应吸收或替代
这个站立校准，不能把 Python 专用偏置当成永久硬件修正。

## Command path

外部控制器给出的速度是 requested command。候选可以在 `deploy.yaml` 中为
`lin_vel_x`、`lin_vel_y`、`ang_vel_z` 分别声明 `command_calibration`，将外部请求分段
线性插值为写入 observation `6..8` 的 policy command。这不改变 ONNX 输入维度。
`external_ranges` 是用户/导航速度的安全范围，`policy_ranges` 是校准后允许进入网络的范围；
没有这些字段的旧候选继续使用 `ranges`。Python、C++ 和 ROS 2 必须使用同一张表，日志和
安全限制仍以外部 requested command 为准。当前 ROS 2/C++ 部署端尚未实现三轴校准，
因此带校准的候选只能算 Python MuJoCo sim2sim 已验证，不能据此声明实机就绪。

## Action

```text
target_base = clip(default_joint_position + 0.25 * policy_action,
                   exported_action_clip)
desired_joint_position = target_base + speed_blend * joint_target_bias
```

处理顺序是：训练导出的 action clip、可选目标 bias、机械/速度/通信安全限位。

`model_4500_yaw_straight` 还声明了可选的 `joint_target_bias`。部署端必须从该候选自己的
`deploy.yaml` 读取 12 维 bias 和 `vx_range`，按前向速度指令线性混合：

```text
blend = clip((requested_vx - vx_min) / (vx_max - vx_min), 0, 1)
desired_joint_position += blend * joint_target_bias
```

bias 使用 policy joint order，并在 `joint_ids_map` 映射到 SDK 顺序之前应用。下一帧
observation 的 `previous policy action` 必须保留原始 ONNX 输出，不能使用含 offset、scale
或 bias 的目标关节位置。没有 `joint_target_bias` 的历史候选按 v1 行为处理。

## Position controller

当前候选使用固定关节侧 `Kp=25`、`Kd=0.5`，目标速度和前馈力矩均为 0。
训练、sim2sim 和实机 RL 状态必须一致。GOM-8010-6 SDK 使用电机侧增益时，由硬件
驱动根据每个关节的总传动比做平方换算；该换算不属于 policy ONNX。

`deploy.yaml` 中 `stiffness` 和 `damping` 已经按下方 canonical SDK joint order 导出，
部署端必须直接按 SDK 电机编号使用，不能再经过 `joint_ids_map`。`default_joint_pos`、
policy action 和可选 `joint_target_bias` 才是 policy joint order，需要由
`joint_ids_map` 映射到 SDK 顺序。

起身状态机的五次曲线终点也必须使用当前候选的 `default_joint_pos`，并按
`joint_ids_map` 映射到 SDK 顺序；不要在部署代码中另写一套固定 HOME 角度。

## Canonical SDK joint order

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

Isaac 导出的 policy action 顺序可能与 SDK 顺序不同。每个训练 run 生成的
`params/deploy.yaml` 中 `joint_ids_map` 是唯一权威映射，不允许凭名称排序猜测。
