# Policy Contract v1.2

控制周期为 0.02 s，即 50 Hz。Actor 输入 45 个浮点数，输出 12 个浮点数。

## Observation

| 索引 | 内容 | 数量 | 缩放 |
|---:|---|---:|---:|
| 0..2 | base angular velocity | 3 | 0.2 |
| 3..5 | projected gravity | 3 | 1.0 |
| 6..8 | commanded x/y/yaw velocity | 3 | 1.0 |
| 9..20 | joint position minus default position | 12 | 1.0 |
| 21..32 | joint velocity | 12 | 0.05 |
| 33..44 | previous policy action | 12 | 1.0 |

## Command path

外部控制器给出的速度是 requested command。候选可以在 `deploy.yaml` 中声明
`command_calibration.lin_vel_x`，将外部请求插值为写入 observation `6..8` 的 policy
command。当前候选用它跨过低速步态死区；这不改变 ONNX 的输入维度。Python、C++ 和
ROS 2 必须使用同一张表，日志和安全限制仍以外部 requested command 为准。

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

当前 v1.2 候选使用固定关节侧 `Kp=25`、`Kd=0.5`，目标速度和前馈力矩均为 0。
训练、sim2sim 和实机 RL 状态必须一致。GOM-8010-6 SDK 使用电机侧增益时，由硬件
驱动根据每个关节的总传动比做平方换算；该换算不属于 policy ONNX。

## Canonical SDK joint order

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

Isaac 导出的 policy action 顺序可能与 SDK 顺序不同。每个训练 run 生成的
`params/deploy.yaml` 中 `joint_ids_map` 是唯一权威映射，不允许凭名称排序猜测。
