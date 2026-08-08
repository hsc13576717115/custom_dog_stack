# Policy Contract v1

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

## Action

```text
desired_joint_position = default_joint_position + 0.25 * policy_action
```

最终命令还必须经过机械限位、速度限制和通信安全限制。

## Canonical SDK joint order

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

Isaac 导出的 policy action 顺序可能与 SDK 顺序不同。每个训练 run 生成的
`params/deploy.yaml` 中 `joint_ids_map` 是唯一权威映射，不允许凭名称排序猜测。
