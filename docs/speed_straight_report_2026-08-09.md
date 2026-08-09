# 0~3 m/s 速度与偏航 sim2sim 报告（2026-08-09）

## 结论

当前保留候选为 `deploy/candidates/model_4500_yaw_straight`。它在标准 MuJoCo 中通过了
请求速度 `0~3 m/s` 的 15 秒测试，以及 `0.1/0.25/0.5/2.5/2.75/3.0 m/s` 的 60 秒
压力测试。低速 command calibration 和高速 target bias 都来自候选自己的 `deploy.yaml`：

- 所有测试均未摔倒；
- 15 秒网格的前向速度绝对误差不超过 `0.067 m/s`；
- 60 秒压力测试的前向速度绝对误差不超过 `0.022 m/s`；
- 最大倾角不超过 `8.97 deg`，最小机身高度不低于 `0.229 m`；
- 3 m/s、60 秒的平均偏航角速度为 `0.008 rad/s`。

这表示该候选已经通过本项目当前的 Python MuJoCo 速度和姿态验收。它还不是实机发布包：
高速关节目标补偿尚未移植到 Unitree C++/ROS 2 控制器，真实电机零点、方向、急停和悬空
测试也没有完成。

## 候选来源

```text
training run: 2026-08-09_02-26-29_yaw_straight_finetune
checkpoint: model_4500.pt
task: CustomDog-Velocity-SpeedStraight-v1
observation/action: 45 -> 12
policy frequency: 50 Hz
joint-side PD: Kp=25, Kd=0.5
candidate: deploy/candidates/model_4500_yaw_straight
```

`SpeedStraight-v1` 把 `vy` 和 `yaw rate` 指令固定为零，并为“实际 yaw rate 与指令 yaw
rate 的差”增加非饱和 L2 奖励。该奖励不改变 observation 维数。候选没有采用同一 run
的最后一个 checkpoint：`model_4695` 在 3 m/s 的 MuJoCo 平均偏航角速度恶化到约
`0.145 rad/s`；后续长微调还出现了高速翻倒。模型选择以 sim2sim 验收为准，不能默认
选择训练轮数最大的文件。

## 外部速度命令校准

原始策略在低速存在步态启动死区：直接把 `0.25 m/s` 写入 observation 时，机身平均
速度只有约 `0.073 m/s`。候选使用显式控制器输入校准；外部请求仍记录为 `command_vx`，
表格中的误差也相对它计算，只有送入 policy 的 `velocity_commands` 使用插值后的值：

```yaml
command_calibration:
  lin_vel_x:
    requested: [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 3.0]
    policy: [0.0, 0.27, 0.286, 0.297, 0.337, 0.378, 0.452, 0.56, 0.62, 0.75, 3.0]
```

这张表属于 policy contract v1.2，Python/C++/ROS 2 必须一致实现；更换 CAD、摩擦、
PD 或电机零点后必须重新标定。它不改变 observation 仍为 45 维。

## 高速关节目标校准

原始 `model_4500` 在高速已经稳定，但 3 m/s 仍有约 `0.044 rad/s` 的系统性偏航。
候选 `deploy.yaml` 增加了策略关节顺序的微小 thigh 目标偏置：

```yaml
joint_target_bias:
  values: [0, 0, 0, 0, 0.02, -0.02, 0.02, -0.02, 0, 0, 0, 0]
  vx_range: [2.6, 3.0]
```

控制器按前向速度指令在 `2.6~3.0 m/s` 之间线性启用该偏置。它在 ONNX 推理后加到目标
关节位置，不修改策略输出，也不修改下一帧 45 维 observation 中的 `last_action`。因此
部署端计算为：

```text
raw_action = policy(observation)
target_base = clip(default_q + 0.25 * raw_action, exported_action_clip)
target_q = target_base + speed_blend * joint_target_bias
last_action_next = raw_action
```

这是对当前 CAD/MJCF 左右接触不对称的部署校准，不是已经在训练中学会的行为。真实机器人
必须重新测量偏置，不能未经悬空和低速测试直接照搬 `0.02 rad`。

## 15 秒速度网格

每个测试使用 3 秒 warmup，外部命令为 `(vx, 0, 0)`；若启用校准，`policy_vx` 另列在
控制器输出中：

| 外部指令 `vx` | 平均 `vx` | 绝对误差 | 平均 `vy` | 平均 yaw rate | 最小高度 | 最大倾角 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | -0.000 | 0.000 | 0.000 | 0.005 | 0.301 | 2.72 deg | 通过 |
| 0.10 | 0.095 | 0.005 | 0.002 | 0.014 | 0.280 | 4.03 deg | 通过 |
| 0.15 | 0.146 | 0.004 | 0.002 | 0.008 | 0.276 | 4.05 deg | 通过 |
| 0.20 | 0.210 | 0.010 | 0.000 | 0.008 | 0.294 | 3.93 deg | 通过 |
| 0.25 | 0.216 | 0.034 | -0.000 | 0.009 | 0.284 | 4.63 deg | 通过 |
| 0.30 | 0.293 | 0.007 | 0.005 | 0.009 | 0.295 | 4.73 deg | 通过 |
| 0.40 | 0.400 | 0.000 | 0.003 | 0.007 | 0.300 | 4.99 deg | 通过 |
| 0.50 | 0.496 | 0.004 | -0.034 | -0.009 | 0.283 | 4.34 deg | 通过 |
| 0.60 | 0.599 | 0.001 | -0.058 | -0.017 | 0.307 | 3.24 deg | 通过 |
| 0.75 | 0.770 | 0.020 | -0.050 | -0.004 | 0.307 | 2.95 deg | 通过 |
| 1.00 | 1.062 | 0.062 | -0.050 | 0.003 | 0.302 | 3.76 deg | 通过 |
| 1.50 | 1.567 | 0.067 | -0.076 | -0.007 | 0.268 | 6.53 deg | 通过 |
| 2.00 | 2.015 | 0.015 | -0.146 | -0.023 | 0.245 | 8.97 deg | 通过 |
| 2.50 | 2.491 | 0.009 | -0.092 | 0.033 | 0.252 | 6.90 deg | 通过 |
| 2.75 | 2.707 | 0.043 | -0.082 | 0.027 | 0.239 | 7.73 deg | 通过 |
| 3.00 | 3.007 | 0.007 | -0.023 | 0.003 | 0.241 | 7.20 deg | 通过 |

表中速度单位均为 `m/s`，yaw rate 单位为 `rad/s`，高度单位为 `m`。

验收门槛为：不摔倒、`max_tilt <= 15 deg`、`min_height >= 0.22 m`；
`0.1~0.5 m/s` 要求 `abs_error <= max(0.05, 0.25*vx)`，`0.5~3 m/s` 要求
`abs_error <= 0.15 m/s`。表中所有请求速度点均通过。`vy` 和 yaw rate 是机身坐标系
统计量，不代表世界系航向锁定。

## 60 秒压力测试

| 外部指令 `vx` | 平均 `vx` | 绝对误差 | 平均 `vy` | 平均 yaw rate | 最小高度 | 最大倾角 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 0.101 | 0.001 | 0.000 | 0.014 | 0.266 | 4.07 deg | 通过 |
| 0.25 | 0.246 | 0.004 | 0.006 | 0.004 | 0.289 | 4.63 deg | 通过 |
| 0.50 | 0.503 | 0.003 | -0.027 | -0.003 | 0.267 | 5.36 deg | 通过 |
| 2.50 | 2.522 | 0.022 | -0.145 | 0.015 | 0.240 | 7.77 deg | 通过 |
| 2.75 | 2.754 | 0.004 | -0.107 | 0.019 | 0.229 | 7.97 deg | 通过 |
| 3.00 | 3.016 | 0.016 | -0.026 | 0.008 | 0.239 | 7.32 deg | 通过 |

表中速度单位均为 `m/s`，yaw rate 单位为 `rad/s`，高度单位为 `m`。

平均 yaw rate 较小不等于全局航向锁定。当前 observation 没有世界系 heading，长距离仍会
积累航向误差；需要直线导航时，应在后续版本增加 IMU heading 外环或训练 heading command，
而不是把本表理解为全局直线轨迹验收。2.0~2.75 m/s 仍有约 `0.1~0.15 m/s` 的侧向
速度偏差，也应在实机前继续优化或由导航外环约束。

## 策略契约复核

0.25 m/s、5 秒 trace 共 250 个策略周期，使用候选 ONNX 和 `deploy.yaml` 离线重算；
其中 observation 中的命令为校准后的 `0.337 m/s`：

| 检查项 | 最大绝对误差 |
|---|---:|
| 45 维 observation | `2.98e-8` |
| ONNX 12 维 action | `0` |
| 处理后的目标位置 | `0` |

全部小于 `1e-5`。`joint_ids_map`、observation scale、command calibration、action
scale、目标偏置和 `last_action` 反馈一致。

## 复现命令

```bash
cd /home/hsc/custom_dog_stack

for vx in 0 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.75 1.0 1.5 2.0 2.5 2.75 3.0; do
  ./scripts/run_sim2sim.sh \
    --policy deploy/candidates/model_4500_yaw_straight/exported/policy.onnx \
    --deploy-yaml deploy/candidates/model_4500_yaw_straight/params/deploy.yaml \
    --command "$vx" 0 0 --duration 15 --warmup 3
done

./scripts/run_sim2sim.sh \
  --policy deploy/candidates/model_4500_yaw_straight/exported/policy.onnx \
  --deploy-yaml deploy/candidates/model_4500_yaw_straight/params/deploy.yaml \
  --command 0.25 0 0 --duration 5 --warmup 0 \
  --trace /tmp/model_4500_trace.csv --trace-limit 0

/home/hsc/.conda/envs/custom_dog_mujoco/bin/python \
  scripts/analyze_policy_trace.py \
  /tmp/model_4500_trace.csv deploy/candidates/model_4500_yaw_straight \
  --expected-command 0.25 0 0
```

可视化回放：

```bash
./scripts/view_mujoco_policy.sh deploy/candidates/model_4500_yaw_straight 3.0 0.0 0.0
```

官方 Unitree MuJoCo + SDK2 bridge 已确认能够加载候选并进入 `Velocity`，但当前 C++
action 后处理不读取 `command_calibration` 或 `joint_target_bias`，只能作为接口闭环检查。在把候选复制到
`deploy/releases/` 或 Orin NX 之前，必须按部署文档把同一公式实现到 C++/ROS 2，并重新
完成 trace 对齐、低速悬空和落地测试。
