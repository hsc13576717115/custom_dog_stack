# 鲁棒微调与 sim2sim 报告（2026-08-08）

## 训练来源

本次 run 从 `2026-08-08_15-57-41/model_4999.pt` 恢复，使用
`CustomDog-Velocity-Robust-v0` 继续训练 500 轮：

```text
run: 2026-08-08_21-31-25_robust_finetune
final checkpoint: model_5498.pt
candidate: deploy/candidates/model_5498_robust
simulation: CPU PhysX
PPO: RTX 4060 CUDA
```

鲁棒任务增加 roll/pitch 初始偏差、关节初始位置偏差和周期性推力，速度采样限制为
`vx [-0.3,0.3]`、`vy [-0.15,0.15]`、`yaw [-0.5,0.5]`。最后一段训练的平均 episode
长度约为 `998/1000`，`bad_orientation` 终止约为 `1.6%`。训练进程退出码为 0；退出时
出现的 PhysX CUDA 设备日志是当前 WSL 图形设备问题，不是本次训练失败。

## 接口逐项校验

Python MuJoCo trace 和官方 SDK2 bridge trace 均使用导出的 `deploy.yaml` 重新构造
observation、重新运行 ONNX，并重新计算目标位置：

| 路径 | observation 最大误差 | ONNX action 最大误差 | 目标位置最大误差 |
|---|---:|---:|---:|
| Python MuJoCo | `2.93e-8` | `0` | `0` |
| Unitree SDK2 bridge | `1.67e-8` | `4.99e-9` | `7.75e-8` |

全部小于 `1e-5`，说明 45 维 observation、12 维 action、缩放和关节映射在两条部署
路径中一致。

## sim2sim 结果

| 测试 | 结果 | 结论 |
|---|---|---|
| 标准 MuJoCo，零速 30 s，`Kp=25/Kd=0.5` | 高度 `0.3115 m`，未摔倒 | 通过站立 |
| 标准 MuJoCo，`vx=0.2 m/s` 30 s | 最终 x `-0.0102 m`，未摔倒 | 稳定但没有速度跟踪 |
| Unitree MuJoCo + SDK2，零速，`Kp=25/Kd=0.5` | `Velocity` 持续到测试超时，1501 个策略周期 | 通过 30 s 闭环站立 |
| SDK2，套用 qr_ws 四足支撑增益 | 进入 `Velocity` 约 `0.78 s` 后回到 `Passive` | 不可直接替换 |

因此 `model_5498_robust` 是“完整训练、ONNX 导出、两套 MuJoCo、SDK2 数据闭环均已
跑通”的候选模型，但只通过零速站立验收，没有通过行走验收，仍保持
`validated_for_hardware: false`。

## Kp/Kd 结论

[`Dog-control` 的 `qr_ws` 分支](https://github.com/hsc13576717115/Dog-control/tree/qr_ws)
把上层 `Kp/Kd` 定义为关节侧增益，再由 `IOSDK` 除以减速比平方后写入 GOM-8010-6：

```text
motor_kp = joint_kp / gear_ratio^2
motor_kd = joint_kd / gear_ratio^2
```

该分支当前 trot 的四足支撑、两足支撑和摆动腿使用不同的关节增益，并叠加 VMC
前馈力矩。它们属于整套 VMC 控制器参数，不能作为固定 RL PD 参数直接复制。

当前 RL 保持关节侧 `Kp=25`、`Kd=0.5`：它与 Isaac 执行器、候选 `deploy.yaml`、
标准 MuJoCo 和 SDK2 bridge 一致。Orin NX 的 GOM 驱动必须在最底层按 hip/thigh
`6.33`、calf 总传动比 `12.66` 做平方换算，不能把 `25/0.5` 原样当作电机侧 SDK 增益。

## 下一阶段

下一轮训练目标不是继续增加站立 reward，而是解决速度跟踪。需要降低静止样本比例、
提高非零速度样本占比，并在保持 reset 扰动的情况下逐步扩大速度范围。只有前进、后退、
侧移和转向在两套 sim2sim 中均通过，才进入单腿和四腿悬空实机测试。
