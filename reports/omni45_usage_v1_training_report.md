# CustomDog Omni45 Usage-v1 第一阶段报告

## 结论

当前可冻结的候选是 `deploy/candidates/omni45_usage_v1_model_4840/`，对应
`model_4840.pt`。它保持 45 维 observation 和 12 维 action，在 MuJoCo 中以 30 秒
回放、2 秒 warmup 验证下面的 6 个命令全部通过：

```text
(0.3, 0.0, 0.0)
(0.5, 0.0, +0.2)  (0.5, 0.0, -0.2)
(0.5, +0.1, 0.0)  (0.5, -0.1, 0.0)
(0.5, +0.1, +0.2)
```

验收门槛为不跌倒、`|vx|` 误差不超过 `max(0.10, 15%)`、`|vy|<=0.07`、
`|wz|<=0.10`、最低高度不低于 `0.23 m`、最大倾角小于 `10 deg`，并且姿态、
动作变化率和足端滑移不劣于 `model_4600` 基线。候选的汇总最差值为：

| 指标 | model_4600 基线 | model_4840 |
|---|---:|---:|
| 最大 `vx` 误差 (m/s) | 0.099 | 0.077 |
| 最大 `vy` 误差 (m/s) | 0.093 | 0.054 |
| 最大 `wz` 误差 (rad/s) | 0.114 | 0.092 |
| 最低高度 (m) | 0.237 | 0.321 |
| 最大倾角 (deg) | 12.95 | 7.15 |
| hip 外展均值 (deg) | 13.19 | 5.95 |
| hip 外展最大值 (deg) | 39.84 | 16.97 |

## 训练链路

种子是 `/home/hsc/unitree_rl_lab` 导出的 `model_4600.pt`，训练使用 CPU PhysX、
CUDA PPO learner、128 个环境、固定学习率 `3e-5`、初始 action std `0.10`。
第一轮 `omni45_usage_v1_from4600_200` 因小命令 reward 阈值和 bucket 统计不合适而拒绝；
修正后运行 `omni45_usage_v1_bucketfix_from4600_200`，再从其 `model_4799.pt` 做
100 iterations 的侧向/转向微调，得到以下筛选点：

| checkpoint | 10 s 通过命令数 | 30 s 通过命令数 | 结论 |
|---|---:|---:|---|
| `model_4800` | 6/6 | 5/6 | 不冻结 |
| `model_4840` | 6/6 | 6/6 | 冻结候选 |
| `model_4860` | 6/6 | 5/6 | 不冻结 |
| `model_4880` | 3/6 | 未进入终选 | 退化 |
| `model_4898` | 2/6 | 未进入终选 | 退化 |

训练中的关键日志包括 `tracking_error/vx|vy|wz`、`body/min_height`、`body/max_tilt`、
`style/hip_outward_*`、`feet/slip`、`feet/impact_velocity`、`action/delta|delta2`、
`termination/*` 以及每个 command mode/speed bucket 的完整窗口计数。只有完成窗口参与
统计；每个实际填充的 bucket 至少需要 50 个完整窗口，并连续 3 个评估窗口通过后才允许
扩展 curriculum。

## 契约验证

使用候选 ONNX 和同目录 `params/deploy.yaml` 做 5 秒 MuJoCo trace，
`scripts/analyze_policy_trace.py` 结果为：

```text
observation contract: max_abs_error=2.96486e-08
ONNX action: max_abs_error=0
processed target: max_abs_error=0
PASS: all contracts are within tolerance 1e-05
```

## 当前边界

这是低速 `vx`、`vx+wz` 以及小幅 `vy` 的闭环候选，不是 `0~3 m/s` 全速模型；也没有
验证纯 `vy`、纯 `wz`、后退、复杂地形、外力扰动或真实电机。趴下到站立仍由部署状态机用
约 2 秒五次曲线插值完成，RL policy 只负责站稳后的速度控制。实机部署前应按单电机、悬空、
安全绳、小速度的顺序执行硬件验收。

## 复现命令

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/custom_dog_velocity
```

查看冻结候选：

```bash
./scripts/run_sim2sim.sh \
  --policy deploy/candidates/omni45_usage_v1_model_4840/exported/policy.onnx \
  --deploy-yaml deploy/candidates/omni45_usage_v1_model_4840/params/deploy.yaml \
  --command 0.5 0.1 0.2 --duration 15 --warmup 2 --viewer \
  --camera-mode tracking
```

30 秒原始结果在候选目录的 `validation/grid_finalists_30s.{csv,json}`；不要把候选模型
直接用于 Orin NX，先完成部署端 observation、关节映射、限位、急停和悬空测试。
