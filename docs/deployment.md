# 训练到实机部署

训练端和实机端不是“导出 ONNX 后直接运行”两步。中间必须固定并验证
机器人模型、关节顺序、观测构造、动作缩放、零点和安全限制。

## 阶段一：训练与导出

1. 用 `./scripts/validate.sh` 检查 URDF 的 17 个 link、16 个 joint、质量、惯量、轴和 mesh。
2. 用 `./scripts/train_smoke.sh` 跑 1 轮，确认环境能创建、观测为 45 维、动作为 12 维。
3. 用 100 轮训练检查 reward 是否正常，再开始 5000 轮或更长训练。每个正式 run 保留
   Git commit、随机种子、依赖提交号、URDF、`params/*` 和 TensorBoard 日志。
4. 对指定 checkpoint 执行 `./scripts/play_export.sh /path/model_N.pt`，得到
   `exported/policy.onnx`、`exported/policy.pt` 和训练生成的 `params/deploy.yaml`。

## 阶段二：sim2sim

`unitree_mujoco` 使用 MJCF，不会直接加载 URDF。因此需要先把
`ros2/src/custom_dog_description/urdf/custom_dog.urdf` 转换为等价的
`custom_dog.xml`，或使用 MuJoCo Python API 创建等价模型。转换后的模型必须保持
同一套几何尺寸、质量/惯量、关节轴、关节顺序、默认姿态、动作缩放和关节限制。
当前上游 `unitree_mujoco` 自带的是 Go2 等 Unitree 场景；自制狗需要单独添加
`unitree_robots/custom_dog/` 场景和控制适配器，不能把 Go2 的 XML 直接当作最终验证模型。
至少验收站立、前后左右速度、转向、随机扰动恢复、关节限位和策略断开后的停止行为。
sim2sim 失败时先修模型、坐标系、零点和执行器参数，不把问题直接带到实机。

## 阶段三：Orin NX Super

目标设备安装 Ubuntu 22.04、ROS 2 Humble、匹配 JetPack/CUDA 的 ONNX Runtime（或
TensorRT）。部署包应包含：

```text
policy.onnx
deploy.yaml
metadata.yaml
sha256sums.txt
```

ROS 2 节点按以下顺序工作：RS485 驱动读取 12 个电机状态 -> 将外部速度请求按候选
`command_calibration` 映射为 policy command -> 按 `docs/policy_contract.md` 构造 45 维
observation -> ONNX 推理 -> `offset + scale * action` -> 训练导出的 action clip ->
候选声明的可选目标 bias -> 机械、速度和通信安全限位 -> 通过四路 RS485 发送。原始
ONNX action 单独保存为下一帧 `last_action`。启动前必须完成电机 ID、方向、编码器零点、
限位、Kp/Kd、端口和急停配置。

当前 RL 候选使用固定的关节侧 `Kp=25`、`Kd=0.5`。GOM-8010-6 驱动按照
`Dog-control/qr_ws` 已采用的关节侧接口，在 SDK 边界做：

```text
motor_kp = joint_kp / gear_ratio^2
motor_kd = joint_kd / gear_ratio^2
```

hip/thigh 的传动比为 `6.33` 时，电机侧约为 `0.624/0.0125`；calf 总传动比为
`12.66` 时，电机侧约为 `0.156/0.00312`。训练、`deploy.yaml` 和实机 RL 状态必须
始终使用同一组关节侧值。VMC 控制器里随支撑/摆动相变化的增益不能直接复用。

实机开放顺序固定为：单电机无负载、单腿悬空、四腿悬空、低增益站立、安全绳落地、
低速策略。任何通信超时、过温、越限、急停或 observation 异常都必须进入零输出/阻尼状态。

当前行走候选是 `model_4500_yaw_straight`。它已经通过带候选 YAML（低速 command
calibration 和高速 target bias）校准的 Python
MuJoCo `0~3 m/s` 验收，详细结果见
[0~3 m/s 速度与偏航 sim2sim 报告](speed_straight_report_2026-08-09.md)。但 Unitree
C++ 和 ROS 2 controller 还没有实现 `command_calibration`/`joint_target_bias`，RS485 硬件驱动也只是接口骨架；
所以 `validated_for_hardware` 仍为 `false`，不能进入落地行走测试。

## Orin NX 与香橙派 5 Plus

Orin NX Super 是首选部署平台：有 NVIDIA GPU，TensorRT/ONNX Runtime 加速路径与
训练端的 CUDA 推理更一致。香橙派 5 Plus（RK3588）也可以作为部署主机，但它不能运行
Isaac Sim 训练；应在 WSL/带 NVIDIA GPU 的电脑训练和导出，然后在香橙派上使用
ARM64 的 ONNX Runtime（通常为 CPU 或 RKNN 转换路径）做推理。需要重新测量端到端
50 Hz 周期、串口调度抖动、温度和内存占用，不能直接假设与 Orin 性能相同。

因此建议先在 Orin NX 完成硬件闭环，再把同一 `policy.onnx` 和 `deploy.yaml` 迁移到
香橙派做性能对比；两者都必须通过悬空和安全绳验收后才能落地运行。
