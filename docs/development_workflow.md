# 开发流程

## 1. 机器人描述

运行 `./scripts/validate.sh`，确认关节树、mesh、质量、惯量、轴和限制。
更换 URDF 后必须重新执行 64 环境训练烟雾测试。

## 2. 训练

先运行 1 轮，再运行 100 轮，最后才运行 5000 轮。正式训练记录：

- Git commit
- 依赖提交号
- seed 和环境数量
- `env.yaml`、`agent.yaml`、`deploy.yaml`
- TensorBoard 日志和最终 checkpoint

## 3. sim2sim

将同一 URDF、关节顺序、电机限制和策略导入 MuJoCo。通过站立、速度指令、
扰动恢复和关节限位测试后，才能进入实机阶段。

## 4. 实机 bring-up

按以下顺序逐级开放：

```text
单电机无负载和关节方向/零点校验
-> 单腿悬空
-> 四腿悬空
-> 固定目标位置测试并逐步升到关节侧 Kp=25/Kd=0.5
-> 安全绳保护下落地
-> 小速度策略
```

每一级必须具备通信超时归零、力矩限幅、温度保护和物理急停。
GOM-8010-6 的电机侧增益由驱动按减速比平方换算，策略和训练配置只保存关节侧增益。

## 5. 策略发布

只有验收后的策略进入 `deploy/releases/vX.Y.Z/`。发布目录必须包含
`policy.onnx`、`deploy.yaml`、`metadata.yaml` 和 SHA-256 校验值。

训练到部署的完整验收清单见 [部署流程](deployment.md)。Orin NX Super 是首选实机平台；
香橙派 5 Plus 只能作为推理平台，不能替代带 NVIDIA GPU 的训练机。
