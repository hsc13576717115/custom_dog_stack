# 工程架构

## 依赖方向

```text
custom_dog_description
        |
        +--> custom_dog_rl --> Isaac Lab --> unitree_rl_lab
        |
        +--> custom_dog_hardware --> USB / four RS485 buses
        |
        +--> custom_dog_controller --> ONNX Runtime
        |
        +--> custom_dog_bringup
```

机器人描述是训练、sim2sim 和 ROS 2 的共同几何来源。策略接口由
`docs/policy_contract.md` 定义，训练和部署必须遵守同一顺序和缩放。

## 边界

### `custom_dog_description`

只保存几何、惯量、关节树、关节限制和可视化启动文件。它不包含电机通信
或策略逻辑。

### `custom_dog_rl`

负责 Isaac Lab asset、task、reward、observation、action、PPO 和策略导出。
当前继承锁定版本的 Go2 locomotion MDP，但使用独立 Gym task ID。

### `custom_dog_hardware`

负责 RS485 帧、端口、超时、状态校验和急停。上层不能直接操作串口。

### `custom_dog_controller`

负责 observation 构造、ONNX 推理、action 后处理和安全限幅，不负责串口协议。

### `custom_dog_bringup`

只组合参数和节点。硬件 ID、方向和零点只允许出现在配置文件中。

## 上游策略

Isaac Lab、unitree_rl_lab 和 unitree_mujoco 保持外部仓库，由
`third_party/versions.yaml` 锁定提交。项目代码不得依赖上游工作区中的
未提交自制文件。
