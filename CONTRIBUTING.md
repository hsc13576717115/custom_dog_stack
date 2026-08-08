# Development Rules

1. 从 `main` 创建短生命周期分支，例如 `feature/rs485-transport`。
2. 修改前先运行 `./scripts/validate.sh`，修改后再次运行。
3. 修改 RL 行为契约时，同时更新 `docs/policy_contract.md`。
4. 不提交训练日志、原始 checkpoint、ROS 构建目录和 Isaac 缓存。
5. 提交信息使用明确范围，例如 `rl: calibrate calf actuator envelope`。
6. 合入 `main` 前至少完成静态验证和 64 环境一轮训练。
7. 实机代码还必须完成超时、限幅、温度和急停测试。
