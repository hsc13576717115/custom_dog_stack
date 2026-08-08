# Custom Dog MuJoCo

这里保存由主 URDF 生成的 MJCF、凸包网格、传感器定义和 ONNX sim2sim 控制器。
URDF 仍然是机器人几何、质量、惯量、关节轴和限位的唯一源文件。

## 生成并验证 MJCF

```bash
cd /home/hsc/custom_dog_stack
./scripts/setup_mujoco.sh
./scripts/generate_mjcf.sh
```

生成脚本会转换 URDF、恢复完整惯量张量、加入浮动基座、12 个位置执行器、IMU、
四个足端接触传感器和 `home` keyframe，然后执行 1 秒无界面站立测试。

## 仅验证模型站立

```bash
./scripts/run_sim2sim.sh --duration 10
```

带 MuJoCo 窗口运行：

```bash
./scripts/run_sim2sim.sh --duration 30 --viewer
```

## 加载 ONNX 策略

`policy.onnx` 和 `deploy.yaml` 必须来自同一个训练 run：

```bash
./scripts/run_sim2sim.sh \
  --policy logs/rsl_rl/custom_dog_velocity/<run>/exported/policy.onnx \
  --deploy-yaml logs/rsl_rl/custom_dog_velocity/<run>/params/deploy.yaml \
  --command 0.3 0.0 0.0 \
  --duration 30 \
  --viewer
```

控制器严格使用 `deploy.yaml` 中的 observation 顺序、缩放、动作 offset/scale、
`joint_ids_map` 和 `step_dt`。当前模型的 MuJoCo 步长为 0.005 秒，策略每四个仿真步
执行一次，即 50 Hz。

当前仓库的 Python runner 是自制 RS485 控制栈的主 sim2sim 路径。
`/home/hsc/unitree_mujoco` 可以用来查看模型或参考 Unitree SDK2 桥接实现，但其默认
通信对象是宇树整机，不会直接替代自制狗的 ROS 2 RS485 驱动。
