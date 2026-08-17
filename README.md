# Custom Dog Stack

自制四足机器人的机器人描述、ROS 2 控制、MuJoCo sim2sim 和历史强化学习工程。

> [!IMPORTANT]
> 本仓库从 2026-08-17 起进入低维护状态，不再作为强化学习训练主线。
> 后续策略训练、地形课程和运动能力优化统一在
> [`hsc13576717115/himloco_custom_dog`](https://github.com/hsc13576717115/himloco_custom_dog)
> 开发。本仓库继续保存机器人描述、ROS 2、MuJoCo、部署接口和历史实验，避免两套
> 训练框架继续分叉。

## 当前结论

- 当前综合最佳 HimLoco 模型：**`model_12000`**。
- 本机 checkpoint：
  `../himloco_lab/logs/himloco_rsl_rl/go2_rough/2026-08-15_14-38-13_custom_dog_default/model_12000.pt`。
- 本机 MuJoCo 候选：`deploy/candidates/himloco_model_12000`。
- `model_14100`、`model_14900`、`model_14999` 的随机地形续训提高了部分速度跟踪和
  Isaac Lab 地形生存指标，但没有超过 `model_12000` 的 MuJoCo 综合验收结果。
- 36 组 MuJoCo 测试中，`model_12000` 完整通过 16 组；表现最好的续训模型
  `model_14999` 完整通过 7 组。原始数据见
  [`reports/himloco_terrain_refine_2026-08-17`](reports/himloco_terrain_refine_2026-08-17)。
- 当前 locomotion 策略只负责从正常站姿开始运动，不具备跌倒后自主起身能力。

“最佳”表示当前固定 MuJoCo 测试矩阵下的综合结果，不代表已经通过真实机器人验证，
也不表示后续 checkpoint 一定更差。模型升级必须重新执行相同验收，不能只比较训练
reward 或 checkpoint 编号。

## 后续开发边界

### HimLoco 主仓库

以下工作只在 `himloco_custom_dog` 继续开发：

- 强化学习任务、PPO 配置和 observation/reward 设计；
- 平地到随机地形的课程训练；
- 速度跟踪、姿态稳定、步态与抗扰优化；
- 独立起身策略以及 locomotion/recovery 状态机；
- MID-360、D435i 等外感知输入和感知运动策略；
- 新 checkpoint、导出器和训练结果选择。

本机路径：

```text
/home/hsc/Dog_RL/himloco_lab
```

### 本仓库

本仓库保留并维护以下内容：

- `ros2/src/custom_dog_description`：URDF、mesh 和机器人几何参数的唯一来源；
- `ros2/src`：ROS 2 Humble 硬件、控制器和 bringup；
- `sim2sim`：MuJoCo 模型、策略运行器和键盘控制；
- `scripts/evaluate_mujoco_grid.py`：固定命令网格与地形回归测试；
- `deploy`：策略接口、候选说明和部署契约；
- `docs`、`reports`：历史设计、实验和迁移记录；
- `rl`：旧 CustomDog Isaac Lab 训练框架，仅作基线和历史参考。

除修复可复现问题外，不再向 `rl` 添加新任务或继续长时间训练。机器人几何、实机接口
或 MuJoCo 物理参数发生变化时，仍应同步更新本仓库。

## 仓库结构

```text
rl/                 旧 Isaac Lab 训练框架，保留但不再作为主线
ros2/src/           机器人描述、硬件接口、控制器和 bringup
sim2sim/            CustomDog MuJoCo 模型与策略运行器
deploy/             策略契约和本地候选目录
docs/               架构、部署、安全和迁移文档
reports/            可复现实验结果
scripts/            环境检查、构建、导出和评估工具
tests/              不依赖完整仿真器的静态与契约测试
third_party/        固定版本的外部依赖和控制子模块
```

## 环境准备

目标系统为 Ubuntu 22.04。完整依赖说明见
[`docs/ubuntu2204_migration.md`](docs/ubuntu2204_migration.md)。

```bash
git clone --recurse-submodules https://github.com/hsc13576717115/custom_dog_stack.git
git clone https://github.com/hsc13576717115/himloco_custom_dog.git himloco_lab

cd custom_dog_stack
git lfs pull
./scripts/check_dependency_versions.sh
./scripts/check_system.sh
```

HimLoco 读取本仓库机器人描述时设置：

```bash
export CUSTOM_DOG_DESCRIPTION_DIR=/home/hsc/Dog_RL/custom_dog_stack/ros2/src/custom_dog_description
export OMNI_KIT_ACCEPT_EULA=YES
```

## 验证机器人描述

URDF、mesh、质量、惯量、关节轴或限位变更后必须执行：

```bash
cd /home/hsc/Dog_RL/custom_dog_stack
./scripts/validate.sh
```

公共 `custom_dog.urdf` 是质量、惯量、视觉、关节树和限位的唯一来源，并保留所有
17 个 link 的完整 primitive 碰撞体。仿真器使用显式派生文件：

- HimLoco/Isaac：`custom_dog_selective_collision.urdf`，使用标定后的 thigh/calf 代理；
- Gazebo 传统控制：`custom_dog_gazebo_point_foot.urdf`，按 NMPC/WBC 四足端接触假设
  去除 thigh/calf 地面碰撞；
- 真机、Pinocchio 和模型契约：公共 `custom_dog.urdf`。

两个派生文件必须通过 `scripts/generate_selective_collision_urdf.py` 和
`scripts/generate_gazebo_point_foot_urdf.py` 从公共模型生成，不能手工修改公共 URDF 来
解决单一仿真器问题。碰撞几何仍需结合 CAD 和实物继续标定；点足模型不代表实机腿段
不会碰撞。

## 使用当前最佳模型

`model_12000` 的导出候选保存在本机忽略目录中。运行 MuJoCo 键盘控制：

```bash
cd /home/hsc/Dog_RL/custom_dog_stack
./scripts/teleop_mujoco_policy.sh deploy/candidates/himloco_model_12000
```

如果候选目录不存在，需要先从 HimLoco checkpoint 导出 encoder、policy 和
`deploy.yaml`。策略文件和训练日志默认不直接提交到普通 Git 历史；需要长期发布时使用
Git LFS 或 GitHub Release，并同时保存 checkpoint、配置、提交号和评估报告。

## MuJoCo 回归测试

固定命令网格评估支持同时加载 HimLoco encoder 和 policy：

```bash
python3 scripts/evaluate_mujoco_grid.py \
  --candidate model_12000=deploy/candidates/himloco_model_12000 \
  --baseline-label model_12000 \
  --absolute-only \
  --stage A \
  --duration 10 \
  --warmup 2 \
  --output-csv /tmp/custom_dog_eval.csv \
  --output-json /tmp/custom_dog_eval.json
```

已有评估数据：

- [`reports/himloco_mujoco_2026-08-16`](reports/himloco_mujoco_2026-08-16)：
  速度网格和 20/35/50 mm 粗糙地形基线；
- [`reports/himloco_terrain_refine_2026-08-17`](reports/himloco_terrain_refine_2026-08-17)：
  `model_12000` 与地形续训 checkpoint 对比。

至少检查速度误差、非指令轴串扰、机身高度、倾斜、非法腿部触地、髋角、足端触地转换
和是否摔倒。真实机器人部署前还必须完成悬空测试、关节方向与零点标定、急停和限幅测试。

## ROS 2

ROS 2 工作区位于 `ros2`：

```bash
cd /home/hsc/Dog_RL/custom_dog_stack/ros2
colcon build --symlink-install
source install/setup.bash
```

硬件控制仍有以下未完成项：

- GOM-8010-6 RS485 协议和异常处理最终验证；
- 编码器零点、方向、减速比和力矩常数逐关节标定；
- 硬件急停、通信超时、姿态和关节软件限位；
- HimLoco encoder/policy 的实时推理接入；
- 仿真到实机的分阶段验收。

在这些项目完成前，任何模型都不能直接视为实机可部署版本。

## 已知限制

- `model_12000` 不能在侧躺、仰躺或翻倒后自主起身；需要单独 recovery policy 和状态机。
- 当前策略没有 MID-360 或 D435i 输入，复杂地形属于无外感知盲走。
- Isaac Lab 与 MuJoCo 的接触、摩擦和碰撞模型仍存在差异。
- 平地站姿仍存在髋关节内收和左右不完全对称问题。
- 地形续训模型存在机身偏高、纯旋转累计误差增大等回归。
- MuJoCo 通过不等于实机通过。

## 维护规则

1. 新强化学习功能提交到 `himloco_custom_dog`。
2. URDF、mesh、ROS 2 和 MuJoCo 兼容修改提交到本仓库。
3. 每个候选模型必须绑定训练仓库 commit、checkpoint、配置和固定评估报告。
4. 不以总 reward 或最后一个 checkpoint 作为唯一选模依据。
5. 大模型和训练日志使用 Git LFS 或 Release，不提交生成缓存和完整日志目录。
6. 提交作者统一为 `ShiChang Huang <1583924560@qq.com>`。

## 历史资料

旧训练脚本和设计文档仍保留，便于复现实验或迁移功能。它们描述的是当时的状态，若与
本 README 冲突，以本 README 的仓库边界和当前评估结论为准。

主要维护者：ShiChang Huang <1583924560@qq.com>
