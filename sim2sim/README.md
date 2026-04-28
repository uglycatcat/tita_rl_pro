# MuJoCo sim2sim (ONNX)

本目录提供一个独立的 MuJoCo sim2sim 推理闭环：
- 固定模型路径：`sim2sim/stairs_test.onnx`
- 固定平地场景：`resources/tita/mjcf/flat_scene.xml`
- 控制链路：`reindex + action_scale + filter + decimation + PD + 轮关节特殊项`

## 1. 依赖

建议在项目已有环境中安装：

```bash
pip install mujoco onnxruntime numpy
```

如果你有 CUDA 版 ONNXRuntime，也可自行替换为 `onnxruntime-gpu`。

## 2. 目录约定

- ONNX 文件必须放在：`sim2sim/stairs_test.onnx`
- 主脚本：`sim2sim/mujoco_onnx_runner.py`
- 一键脚本：`sim2sim/run_mujoco_onnx.sh`

## 3. 一键运行（可视化）

```bash
./sim2sim/run_mujoco_onnx.sh
```

可视化模式下支持键盘实时改指令：
- `↑ / ↓`：增加/减小前向线速度
- `← / →`：增加/减小偏航角速度
- `Space`：线速度和角速度清零
- 方向键在 MuJoCo 窗口和启动终端里都可生效（已做双通道监听）

可选：通过环境变量调整时长（单位秒）：

```bash
SIM_TIME=60 ./sim2sim/run_mujoco_onnx.sh
```

## 4. 常用参数

```bash
./sim2sim/run_mujoco_onnx.sh --help
```

常用参数：
- `--sim-time`：仿真时长（秒）
- `--no-realtime`：关闭实时节流（加速跑）
- `--verbose-interval`：每 N 个策略步打印一次统计
- `--headless`：无窗口运行（适合远程自检）

## 5. 关键实现说明

- **观测**：按 Isaac 语义构造 `obs_prop[33]`，并维护 `obs_hist[10,33]` FIFO。
- **推理**：ONNX 两输入（`obs_prop` 和 `obs_hist`），输出 8 维动作。
- **动作映射**：策略输出按 `reindex([4,5,6,7,0,1,2,3])` 映射到 sim 关节顺序。
- **控制链**：
  - 低通：`a_f = 0.2 * last + 0.8 * current`
  - 缩放：`action_scale=0.35`，髋关节额外 `0.5`
  - 目标角：`target = scaled + default_dof_pos`
  - 力矩：`tau = kp*(target-q) - kd*dq`，轮关节使用特殊覆盖项
  - 子步：每个策略步执行 `decimation=4` 个 MuJoCo step
- **执行器冲突处理**：每个策略步会先中和 MJCF 内置 actuator（`ctrl` 跟踪当前状态），避免和 `qfrc_applied` 冲突导致倒地。

## 6. 排障

- 提示 `missing ONNX file`：
  - 将模型放到 `sim2sim/stairs_test.onnx`。
- 提示 `sensor 'gyro' not found`：
  - 检查 `resources/tita/mjcf/tita_description.xml` 是否包含 `gyro` 传感器。
- 无法弹出窗口：
  - 检查图形环境是否可用（远程机器通常需要 X11/桌面会话）。
- 机器人很快倒地：
  - 确认使用的是当前版本脚本（包含“执行器冲突处理”）；
  - 确认 ONNX 位于 `sim2sim/stairs_test.onnx` 且输入维度为 `obs_prop[33] + obs_hist[10,33]`。

## 7. PhysX-MuJoCo 数据采集

本仓已提供统一口径的数据采集流水线，会分别运行 PhysX 与 MuJoCo 并输出：
- 时间序列：`sim2sim/results/mujoco_vs_phyX/physx/*.csv`、`sim2sim/results/mujoco_vs_phyX/mujoco/*.csv`
- 汇总结果：`sim2sim/results/mujoco_vs_phyX/summary.json`、`sim2sim/results/mujoco_vs_phyX/compare.csv`

默认工况配置文件：
- `sim2sim/scenarios_default.json`
- PhysX 采集脚本（独立于 `simple_play.py`）：`sim2sim/run_physX_collection.py`
- PhysX 采集默认 ONNX：`sim2sim/stairs_test.onnx`

一键运行：

```bash
./sim2sim/run_compare_suite.sh
```

常用环境变量：

```bash
REPEATS=3 SEED=0 PHYSX_MODEL=tita_example_10000.pt ./sim2sim/run_compare_suite.sh
```

如果 PhysX 侧出现 `libpython3.8.so` 或 Isaac Gym 插件加载报错，先确认：
- 已激活训练时同一 conda 环境；
- `LD_LIBRARY_PATH` 包含 `${CONDA_PREFIX}/lib`（`run_compare_suite.sh` 已自动追加）。

也可直接调用 Python：

```bash
python sim2sim/experiment_suite.py --scenarios sim2sim/scenarios_default.json --repeats 5 --seed 0
```

当前统一采集字段：
- 元数据：`engine, scenario_id, repeat_id, seed, step, time_s`
- 指令：`cmd_vx, cmd_yaw`
- 状态：`base_z, base_roll, base_pitch, base_lin_vx, base_ang_yaw`
- 控制：`action_l2, action_delta_l2`
- 稳定性：`fallen_flag`

对 `action_delta_l2` 的可比性处理：
- MuJoCo 采集控制频率已对齐到 `0.01s`（与 PhysX 一致）；
- 汇总时默认忽略前 `1s` 瞬态（`--metrics-warmup-s` 可调）；
- `compare.csv` 额外提供 `action_delta_rate_mean_*`（`action_delta_l2 / dt`）与 `action_delta_rel_mean_*`（`action_delta_l2 / action_l2`）用于更稳健比较；
- `experiment_suite.py` 默认会清理当前输出目录下的 `physx` 与 `mujoco` 子目录旧结果（可用 `--no-clean-output` 关闭）。

## 8. 泛化性实验（flat + stairs + slope）

新增场景文件：
- `resources/tita/mjcf/flat_scene.xml`
- `resources/tita/mjcf/stairs_scene.xml`
- `resources/tita/mjcf/slope_scene.xml`

新增场景配置：
- `sim2sim/scenarios_generalization.json`

运行命令（仅 MuJoCo）：

```bash
python sim2sim/experiment_suite.py --scenarios sim2sim/scenarios_generalization.json --repeats 3 --skip-physx
```

默认输出目录：
- `sim2sim/results/terrain_contrast/`

## 9. 鲁棒性实验（动力学参数扰动）

鲁棒性脚本：
- `sim2sim/run_robustness_mujoco.py`

支持扰动等级：
- `--rand-level low|medium|high`

默认 `medium` 对应：
- 摩擦缩放：`0.8 ~ 1.2`
- 基座质量缩放：`0.9 ~ 1.1`
- 基座 COM 偏置：`±0.01m`（xyz）

运行命令：

```bash
python sim2sim/run_robustness_mujoco.py --scenarios sim2sim/scenarios_generalization.json --repeats 3 --rand-level medium
```

输出文件：
- 对比表：`sim2sim/results/robustness/robustness_compare.csv`
- 运行记录：`sim2sim/results/robustness/robustness_runs.json`

## 10. 总控脚本（推荐）

新增总控脚本：
- `sim2sim/run_all_experiments.py`

一键跑全套实验（compare + terrain + robustness）：

```bash
python sim2sim/run_all_experiments.py --suite all --repeats 3 --seed 0 --rand-level medium
```

只跑某一类实验：

```bash
python sim2sim/run_all_experiments.py --suite compare
python sim2sim/run_all_experiments.py --suite terrain
python sim2sim/run_all_experiments.py --suite robustness
```
