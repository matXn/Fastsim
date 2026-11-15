# Fastsim

已知赛道地图下的单圈仿真、路线规划与控制器训练项目。

车辆第一圈建图后，第二圈可以利用完整边界规划路线和速度。Fastsim 提供比赛 A、B、C、D 四套赛道、车辆动力学、车身碰撞检测、扰动评测，以及纯追踪和神经网络参考控制器，支持在同一套仿真判定下迭代控制方法。

![四条赛道](docs/tracks.png)

## 安装

使用 Python 3.10 或更新版本，在仓库根目录运行。

```bash
python -m pip install -r requirements.txt
```

核心仿真与训练使用 NumPy，轨迹绘图使用 Matplotlib，测试使用 pytest。

## 快速开始

```bash
# 1 m/s 中心线纯追踪基线，运行 A–D 四条赛道
python grade.py

# 已保存的神经控制器，运行原版和每条三个扰动版本
python grade.py --module controllers.neural --perturb 3

# 绘制速度着色轨迹，输出到 out/
python grade.py --module controllers.neural --plot

# 完整测试
python -m pytest -q
```

`--tracks A B` 可选择赛道，`--module` 指定提供 `Controller` 类的 Python 模块。`grade.py --perturb 3` 使用种子 1、2、3，并同时加入地图、车辆参数和观测扰动。

## 目录

| 路径 | 内容 |
|---|---|
| `fastsim/` | 车辆模型、赛道、碰撞、圈速判定和绘图 |
| `fastsim/tracks/` | A–D 原始地图 |
| `controllers/baseline.py` | 中心线、1 m/s 纯追踪基线 |
| `controllers/planner.py` | 中心线路线与曲率速度表 |
| `controllers/neural.py` | 21 → 64 → 64 → 2 神经跟踪器与训练方法 |
| `controllers/model.npz` | 已训练的参考权重 |
| `training/train.py` | 行为克隆、DAgger、课程训练与独立测试 |
| `grade.py` | 控制器批量评测和轨迹出图 |
| `accept.py` | 控制器包的独立进程评测 |
| `tests/` | 仿真、评测、模型和训练回归测试 |
| `runs/`、`out/` | 本地运行产物，由 Git 忽略 |

## 控制器接口

```python
class Controller:
    def __init__(self, track, car_params):
        # 初始化时取得完整地图和标称车辆参数。
        ...

    def __call__(self, state):
        # 每 0.02 秒仿真时间调用一次。
        return v_cmd, steer_cmd
```

`v_cmd` 单位为 m/s，`steer_cmd` 为前轮转角，单位 rad，左转为正。返回值需为两个有限数值。

### 地图与参数

| 字段或方法 | 含义 |
|---|---|
| `track.left`、`track.right` | 内、外边界闭合折线 |
| `track.centerline` | 约 5 cm 间距的闭合中心线，首点为起点 |
| `track.bounds()` | 中心线法向与到两侧边界的最近距离 |
| `track.on_track(points)` | 批量查询点是否位于赛道内 |
| `track.project(xy)` | 投影到中心线，返回弧长与横向偏移 |
| `track.length` | 一圈中心线长度 |
| `car_params` | 标称车辆参数，定义于 `fastsim/params.py` |

### 状态

| 字段 | 含义 |
|---|---|
| `t` | 当前圈仿真时间，s |
| `x`、`y` | 地图坐标下的后轴中心位置，m |
| `yaw` | 车头朝向，rad |
| `v` | 车速，m/s |
| `yaw_rate` | 横摆角速度，rad/s |
| `accel` | 纵向加速度，m/s² |
| `steer` | 当前实际前轮转角，rad |

原版赛道使用精确状态。扰动评测对状态加入观测噪声，碰撞、进度与轨迹记录使用仿真真值。控制器接收当前地图绝对位姿，车辆响应包含速度惯性和转向速率限制。

## 仿真与评测

参数从 `fastsim/params.py` 的 `CarParams` 读取。当前标称极速 6 m/s，总抓地 0.4 G，驱动加速上限 0.3 G，制动上限 0.4 G，纵横向运动共享摩擦圆。尺寸、公式和实现假设见参数文档。[车辆参数与物理模型](docs/车辆参数.md)

- 从起跑位置静止出发，累计前进一圈，冲线时间按步内进度插值。
- 使用 0.56 × 0.35 m 矩形车身检测边界接触，并按车身运动量以约 1 cm 间隔采样步内位姿。
- 单圈时间上限为 120 s。
- 扰动地图包含边界折点 ±3 cm 抖动及整体旋转、平移。
- 车辆摩擦系数浮动 ±10%，速度响应时间常数、转角速率浮动 ±20%。控制器使用标称车辆参数。
- 观测噪声标准差分别为位置 2 cm、朝向 0.01 rad、车速 0.02 m/s、横摆角速度 0.02 rad/s、纵向加速度 0.1 m/s²、转角 0.005 rad。

训练和自定义评测可复用 `run_lap()`，或使用 `Car`、`car_off_track()`、`Progress` 构建闭环。使用 `noise_seed` 可复现车辆与观测扰动。

## 神经控制器与训练

参考方法先由完整地图生成中心线路线和曲率速度表，再将前方 8 个路线点的车体坐标、车辆状态与参考速度编码为 21 维输入。两层 64 单元 ReLU 网络输出目标速度和转角。纯追踪控制器提供监督标签，训练采用 NumPy Adam、行为克隆和 DAgger。

```bash
# 检查采样、损失下降、梯度与模型读写
python -m training.train --smoke

# 12 轮课程训练，每轮 500 次更新
python -m training.train

# 复测入库模型，使用 A–D 原版及种子 90001、90002、90003
python -m training.train --evaluate

# 使用指定模型和新的输出目录
python -m training.train --evaluate --model controllers/model.npz --out runs/reference-check
```

课程限速为 1.5、2.5、4、6 m/s。训练种子从 10000 起，选择集使用每条原版及种子 50001、50002，连续两轮 12/12 完赛后晋级。最终评测使用每条原版及种子 90001–90003，共 16 局。

每次运行在 `runs/` 下创建独立目录，保存参数配置、候选、训练进度、评测明细和汇总。`--out` 可指定新目录。训练结束且独立测试 16/16 完赛后，导出该运行目录内的 `model.npz`，用于后续复核和选择。

### 保存模型的来源与复测

入库模型来自本地历史训练 `20260929_235117/best_cap_6.npz`，训练为 12 轮、每轮 500 次更新，模型 schema 为 `known_map_tracking_v2_lat03`，限速 6 m/s。模型文件原样保留，SHA-256 为 `122d45a07d1c9257de8199005d06fd82dcdaead390068ad829af54cfc29752c9`。

历史训练配置为独立侧向 0.3 G、纵向 0.4 G，原测试记录 16/16 完赛。本次在当前统一摩擦圆仿真配置下重新评测，使用仓库基线 `e2a02d0` 的车辆模型，日期为 2026-11-15。A–D 原版加每条种子 90001、90002、90003，共 16 局，控制器限速 6 m/s，结果为 **16/16 完赛**。

| 赛道 | 原版圈速 / s | 本组完赛 |
|---|---:|---:|
| A | 12.579 | 4/4 |
| B | 12.572 | 4/4 |
| C | 13.130 | 4/4 |
| D | 12.751 | 4/4 |

上述圈速属于当前 Fastsim 配置下的仿真结果，实车适用性待验证。通过 `python -m training.train --evaluate` 可重新生成逐局结果和参数快照。

## 兼容入口

`homework.controller` 保留为 `controllers.baseline` 的兼容导入路径。`accept.py` 支持含 `controller.py` 的目录或压缩包，将控制器放入独立进程执行。

```bash
python accept.py homework --perturb 3 --seed 90001 --csv out.csv
```

隔离评测的初始化上限为 60 s，控制器每步平均计算上限为 20 ms，单步连同进程通信上限为 1 s。
