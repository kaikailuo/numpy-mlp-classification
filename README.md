# A1 NumPy MLP 图像分类 v1.1

本仓库是北京邮电大学《神经网络与深度学习（理论）》A1 作业的 NumPy-only 多层感知机实现。前向传播、反向传播、损失求导、正则化、优化器与评估均由 NumPy 实现，不依赖 PyTorch、TensorFlow 或 JAX。

v1.1 已完成源码重构、五类损失函数、七类优化器、正则化强度与组合、CIFAR-10 扩展以及五组配置的三随机种子复验。v1.0 的 16 组历史实验、历史 Checkpoint 与历史 PDF 均保留且未改写；v1.1 新增 31 次真实运行，其中 10 次为重复种子实验，成功 31、失败 0。

> **最终中文版 LaTeX 论文尚未制作，将在独立论文阶段完成。**
>
> `output/pdf/A1_NumPy_MLP_Report.pdf` 仅是 v1.0 历史参考，不是 v1.1 最终论文；`report/generate_report.py` 是 Legacy 工具，不属于 v1.1 标准交付流程。

## 关键结果

| 配置 | 验证准确率 | 冻结后测试准确率 | 测试 CE |
|---|---:|---:|---:|
| v1.0 MNIST 基准 B-M | 96.90% | 96.83% | 0.1056 |
| v1.0 CIFAR-10 基准 B-C | 51.50% | 50.39% | 1.4425 |
| v1.1 Hinge L-HINGE-1 | 97.46% | 97.63% | 0.1703 |
| v1.1 Dropout 0.1 R-DROP-0.1 | 96.88% | 97.05% | 0.0936 |
| v1.1 Nesterov O-NAG-0.05 | 98.30% | 97.98% | 0.0820 |
| v1.1 CIFAR Momentum C-MOM | 52.16% | 50.63% | 1.4544 |

测试集只在 `experiments/selection_v1_1.json` 固化上述四个 v1.1 Checkpoint 后执行一次。v1.0 的测试结果此前已知，因此可能影响了方法类别的宏观选择；这一限制已如实记录，但 v1.1 的候选学习率与最终 Checkpoint 没有根据新测试结果反向修改。

## 目录

```text
src/
├── cli.py, config.py
├── nn/          # layers, model, losses, regularization
├── optim/       # SGD, Momentum, AdaGrad, RMSProp, Adam, Nesterov, Adamax
├── data/        # MNIST/CIFAR loader, split, preprocessing
├── training/    # trainer, evaluation, metrics, checkpoints
└── utils/       # plotting, atomic I/O, reproducibility

configs/formal/          # v1.0 的 16 组不可变配置
configs/v1_1/            # v1.1 的 31 个预声明运行配置
results/baseline_v1/     # v1.0 可提交轻量证据，不含 Checkpoint
results/formal/          # v1.0 本地完整运行，Git 忽略
results/formal_v1_1/     # v1.1 本地完整运行，Git 忽略
results/evidence/v1_1/   # v1.1 可提交轻量证据
results/expanded/        # 47 次运行的表格与 PNG/PDF 图表
results/verification/v1_1/
docs/v1.1_实验结果与方法说明.md
```

仓库中不存在 `src/a1_mlp/`。统一模块入口为：

```python
from src.nn.model import MLP
from src.nn.losses import SoftmaxCrossEntropy
from src.optim.optimizers import Adam
```

命令行入口为 `uv run python -m src.cli`，安装后也可使用 `a1-mlp`。

## 环境安装与构建

要求 CPython 3.12，推荐 uv：

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv sync --locked --group dev
UV_CACHE_DIR=/tmp/a1-uv-cache uv build
```

执行测试、梯度检查和 toy overfit：

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run --group dev pytest -q
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python experiments/gradient_check.py
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python -m src.cli toy-overfit --output results/verification/v1_1/tiny_overfit-reproduced.json
```

## 数据下载与校验

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m src.cli prepare-data \
  --dataset mnist --root data/raw/mnist --download
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m src.cli prepare-data \
  --dataset cifar10 --root data/raw/cifar10 --download
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python experiments/verify_data.py \
  --output results/verification/v1_1/data_validation-reproduced.json
```

`verify_data.py` 的 `--output` 是必需参数。固定划分为 MNIST 55,000/5,000/10,000 与 CIFAR-10 45,000/5,000/10,000；CIFAR 标准化统计只从有效训练子集拟合并固化在 Checkpoint 中。训练和验证调用 `include_test=False`，底层不会读取官方测试文件。

## 单模型、扩展矩阵与评估

单独运行一个模型时必须选择新的输出目录，CLI 拒绝覆盖：

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python -m src.cli train \
  --config configs/formal/B-M.toml \
  --output results/reproduced/B-M
```

31 次 v1.1 冻结矩阵：

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python experiments/generate_v1_1_configs.py
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python experiments/run_v1_1_matrix.py
```

矩阵运行器保留已存在的目录，不会覆盖；失败摘要也会导出到 `results/evidence/v1_1/`。不要在验证选择冻结前使用 `--split test`。已冻结模型的显式评估示例：

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python -m src.cli evaluate \
  --checkpoint results/formal_v1_1/O-NAG-0.05/best.npz \
  --split validation \
  --output results/reproduced/O-NAG-0.05-validation.json
```

## 汇总、图表和轻量复现

```bash
MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache \
  uv run python experiments/aggregate.py
```

聚合器只读取 `results/baseline_v1/` 与 `results/evidence/v1_1/`，无需 Checkpoint 或重新训练。默认要求 16+31 条证据完整且全部成功；缺项时在写输出前安全失败。只有明确传入 `--allow-partial` 才生成标记为 partial 的汇总。写入先在临时目录完成，再逐文件原子替换，并且绝不覆盖 `results/aggregate/` 中的 v1.0 历史汇总。

主要论文素材位于 `results/expanded/`：

- `all_runs_v1_1.csv/json`：47 次运行及公平性字段；
- `repeat_statistics.csv/json`：五组配置的均值、样本标准差及原始种子结果；
- 损失、正则、优化器、数据集和重复实验图表：300 DPI PNG + 矢量 PDF；
- `selected_test_metrics_v1_1.csv/json`：冻结后的四项 v1.1 测试结果。

## 两条复现路径

轻量路径 A：只保留源码、配置、`results/baseline_v1/` 和 `results/evidence/v1_1/`，运行 `experiments/aggregate.py`。该路径已在不含 `results/formal*` 和任何 Checkpoint 的 `/tmp/a1-light-repro-20261011` 实际通过并重建 47 条汇总与全部图表。

完整路径 B：安装依赖、下载并校验数据、运行 pytest 与梯度检查、训练一个 MNIST 基准、重载 Checkpoint 验证、再用 `configs/v1_1/C-ADAM.toml` 切换到 CIFAR-10/Adam。该路径已在独立 Python 3.12 环境实际通过，证据见 `results/verification/v1_1/clean_environment_reproduction.json`。

## 打包与审计

```bash
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python scripts/audit_v1_1.py
UV_CACHE_DIR=/tmp/a1-uv-cache uv run python scripts/build_submission.py
```

v1.1 包含源码、配置、轻量证据、汇总 CSV/JSON、图表、测试与方法材料；不含原始数据和 Checkpoint。历史 PDF 会清楚标记为 v1.0 Legacy，不能作为 v1.1 最终论文提交。

## 已知边界

- 多种子结果是 3 次运行的描述性均值与样本标准差，不构成统计显著性证明。
- CIFAR-10 使用展平像素 MLP；约 50% 的准确率和明显 Train–Validation Gap 反映全连接模型在彩色图像空间结构上的限制，不能与 CNN 作架构等价比较。
- 不同损失的训练目标定义和尺度不同，只以统一的验证 Accuracy/CE 比较泛化表现。
- Batch 对照固定样本遍历量，但更新次数和计算时间不同，不能称为三种预算严格等价。
- 稳定 CE 改为在平移 logits 上计算后，float32 累积舍入使干净复现实验相对 v1.0 历史 B-M 验证准确率变化 0.06 个百分点；float64 固定前向/反向回归和历史 Checkpoint 重载均通过。
