# v1.1 阶段验收记录

本文件仅记录 v1.1 的真实执行证据；v1.0 历史验收见 `docs/stage_acceptance.md`，二者不可互相覆盖。

## S6：工程重构（通过）

- 分支基线：`61cae0dfa89673e44b7d0e85ebb183392a2f7c14`；工作开始前 Git 工作区干净。
- `src/a1_mlp/` 已删除；包入口改为 `src.cli`，源码按 `nn/optim/data/training/utils` 分工。
- 配置中的 `hidden_activation`、`hidden_init`、`output_init` 已接入模型构造；未知或不支持的配置明确报错。
- float32 巨大且相同 logits 的 CE 回归用例返回 `log(2)`，避免原始坐标相减的精度丢失。
- `include_test=False` 从底层 loader 起不读取测试文件；测试集是否加载写入 manifest。
- Checkpoint v2 保存完整 resolved config、模型初始化定义和预处理统计；旧 v1 Checkpoint 仍可加载。
- 历史 B-M 验证重载：accuracy `0.9690`，CE `0.1142299739`；历史 B-C：accuracy `0.5150`，CE `1.4449032360`，与 v1.0 记录一致。
- pytest：40/40 通过；证据 `results/verification/v1_1/pytest.xml`。
- 五类损失、正则项和完整 MLP 的有限差分检查通过；证据 `results/verification/v1_1/gradient_check.json`。
- toy overfit：accuracy `1.0`；证据 `results/verification/v1_1/tiny_overfit.json`。
- MNIST/CIFAR-10 数据与固定划分校验通过；证据 `results/verification/v1_1/data_validation.json`。
- sdist 与 wheel 构建成功：`a1_numpy_mlp-1.1.0`。

## S7：扩展实验（通过）

- 真实 100-step benchmark：MNIST `52,257 samples/s`，CIFAR-10 `13,503 samples/s`；两者均远低于 45 分钟/运行治理线，因此未裁剪冻结矩阵。
- 冻结矩阵：3 个新损失、8 个新正则配置、6 个新优化器候选、4 个 CIFAR-10 扩展、10 个额外随机种子，共 31 次。
- 执行结果：31 completed / 0 failed；原始目录 `results/formal_v1_1/`，轻量证据 `results/evidence/v1_1/`，状态 `results/verification/v1_1/matrix_status.json`。
- 新损失验证最佳：Hinge 0.9746；新优化器验证最佳：Nesterov lr=0.05，0.9830；CIFAR 扩展验证最佳：Momentum 0.5216。
- Batch 历史核查保留原始口径：三组 samples seen 均为 50,000，但更新步数为 50,000/400/10，不能称为等更新或等时间预算。
- 三随机种子：MNIST 基准 `0.9682±0.00106`、Momentum `0.97887±0.00168`、Focal `0.96353±0.00160`、L2+Dropout `0.96547±0.00081`、CIFAR Adam `0.52533±0.00580`（样本标准差）。
- v1.1 测试选择在验证汇总后冻结；只评估 Hinge、Dropout 0.1、Nesterov 0.05、CIFAR Momentum 四个 Checkpoint，测试准确率分别为 0.9763、0.9705、0.9798、0.5063。

## S8：汇总、图表与最终验收（通过）

- 完整聚合：历史 16 + 新增 31 = 47 条；重复种子 10 条；`results/expanded/all_runs_v1_1.csv/json`。
- 图表：15 组 300 DPI PNG + 矢量 PDF，覆盖损失、正则、优化器、数据集、耗时/步数、重复误差棒；已进行 contact-sheet 视觉检查，中文字体、图例和标签正常。
- 安全性：聚合器在输入不完整时写前失败；只有显式 `--allow-partial` 才允许部分聚合；v1.0 `results/aggregate/` 从未被覆盖。
- 轻量复现：不含 `results/formal*` 与 Checkpoint 的临时目录成功重建全部 47 条汇总和图表。
- 完整复现：独立 Python 3.12 环境从 wheel 安装；40/40 测试、梯度检查、MNIST 基准训练/Checkpoint 重载、CIFAR-10 Adam 配置切换均通过。
- 最终审计：`results/verification/v1_1/final_audit_v1_1.json` passed；v1.0 历史 PDF SHA-256 未变化；没有生成 v1.1 最终报告 PDF。
- 论文事实材料：`docs/v1.1_实验结果与方法说明.md`。
