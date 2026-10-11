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

## S7：扩展实验

待真实运行后填写。

## S8：汇总、图表与最终验收

待真实运行后填写。
