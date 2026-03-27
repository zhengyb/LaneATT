# LaneATT 超参数调优实验方案

## 背景

应用训练加速方案（增大 batch_size、DataLoader 优化等）后，需要验证新参数不损失模型精度。本文档描述一套低成本的实验方法，在数据集准备好后执行。

## 核心思路

- **不使用小数据集做超参搜索**：小数据集上的最优参数不能迁移到完整数据集（分布不同、容易过拟合）
- **使用完整数据集 + 短 epoch 对比**：只跑 15 epoch，通过 loss 下降趋势和早期 val F1 排除差参数
- **先验证 baseline 再做缩放**：原始 lr=0.0003 是在 TuSimple 小数据集（~3.6K）上调的，数据集扩大到 ~100K 后最优 lr 可能不同，需先搜索 bs=8 的最优 lr，再基于它做 batch size 缩放
- **分阶段推进**：阶段零找 lr -> 阶段一/二确定 bs=16 配置 -> 阶段三探索 bs=32
- **不使用 AMP**：全部使用 FP32 训练，避免自定义 NMS CUDA 扩展的 FP16 兼容性风险

## 指标定义

本方案所有判断基于以下两个指标，不使用主观描述。

### loss_avg(X, ep)

实验 X 在第 ep 个 epoch 的**平均 train loss**。对该 epoch 内所有 iteration 的 loss 取算术平均。

**数据来源**：训练日志 `experiments/<exp>/log_train.txt`，每 iteration 一行：

```
[2026-04-01 10:05:23,456] [lib.experiment] [DEBUG] Epoch [1/15] - Iter [0/12500] - Loss: 12.34567 - cls_loss: 0.65432 - reg_loss: 4.32109 - batch_positives: 125.00000 - lr: 0.00030
```

### F1(X, ep)

实验 X 在第 ep 个 epoch 的**test F1**。通过 `python main.py test --exp_name X --epoch ep` 获取。

**数据来源**：评估结果 `experiments/<exp>/results/epoch_XXXX/test_metrics.json`：

```json
{"F1": 0.5603, "Precision": 0.5966, "Recall": 0.5282, "TP": 18862, "FP": 12754, "FN": 16850}
```

## 数据记录格式

每组实验的完整结果存储为 JSON 文件 `experiments/<exp>/ablation_result.json`，由脚本自动生成。

### JSON 结构

```json
{
  "experiment": {
    "name": "ablation_bs16_linear",
    "group": "B",
    "phase": 1
  },
  "config": {
    "batch_size": 16,
    "lr": 0.0006,
    "warmup_epochs": 0,
    "epochs": 15,
    "T_max": 93750
  },
  "environment": {
    "iters_per_epoch": 6250,
    "peak_vram_mb": 6200
  },
  "training": {
    "status": "completed",
    "total_duration_sec": 2700,
    "epochs": [
      {
        "epoch": 1,
        "loss_avg": 8.12345,
        "cls_loss_avg": 0.54321,
        "reg_loss_avg": 3.21098,
        "lr_start": 0.0006,
        "lr_end": 0.000599,
        "num_iters": 6250,
        "duration_sec": 180,
        "has_nan": false
      }
    ]
  },
  "evaluation": {
    "15": {
      "test": {
        "F1": 0.72,
        "Precision": 0.75,
        "Recall": 0.69,
        "TP": 12345,
        "FP": 4567,
        "FN": 5678
      }
    }
  },
  "judgment": {
    "oscillation": false,
    "oscillation_detail": []
  }
}
```

### 字段说明

| 字段路径 | 类型 | 说明 |
|---------|------|------|
| `experiment.name` | string | 实验名称（experiments/ 下目录名） |
| `experiment.group` | string | 实验组编号：A1/A2/A3/B/C/D/E/F/G/H |
| `experiment.phase` | int | 所属阶段：0/1/2/3 |
| `config.batch_size` | int | batch size |
| `config.lr` | float | 学习率 |
| `config.warmup_epochs` | int | warmup epoch 数（0=无 warmup） |
| `config.epochs` | int | 总 epoch 数 |
| `config.T_max` | int | CosineAnnealingLR 的 T_max |
| `environment.iters_per_epoch` | int | 每 epoch 的 iteration 数（自动从日志提取） |
| `environment.peak_vram_mb` | int/null | GPU 显存峰值 MB（手动传入，可选） |
| `training.status` | string | `"completed"` / `"diverged"` (NaN/Inf) / `"no_log"` |
| `training.total_duration_sec` | int/null | 全部 epoch 的训练总耗时（秒） |
| `training.epochs[]` | array | 每个 epoch 的详细数据（见下表） |
| `evaluation.<ep>.<split>` | object | 第 ep 个 epoch 在 split (test/val) 上的评估指标 |
| `judgment.oscillation` | bool | epoch 1-5 是否存在 loss 环比上升 > 5% |
| `judgment.oscillation_detail[]` | array | 每次环比上升的具体 epoch 对和比值 |

**`training.epochs[]` 每条记录：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `epoch` | int | epoch 编号 |
| `loss_avg` | float/null | 该 epoch 平均 total loss（null 表示含 NaN） |
| `cls_loss_avg` | float/null | 该 epoch 平均分类 loss |
| `reg_loss_avg` | float/null | 该 epoch 平均回归 loss |
| `lr_start` | float | 该 epoch 第一个 iteration 的学习率 |
| `lr_end` | float | 该 epoch 最后一个 iteration 的学习率 |
| `num_iters` | int | 该 epoch 的 iteration 数 |
| `duration_sec` | int/null | 该 epoch 从第一条到最后一条日志的耗时（秒） |
| `has_nan` | bool | 该 epoch 是否出现 NaN/Inf |

### 工具脚本总览

`scripts/` 目录下有四个脚本，覆盖实验全流程：

| 脚本 | 用途 | 何时使用 |
|------|------|---------|
| `generate_configs.py` | 从基础配置自动生成全部 ablation YAML | 实验开始前 |
| `run_ablation.py` | 编排训练->评估->收集流程 | 执行实验时 |
| `collect_result.py` | 解析日志生成 `ablation_result.json` | 单次实验结束后 |
| `compare_results.py` | 对比实验结果并输出 verdict | 每阶段结束后 |

**`scripts/generate_configs.py`** — 自动生成配置文件：

```bash
# Phase 0 only (lr search):
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml

# All phases (after lr_base determined):
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml --lr-base 0.0003

# Skip dataset loading, provide N manually:
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml -N 100000 --lr-base 0.0003
```

读取基础配置，自动计算 T_max 和各组 lr，生成 `cfgs/ablation/ablation_*.yml`。
保留基础配置中的模型、数据集、增强等设置，只覆盖 bs/lr/epochs/scheduler 等实验参数。

**`scripts/run_ablation.py`** — 编排实验流程：

```bash
# Run a single experiment (train + test + collect)
python3 scripts/run_ablation.py run ablation_bs8_lr0003 --group A2 --phase 0

# Run all experiments in a phase
python3 scripts/run_ablation.py phase0
python3 scripts/run_ablation.py phase1 --lr-base 0.0003
python3 scripts/run_ablation.py phase2 --lr-base 0.0003
python3 scripts/run_ablation.py phase3 --lr-base 0.0003 --bs16-winner ablation_bs16_linear
```

每个 phase 命令顺序执行该阶段所有实验的 train->test->collect，完成后自动调用 compare_results.py 输出 verdict 和下一步建议。

**`scripts/collect_result.py`** — 收集单个实验结果：

```bash
# 训练 + 评估完成后执行
python3 scripts/collect_result.py <exp_name> --group <A1|B|...> --phase <0|1|2|3>

# 可选：手动传入显存峰值（从训练期间 nvidia-smi 读取）
python3 scripts/collect_result.py <exp_name> --group B --phase 1 --peak-vram-mb 6200
```

自动解析 `log_train.txt` 和 `results/` 目录，生成 `experiments/<exp>/ablation_result.json`。

**`scripts/compare_results.py`** — 对比多组实验：

```bash
# 阶段零：从 lr 候选中选最优（自动应用选择规则）
python3 scripts/compare_results.py --pick-best \
    ablation_bs8_lr0001 ablation_bs8_lr0003 ablation_bs8_lr001

# 阶段一/二/三：与 baseline 对比（自动应用判断标准，输出 verdict）
python3 scripts/compare_results.py --baseline ablation_bs8_lr0003 \
    ablation_bs16_linear ablation_bs16_sqrt

# 可选：保存对比结果为 JSON
python3 scripts/compare_results.py --baseline ablation_bs8_lr0003 \
    ablation_bs16_linear ablation_bs16_sqrt --save comparison_phase1.json
```

输出示例：

```
Baseline: ablation_bs8_lr0003
  loss_avg(ep15) = 2.34567
  F1(ep15)       = 0.7200

experiment                          bs         lr   loss_avg       F1  loss/bl  osc      verdict
----------------------------------------------------------------------------------------------------
ablation_bs8_lr0003                  8   0.000300    2.34567   0.7200    1.000    N     baseline
ablation_bs16_linear                16   0.000600    2.45678   0.7100    1.047    N       on_par
ablation_bs16_sqrt                  16   0.000424    2.89012   0.6500    1.232    N         slow

Verdicts: on_par=pass | oscillation=epoch1-5 loss spike >5% |
          slow=loss >1.3x baseline | marginal=between thresholds | diverged=NaN/Inf
```

`verdict` 列即为量化判断结果，直接用于决策流程。

## 判断标准

所有阶段使用统一的量化标准。以下 **baseline** 指当前阶段的对照组（阶段一/二/三的 baseline 是阶段零 winner）。

### 稳定性判断：是否震荡

**震荡**的定义：epoch 1-5 中存在**相邻两个 epoch 的 loss_avg 环比上升超过 5%**，即：

```
存在 n ∈ [1, 4]，使得 loss_avg(X, n+1) > loss_avg(X, n) × 1.05
```

满足此条件 -> 判定为"震荡"。

### 收敛性判断：是否与 baseline 持平

**持平**的定义（三个条件同时满足）：

1. loss_avg(X, 15) <= loss_avg(baseline, 15) × 1.20（最终 loss 不超过 baseline 的 120%）
2. F1(X, 15) >= F1(baseline, 15) × 0.95（F1 不低于 baseline 的 95%）
3. 不满足"震荡"条件

三条全部满足 -> 判定为"持平"。

### 偏慢判断

**偏慢**的定义：

```
loss_avg(X, 15) > loss_avg(baseline, 15) × 1.30
```

最终 loss 超过 baseline 的 130% -> 判定为"偏慢"。

### 严重异常

- 训练过程出现 **NaN 或 Inf** -> 判定为"发散"，直接排除该组。
- GPU 显存不足（**OOM**）-> 降 batch_size。

## 实验前准备

### 1. 确定数据集大小和 iterations

数据集准备好后，先计算基础数值：

```bash
# 在容器中执行
python -c "
from lib.config import Config
import math
cfg = Config('experiments/laneatt_r18_tusimple/config.yaml')
ds = cfg.get_dataset('train')
bs = cfg['batch_size']
iters = math.ceil(len(ds) / bs)
print(f'训练样本总数: {len(ds)}')
print(f'batch_size: {bs}')
print(f'iterations/epoch: {iters}')
print(f'--- T_max 参考值 ---')
for epochs in [15, 100]:
    for b in [8, 16, 32]:
        t = epochs * math.ceil(len(ds) / b)
        print(f'  epochs={epochs}, bs={b}: T_max={t}')
"
```

记录输出的 `N`（训练样本总数），后续所有 T_max 基于此计算。预计 N ~ 100K。

### 2. 配置文件模板

所有 ablation 配置放在 `cfgs/ablation/` 目录下。以下用 `N` 表示训练样本总数，实际使用时替换为真实值。

### 4. 显存预估（FP32，无 AMP）

| batch_size | 显存占用（估计） | 最低显存要求 |
|------------|---------------|------------|
| 8 | ~3-4 GB | 8 GB |
| 16 | ~6-8 GB | 10 GB |
| 32 | ~12-16 GB | 16 GB |

阶段三（bs=32）开始前须确认 GPU 显存充足，不足则跳过。

## 实验设计

### 实验总览

分四个阶段执行。阶段零确定 baseline lr，后续阶段基于它做 batch size 缩放。

以下用 **lr_base** 表示阶段零选出的最优 lr（预期在 0.0001 ~ 0.001 之间）。

#### 阶段零：bs=8 lr 搜索（必跑）

原始 lr=0.0003 是在 TuSimple 小数据集上调的。数据集扩大一个数量级后，最优 lr 可能偏移。本阶段在 bs=8 下搜索三个 lr，覆盖一个数量级范围，找到适合新数据集的 baseline。

| 编号 | 目的 | bs | lr | T_max | warmup |
|------|------|----|----|-------|--------|
| A1 | lr 偏低 | 8 | 0.0001 | 15 × ceil(N/8) | 无 |
| A2 | lr 原值 | 8 | 0.0003 | 15 × ceil(N/8) | 无 |
| A3 | lr 偏高 | 8 | 0.001 | 15 × ceil(N/8) | 无 |

A1、A2、A3 三组必跑。

**选择 lr_base 的规则**（按优先级）：

1. 排除所有发散的组（出现 NaN/Inf）
2. 排除所有震荡的组（epoch 1-5 存在环比上升 > 5%）
3. 在剩余组中，选 **loss_avg(X, 15) 最低**的。若两组 loss_avg(15) 差距 < 5%，选 lr 较大的（大 lr 在后期收敛更快）

**特殊情况**：
- 三组都震荡 -> 选 loss_avg(15) 最低的，后续阶段一的 B/C 组全部加 warmup
- 只剩一组 -> 直接使用

#### 阶段一：bs=16 缩放（必跑）

基于 lr_base 做 batch size 缩放。

| 编号 | 目的 | bs | lr | lr 缩放方式 | T_max | warmup |
|------|------|----|----|-----------|-------|--------|
| B | bs16 线性缩放 | 16 | lr_base x 2 | linear | 15 × ceil(N/16) | 无 |
| C | bs16 sqrt 缩放 | 16 | lr_base x sqrt(2) | sqrt | 15 × ceil(N/16) | 无 |

B、C 两组必跑。baseline 为阶段零 winner。

#### 阶段二：bs=16 修复组（条件触发）

| 编号 | 目的 | bs | lr | lr 缩放方式 | T_max | warmup |
|------|------|----|----|-----------|-------|--------|
| D | bs16 线性 + warmup | 16 | lr_base x 2 | linear | 15 × ceil(N/16) | 3 epoch |
| E | bs16 sqrt + warmup | 16 | lr_base x sqrt(2) | sqrt | 15 × ceil(N/16) | 3 epoch |

触发条件（基于量化判断标准）：
- B 被判定为"震荡" -> 跑 D
- C 被判定为"偏慢" -> 跑 E
- B、C 都不满足"持平" -> D、E 都跑

#### 阶段三：bs=32 探索（阶段一/二确认 bs=16 可用后）

| 编号 | 目的 | bs | lr | lr 缩放方式 | T_max | warmup |
|------|------|----|----|-----------|-------|--------|
| F | bs32 线性缩放 | 32 | lr_base x 4 | linear | 15 × ceil(N/32) | 无 |
| G | bs32 sqrt 缩放 | 32 | lr_base x 2 | sqrt | 15 × ceil(N/32) | 无 |
| H | bs32 线性 + warmup | 32 | lr_base x 4 | linear | 15 × ceil(N/32) | 5 epoch |

> bs=32 FP32 显存约 12-16 GB。需确认 GPU 显存 >= 16 GB，否则跳过阶段三。

**lr 缩放说明**：
- **线性缩放**（lr x k，k = new_bs / old_bs）：SGD 的经典法则，Adam 下可能偏激进
- **sqrt 缩放**（lr x sqrt(k)）：对 Adam 更稳健，理论依据是 Adam 的自适应学习率部分抵消了 batch size 的影响
- 注意：bs=32 sqrt 的 lr = lr_base x 2，恰好等于 bs=16 linear 的 lr，可直接参考 B 组的观察结论

### lr 速查表

阶段零确定 lr_base 后，查下表得出各组的具体 lr 值：

| lr_base | B (x2) | C (x1.414) | F (x4) | G (x2) |
|---------|--------|------------|--------|--------|
| 0.0001 | 0.0002 | 0.000141 | 0.0004 | 0.0002 |
| 0.0003 | 0.0006 | 0.000424 | 0.0012 | 0.0006 |
| 0.0005 | 0.001 | 0.000707 | 0.002 | 0.001 |
| 0.001 | 0.002 | 0.00141 | 0.004 | 0.002 |

D 的 lr 同 B，E 的 lr 同 C，H 的 lr 同 F。

### 配置文件示例

以下示例以 lr_base = 0.0003 为例。实际使用时根据阶段零结果替换。

**A1 -- bs8 lr=0.0001（`cfgs/ablation/ablation_bs8_lr0001.yml`）**：

```yaml
batch_size: 8
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: 0.0001
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/8)
```

**A2 -- bs8 lr=0.0003（`cfgs/ablation/ablation_bs8_lr0003.yml`）**：

```yaml
batch_size: 8
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: 0.0003
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/8)
```

**A3 -- bs8 lr=0.001（`cfgs/ablation/ablation_bs8_lr001.yml`）**：

```yaml
batch_size: 8
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: 0.001
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/8)
```

**B -- bs16 线性缩放（`cfgs/ablation/ablation_bs16_linear.yml`）**：

```yaml
batch_size: 16
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × 2
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/16)
```

**C -- bs16 sqrt 缩放（`cfgs/ablation/ablation_bs16_sqrt.yml`）**：

```yaml
batch_size: 16
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × sqrt(2)
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/16)
```

**D -- bs16 线性 + warmup（`cfgs/ablation/ablation_bs16_linear_warmup.yml`）**：

```yaml
batch_size: 16
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × 2
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/16)
# warmup: 前 3 epoch lr 从 0 线性增长到目标 lr
```

**E -- bs16 sqrt + warmup（`cfgs/ablation/ablation_bs16_sqrt_warmup.yml`）**：

```yaml
batch_size: 16
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × sqrt(2)
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/16)
# warmup: 前 3 epoch lr 从 0 线性增长到目标 lr
```

**F -- bs32 线性缩放（`cfgs/ablation/ablation_bs32_linear.yml`）**：

```yaml
batch_size: 32
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × 4
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/32)
```

**G -- bs32 sqrt 缩放（`cfgs/ablation/ablation_bs32_sqrt.yml`）**：

```yaml
batch_size: 32
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × 2
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/32)
```

**H -- bs32 线性 + warmup（`cfgs/ablation/ablation_bs32_linear_warmup.yml`）**：

```yaml
batch_size: 32
epochs: 15
val_every: 5
model_checkpoint_interval: 5
optimizer:
  name: Adam
  parameters:
    lr: __FILL__  # lr_base × 4
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: __FILL__  # 15 × ceil(N/32)
# warmup: 前 5 epoch lr 从 0 线性增长到目标 lr
# bs=32 比 bs=16 需要更长的 warmup 期
```

### Warmup 实现

D、E、H 组需要实现 warmup，通过 `SequentialLR` 组合 `LinearLR` + `CosineAnnealingLR`：

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_epochs = 3  # D/E 组用 3，H 组用 5
warmup_iters = warmup_epochs * iters_per_epoch

warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_iters)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max - warmup_iters)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                         milestones=[warmup_iters])
```

仅在需要 warmup 的组才实现此逻辑。

### 其余配置

模型参数、数据集参数、augmentation 等与正式训练配置完全一致。`generate_configs.py` 从基础配置自动继承这些参数，只覆盖 bs/lr/epochs/scheduler。

## 执行步骤

### 0. 生成配置文件

```bash
# Phase 0 configs (lr search)
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml

# Phase 0-3 configs (after lr_base determined, e.g. 0.0003)
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml --lr-base 0.0003
```

### 方式一：使用 run_ablation.py 自动编排（推荐）

```bash
# Phase 0: lr search (train + test + collect + compare, 自动执行)
python3 scripts/run_ablation.py phase0

# 查看输出的 lr_base，重新生成 Phase 1-3 configs
python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml --lr-base <lr_base>

# Phase 1: bs=16 scaling
python3 scripts/run_ablation.py phase1 --lr-base <lr_base>

# Phase 2: bs=16 + warmup (根据 Phase 1 verdict 决定是否需要)
python3 scripts/run_ablation.py phase2 --lr-base <lr_base>

# Phase 3: bs=32 (自动检测 bs16 winner，或手动指定)
python3 scripts/run_ablation.py phase3 --lr-base <lr_base>
python3 scripts/run_ablation.py phase3 --lr-base <lr_base> --bs16-winner ablation_bs16_linear
```

每个 phase 命令自动执行：顺序训练所有实验 -> 评估 -> 收集结果 -> 对比输出 verdict 和下一步建议。

### 方式二：手动逐步执行

如需更精细控制，可逐个运行：

```bash
# 单个实验: train + test + collect
python3 scripts/run_ablation.py run ablation_bs8_lr0003 --group A2 --phase 0

# 或完全手动
python main.py train --exp_name ablation_bs8_lr0003 --cfg cfgs/ablation/ablation_bs8_lr0003.yml
python main.py test --exp_name ablation_bs8_lr0003 --epoch 15
python3 scripts/collect_result.py ablation_bs8_lr0003 --group A2 --phase 0

# 对比
python3 scripts/compare_results.py --pick-best ablation_bs8_lr0001 ablation_bs8_lr0003 ablation_bs8_lr001
python3 scripts/compare_results.py --baseline <baseline> ablation_bs16_linear ablation_bs16_sqrt
```

### 耗时预估（N ~ 100K）

| 实验组 | 每 epoch 迭代数 | 15 epoch 预估耗时 |
|--------|---------------|-----------------|
| A1/A2/A3 (bs=8) | ~12,500 | ~80 min |
| B/C/D/E (bs=16) | ~6,250 | ~45 min |
| F/G/H (bs=32) | ~3,125 | ~25 min |

| 阶段 | 实验数 | 合计耗时 |
|------|-------|---------|
| 阶段零（A1+A2+A3） | 3 | ~240 min |
| 阶段一（B+C） | 2 | ~90 min |
| 阶段二（D+E，条件触发） | 0-2 | 0-90 min |
| 阶段三（F+G+H） | 0-3 | 0-75 min |
| **最坏情况全部跑完** | **10** | **~8 小时** |
| **顺利情况（零 + 一 + 三各 1-2 组）** | **5-6** | **~4-5 小时** |

## 决策流程

所有分支判断均基于"判断标准"中的量化定义。

```
阶段零：跑 A1、A2、A3，提取 loss_avg 和 F1，填入结果记录表
   |
   +-- 按 lr_base 选择规则（见阶段零说明）确定 lr_base
   +-- 查 lr 速查表，填入阶段一配置文件

阶段一：跑 B、C，以阶段零 winner 为 baseline 判断
   |
   +-- B 满足"持平" --> bs16_winner = B，进入阶段三
   |
   +-- B 不满足"持平"，C 满足"持平" --> bs16_winner = C，进入阶段三
   |
   +-- B 被判定"震荡"（epoch 1-5 环比上升 > 5%）
   |   --> 进入阶段二，跑 D
   |   +-- D 满足"持平" --> bs16_winner = D，进入阶段三
   |   +-- D 不满足"持平" --> 跑 E
   |       +-- E 满足"持平" --> bs16_winner = E，进入阶段三
   |       +-- E 不满足"持平" --> 保持 bs=8，或尝试 bs=12 折中
   |
   +-- B 被判定"偏慢"（loss_avg(15) > baseline × 1.3）
   |   --> 看 C 结果
   |   +-- C 满足"持平" --> bs16_winner = C，进入阶段三
   |   +-- C 也不满足"持平" --> 跑 E
   |       +-- E 满足"持平" --> bs16_winner = E，进入阶段三
   |       +-- E 不满足"持平" --> 保持 bs=8，或尝试 bs=12 折中
   |
   +-- B/C 显存 OOM --> 尝试 bs=12 折中

阶段三：以 bs16_winner 为 baseline 判断
   |
   |  bs16_winner 用 linear --> 先跑 F
   |  bs16_winner 用 sqrt   --> 先跑 G
   |
   +-- F/G 满足"持平" --> 采用 bs=32，进入正式训练
   |
   +-- F/G 被判定"震荡" --> 跑 H
   |   +-- H 满足"持平" --> 采用 bs=32 + warmup
   |   +-- H 不满足"持平" --> 退回 bs16_winner
   |
   +-- F/G 被判定"偏慢" --> 退回 bs16_winner
   |
   +-- 显存 OOM --> 退回 bs16_winner
```

### bs=12 折中方案

当 bs=16 全部不满足"持平"或 OOM 时，可尝试 bs=12 作为折中：

| bs | lr (linear, x1.5) | lr (sqrt, x1.225) |
|----|-------------------|-------------------|
| 12 | lr_base x 1.5 | lr_base x 1.225 |

bs=12 不在正式实验矩阵中，仅作为 fallback。配置从 B 或 C 组修改 batch_size 和 lr 即可。判断标准同上。

## 正式训练参数

确定最优参数后，计算完整训练的 T_max 并更新正式配置：

```yaml
# experiments/laneatt_r18_tusimple/config.yaml
batch_size: <选定值>      # 8 / 12 / 16 / 32
epochs: 100
optimizer:
  name: Adam
  parameters:
    lr: <选定值>          # 基于 lr_base 和选定的缩放方式
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: <100 × ceil(N / batch_size)>
```

以 N = 100,000 为例的参考值：

| batch_size | iterations/epoch | T_max (100 epoch) |
|------------|-----------------|-------------------|
| 8 | 12,500 | 1,250,000 |
| 12 | 8,334 | 833,400 |
| 16 | 6,250 | 625,000 |
| 32 | 3,125 | 312,500 |

> 若 warmup 组胜出，正式训练也需保留 warmup。此时 CosineAnnealing 的 T_max = (epochs - warmup_epochs) x ceil(N / batch_size)。
