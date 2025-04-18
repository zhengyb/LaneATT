# Trainning log

## 0. Tusimple r18官方预训练模型：
- 在tusimple vaild数据集上准确率:
 `{'Accuracy': 0.9453561452513968, 'FP': 0.07309124767225321, 'FN': 0.025372439478584728, 'FPS': 1000.0}；`
- 在tusimple test数据集上准确率:
` {'Accuracy': 0.9499888740542902, 'FP': 0.0973220704529109, 'FN': 0.03633477114785523, 'FPS': 1000.0}`
- 在2025 April 17混合后的test数据集上准确率:
`{'Accuracy': 0.7941927901220025, 'FP': 0.21595096226285823, 'FN': 0.322426546126977, 'FPS': 1000.0}`
- 在2025 April 17混合后的valid数据集上准确率0.63 (比test混了更多的数据)
` {'Accuracy': 0.6345005049328051, 'FP': 0.38824904839586644, 'FN': 0.6404839586731994, 'FPS': 1000.0}`
  
## 1. 2025 April 17
- 数据集： 1/10采样LLAMAS + TUSIMPLE，训练数据集共11542图片；
- anchors文件： data/tusimple_anchors_freq.pt
- epochs: 100
- 开始时间: April 17, 19:58
- 结束时间: 2025-04-18 10:44；
- 最优性能(Valid数据集): ![Metrics](./docs/img/tusimple_metrics_plot-20250417.png)
- 最有epoch模型在test数据集的表现：`
 {'Accuracy': 0.895974590781457, 'FP': 0.04631189644544894, 'FN': 0.16668378878158982, 'FPS': 1000.0}`
- 总结:
  - 经过重新训练后，在valid数据集上Accuracy提升明显，但是FP和FN没有下降，依旧很高。
  - 在test数据集上：Accuracy提升明显；FP显著下降，性能良好；FN显著下降，依旧很高。
  - 在整个训练过错中，Metric没有明显改善！！！
- 分析:
  - 超参数anchor文件，
  - 超参数maxlanes： tusimple数据集有5条线，llamas转换后只有4条线？