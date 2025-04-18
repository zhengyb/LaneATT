# Trainning log

本文记录LaneATT的调优过程。

## 0. Tusimple r18官方预训练模型：
- 在tusimple vaild数据集上准确率:
 `{'Accuracy': 0.9453561452513968, 'FP': 0.07309124767225321, 'FN': 0.025372439478584728, 'FPS': 1000.0}；`
- 在tusimple test数据集上准确率:
` {'Accuracy': 0.9499888740542902, 'FP': 0.0973220704529109, 'FN': 0.03633477114785523, 'FPS': 1000.0}`
- 在2025 April 17混合后的test数据集上准确率:
`{'Accuracy': 0.7941927901220025, 'FP': 0.21595096226285823, 'FN': 0.322426546126977, 'FPS': 1000.0}`
- 在2025 April 17混合后的valid数据集上准确率0.63 (比test混了更多的llamas数据)
` {'Accuracy': 0.6345005049328051, 'FP': 0.38824904839586644, 'FN': 0.6404839586731994, 'FPS': 1000.0}`
- **可以看到，预训练模型的泛化效果并不好。误报和漏报率都很高。**
  
## 1. 2025 April 17
- 数据集： ==1/10采样LLAMAS + 1/1 采样TUSIMPLE，train数据集共11542图片；==
- anchors文件： data/tusimple_anchors_freq.pt
- epochs: 100
- 开始时间: April 17, 19:58
- 结束时间: 2025-04-18 10:44；
- 最优性能(Valid数据集):  ![Metrics](./img/tusimple_metrics_plot-20250417.png)
- 最优epoch模型在test数据集的表现：
`{'Accuracy': 0.895974590781457, 'FP': 0.04631189644544894, 'FN': 0.16668378878158982, 'FPS': 1000.0}`
- **总结**:
  - 经过重新训练后，在valid数据集上Accuracy提升明显，但是FP和FN没有下降，依旧很高。
  - 在test数据集上：**Accuracy提升明显；FP显著下降，性能良好；FN显著下降，依旧很高。**
  - 在整个训练过程中，Metric没有明显改善！！！
- **分析**:
  - 超参数anchor文件，
  - 超参数maxlanes： tusimple数据集有5条线，llamas转换后只有4条线？
  - 官方Tusimple训练配置文件把val数据集加入了train数据集，==存在泄题问题==。该做法也影响到anchors mask的生成过程。
  - label_llamas_images-2014-12-22-14-19-07_mapping_280S_3rd_lane.json在train和test都出现了，==存在泄题问题==。


## 2. 2025 April 18 AM
- 数据集： 同上，1/10采样LLAMAS + TUSIMPLE，训练数据集共11542图片；
- anchors文件：==**data/tusimple_250418_anchors_mask.pt**==, 生成方法参考文档[debug.md](./debug.md)
- epochs: ==20==
- 开始时间: 2025-04-18 15:20（容器内时间）
- 结束时间：约40分钟；
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250418AM.png)
- 最优epoch模型在test数据集的表现：
  ` {'Accuracy': 0.9412176392027897, 'FP': 0.1777378261762927, 'FN': 0.07309430860899933, 'FPS': 1000.0}`
- 分析与总结：
  - ==可以看出“好的”anchors mask可以大幅度改善误报和漏报， 对Accuracy也有明显提升==；
  - ==在val和在test数据集上FP差距明显==， FP泛化性能似乎不好。==事后发现：test时没有使用更新后的anchors mask!!!==

## 3. 2025 April 18 PM
- 数据集： 基于上述融合数据集，作出调整，避免明显泄题。
  - ==**把val数据集从training数据集中分离出来，防止过拟合**==
  - ==**把label_llamas_images-2014-12-22-14-19-07_mapping_280S_3rd_lane.json从train中剔除**==
  - 最终train数据集包含**7395** images，max lanes为5；
  - val数据集3065 images, max lanes 4.
- anchors文件：同上，**基于train+val数据集生成**。
- epochs: 20
- 开始时间: 2025-04-18 16:14（容器内时间）
- 结束时间： 2025-04-18 16:50
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250418PM.png)
- 最优epoch模型在test数据集的表现(**从这里开始使用训练时相同的anchors mask**)：
  `{'Accuracy': 0.9553819087733689, 'FP': 0.09751729333607163, 'FN': 0.04123005273611405, 'FPS': 1000.0}`
- **分析与总结**:
  - ==**总体表现良好！**== 是否可以把val从anchors中剔除？？
  - 在自有数据上的表现: ==总体效果尚可，路口转弯，进出匝道较差==。

## 4. 2025 April 18 PM2
- 数据集： 同上。
- anchors文件：**仅基于train数据集生成**。`data/tusimple_250418pm_anchors_mask.pt`
- epochs: 20
- 开始时间: 2025-04-18 18:55（容器内时间）
- 结束时间： 2025-04-18 19:20，25分钟左右完成20个epochs训练。
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250418PM2.png)
- 最优epoch模型在test数据集的表现：
  `{'Accuracy': 0.9539399820951597, 'FP': 0.08173755222244995, 'FN': 0.042103280597219496, 'FPS': 1000.0}`
- **分析与总结**:
  - 从可视化anchors mask图中可以看到，与上一实验所使用的anchors mask有细微差异，不明显。因为train和val数据集都包含Llamas数据，数据分布相当。模型预期结果应该相当。
  - 总体性能差不多。

## 5. 2025 April 18 PM3
- 数据集： 同上。
- anchors文件： ==不给定anchors mask文件！！==
- epochs: 20
- 开始时间: 2025-04-18 19:27
- 结束时间： 2025-04-18 20:56
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250418PM3.png)
- 最优epoch（15）模型在test数据集的表现：
  ` {'Accuracy': 0.9509644031230693, 'FP': 0.07113211423875045, 'FN': 0.05384905143483343, 'FPS': 1000.0}`
- 最优epoch（15）模型在valid数据集的表现：
  ` {'Accuracy': 0.957186067738672, 'FP': 0.07412724306688434, 'FN': 0.05489396411092995, 'FPS': 1000.0}`
- **分析与总结**:
  - test与valid数据集表现一样；均达到较好效果；
  - 速率？==训练时间接近1小时30分钟，正好与anchors数量成正比。使用utils/speed.py测试pytorch推理速率，FPS降低了约50%==
    - 当前速率测试结果：
    ```
    MACs: 14.298G
    Params: 13.277M
    Average latency (ms): 2.14
    Average FPS: 467.30
    ```
    - 上一个模型速率测试结果：
    ```
    MACs: 9.358G
    Params: 12.019M
    Average latency (ms): 1.17
    Average FPS: 853.69
    ```