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
  
**从这里开始混入LLAMAS数据**

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
- anchors文件： ==不给定anchors mask文件！！使用全部2876个anchors进行训练和推理。==
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


## 6. 2025 April 21 AM1
- 数据集： 同上。
- anchors文件： 仅基于train数据集生成。`data/tusimple_250418pm_anchors_mask.pt`
- epochs: 20
- 训练参数修改：lr_scheduler.T_max从100 * 454改为20 * 925
- 开始时间: 2025-04-21 14:22
- 结束时间： 2025-04-21 14:47
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250421am1.png)
- 最优epoch（19）模型在test数据集的表现：
  `{'Accuracy': 0.9565514103730626, 'FP': 0.0803883295664674, 'FN': 0.04018560372577228, 'FPS': 1000.0}`
- 最优epoch（19）模型在valid数据集的表现：
  ` {'Accuracy': 0.9611439641109293, 'FP': 0.06129418162044602, 'FN': 0.051169113648722155, 'FPS': 1000.0}`
- **分析与总结**:
  - 最优模型性能相当。

## 7. 2025 April 21 AM2
- 数据集: culane
- anchors文件： 官方anchors文件
- laneatt_r18_culane官方预训练模型在test数据集上的性能：
`{'TP': 72067, 'FP': 15685, 'FN': 32819, 'Precision': 0.8212576351536147, 'Recall': 0.6870983734721507, 'F1': 0.7482116716328034}`
- 上一个实验生成的epoch 19模型在culane test数据集上的性能(使用culane anchors)：
`{'TP': 1284, 'FP': 1374, 'FN': 103602, 'Precision': 0.48306997742663654, 'Recall': 0.01224186259367313, 'F1': 0.02387859852711448}`
- 上一个实验生成的epoch 19模型 + 'data/tusimple_250418pm_anchors_mask.pt'在culane test数据集上的性能:
` {'TP': 7448, 'FP': 5717, 'FN': 97438, 'Precision': 0.5657424990505128, 'Recall': 0.07101043037202295, 'F1': 0.126182751522647}`
- laneatt_r18_culane官方预训练模型+ 'data/tusimple_250418pm_anchors_mask.pt',在混合后的tusimple-test数据集上的性能：
` {'Accuracy': 0.732033703660219, 'FP': 0.736877611122531, 'FN': 0.6548866515992149, 'FPS': 1000.0}`
- laneatt_r18_culane官方预训练模型+ 'data/culane_anchors_freq.pt',在混合后的tusimple-test数据集上的性能：
` {'Accuracy': 0.7230097375913576, 'FP': 0.6622320389014393, 'FN': 0.6208136429011787, 'FPS': 1000.0}`
- **分析：**
  - culane与tusimple相互之间的泛化性能都很差；tusimple是北美高速路场景；culane是北京城区场景；
  - 暂时不混入culane数据集，需要再仔细思考。


**从这里开始，改用F1 Score评价指标**

## 8. 2025 April 23 PM1
- 数据集： 同上。
- anchors文件： `data/tusimple_250418pm_anchors_mask.pt`
- epochs: 20
- 训练参数修改：lr_scheduler.T_max为20 * 925. 改用F1 Score度量性能
- 开始时间: 2025-04-23 16:26
- 结束时间： TODO
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250423pm1.png)
- 最优epoch（TODO）模型在test数据集的表现：
  `TODO`
- 最优epoch（TODO）模型在valid数据集的表现：
  ` TODO`
- **分析与总结**:
  - 采用F1评估，epoch20未收敛；


## 9. 2025 April 23 PM2
- 数据集： 同上（**无openlane数据**）。
- anchors文件： `data/tusimple_250418pm_anchors_mask.pt`
- epochs: 40
- 训练参数修改：lr_scheduler.T_max为40 * 925. 改用F1 Score度量性能
- 开始时间: TODO
- 结束时间： TODO
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250423PM2.png)
- 最优epoch（35）模型在test数据集的表现：
  ` {'F1': 0.9316309954686653, 'Precision': 0.9222479264232918, 'Recall': 0.9412069561211562, 'FPS': 1000.0, 'TP': 16345, 'FP': 1378, 'FN': 1021}`
- 最优epoch（35）模型在混合了1/5 openlane的valid数据集的表现：  
`{'F1': 0.5653014197576749, 'Precision': 0.4780914062991136, 'Recall': 0.6914264933175743, 'FPS': 1000.0, 'TP': 15210, 'FP': 16604, 'FN': 6788}`
- **分析与总结**:
  - F1指标在训练过程中会波动；
  - 最后10个epochs性能基本稳定；
  - 在openlane上泛化性能不好；参考性能：[paperswithcode](https://paperswithcode.com/sota/lane-detection-on-openlane)上记录的2D最佳模型F1为63,3D模型为66.


**从这里开始混入Openlane v1.x数据，1/5采样率**

## 10. 2025 April 23 PM3
- 数据集： 1/5混合了openlane的training和validation数据集。数据集信息参考[openlanev1_dataset.md](openlanev1_dataset.md)的`20250423`记录。混合后，training数据集有39225, valid数据集有7221, test数据集有8779.
- anchors文件： 同上。
- epochs: 20
- 训练参数修改：lr_scheduler.T_max为20 * 4904. 改用F1 Score度量性能
- 开始时间: 2025-04-23 18:20
- 结束时间： 2025-04-23 20:08
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250423PM3.png) **epoch 40是垃圾数据**
- 最优epoch（15）模型在test数据集(only tusimple + llamas)的表现：
  `{'F1': 0.9242215328047798, 'Precision': 0.9221024876762581, 'Recall': 0.926350339744328, 'FPS': 1000.0, 'TP': 16087, 'FP': 1359, 'FN': 1279} `
- 最优epoch（15）模型在混合了1/10 openlane的valid数据集的表现：  
`{'F1': 0.7929401077152682, 'Precision': 0.7563441303899319, 'Recall': 0.8332575688698972, 'FPS': 1000.0, 'TP': 18330, 'FP': 5905, 'FN': 3668}`
- 将validation数据集拆分一半混入test数据集：
`{'F1': 0.8637516914749663, 'Precision': 0.8445032455451276, 'Recall': 0.8838980483794193, 'FPS': 1000.0, 'TP': 20426, 'FP': 3761, 'FN': 2683}`
- 在拆分后的validation数据集的性能:
`{'F1': 0.8291208628403806, 'Precision': 0.7997599176860638, 'Recall': 0.8607197785296832, 'FPS': 1000.0, 'TP': 13991, 'FP': 3503, 'FN': 2264}`
- **分析与总结**:
  - 综合训练后的模型在tusimple+llamas的test数据集上依然有超过92的F1分数；


## 11. 2025 April 23 PM4, 更新anchors后重新训练，观察Metrics变化；
- 数据集： 同上。
- anchors文件： data/tusimple_250423_anchors_mask.pt
- epochs: 20
- 训练参数修改：lr_scheduler.T_max为20 * 4904. 改用F1 Score度量性能
- 开始时间: 2025-04-24 00:52
- 结束时间： 2025-04-24 02:35
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250423PM4.png) 
- 最优epoch（15）模型在test数据集(only tusimple + llamas)的表现：
  `{'F1': 0.9216704878544272, 'Precision': 0.9290854835878533, 'Recall': 0.9143729125878153, 'FPS': 1000.0, 'TP': 15879, 'FP': 1212, 'FN': 1487}`
- 最新test数据集：
`{'F1': 0.8533265595563176, 'Precision': 0.8352395159953588, 'Recall': 0.8722142888052274, 'FPS': 1000.0, 'TP': 20156, 'FP': 3976, 'FN': 2953}`
- valid数据集：
`{'F1': 0.8217295551314051, 'Precision': 0.7860674157303371, 'Recall': 0.8607812980621348, 'FPS': 1000.0, 'TP': 13992, 'FP': 3808, 'FN': 2263}`
- **分析与总结**:
  - 从anchors mask的可视图上可以看到，底部anchors与前一个数据集有明显区别。底部中间的anchors被过滤掉，是否因为openlane的标注问题去除了变道过程的数据？？！！**这样训练的模型对于变道过程的识别准确率可能有影响！！！**
  - Metrics并没有明显变化。混合了openlane数据后，变道数据较少，在Metrics中影响较小。

## 12. 用变更后的vaild数据集复现实验#10， 20250423PM5
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250423PM4.png) 
- 最优epoch（15）模型在test数据集(only tusimple + llamas)的表现：
  `{'F1': 0.9211286126754523, 'Precision': 0.9146182231053079, 'Recall': 0.9277323505700794, 'FPS': 1000.0, 'TP': 16111, 'FP': 1504, 'FN': 1255}`
- 最新test数据集：
` {'F1': 0.8609831513851203, 'Precision': 0.837671905697446, 'Recall': 0.885628975723744, 'FPS': 1000.0, 'TP': 20466, 'FP': 3966, 'FN': 2643}`
- valid数据集：
` {'F1': 0.8233634145624505, 'Precision': 0.7883922539968475, 'Recall': 0.8615810519840049, 'FPS': 1000.0, 'TP': 14005, 'FP': 3759, 'FN': 2250}`
- **分析与总结**:
  - 与实验11的结果进行对比分析，把openlane数据集加入anchors mask的统计范围内，并不能提升F1. 根据理论分析，对变道场景可能有不利影响。因此保持data/tusimple_250418pm_anchors_mask.pt.
  - 

## 13. Apollo syntheics合成数据研究
- “脑补”太严重，而且无法区分脑补的原因；
- Merge和Fork车道标注似乎有问题，不好预处理；

## 14. 2025 April 24 PM1, 重新按照1/2采样openlane数据集；
- 数据集： 重新按照1/2采样openlane数据集并混入TUSimple + LLAMAS。 Train 86505 images; Valid 13277 images; Test 16695 images
- anchors文件： data/tusimple_250423_anchors_mask.pt
- epochs: 40
- 训练参数修改：
- 开始时间: 2025-04-25 21:05
- 结束时间： 2025-04-26 04:42
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250425PM1.png) 
- 最优epoch（20）模型在test数据集(only tusimple + llamas)的表现：
  `{'F1': 0.9126219065336265, 'Precision': 0.893827296819788, 'Recall': 0.9322238857537717, 'FPS': 1000.0, 'TP': 16189, 'FP': 1923, 'FN': 1177}`
- 最新test数据集：
  `{'F1': 0.792571493034012, 'Precision': 0.7467846513605442, 'Recall': 0.8443396226415094, 'FPS': 1000.0, 'TP': 28103, 'FP': 9529, 'FN': 5181}`
- valid数据集：
  `{'F1': 0.768491868698462, 'Precision': 0.7124942701853185, 'Recall': 0.8340423900962017, 'FPS': 1000.0, 'TP': 21761, 'FP': 8781, 'FN': 4330}`
- **分析与总结**:
  - 高速场景的test性能略有下降。综合场景也下降。


## 15. 2025 April 26 AM1, 提高epoch和T max；
- 数据集： 同上
- anchors文件： 同上。
- epochs: 240. T_max: 865120  # 80 * 10814, 20250426AM1
- 训练参数修改：
- 训练时长： ==**约48小时**==
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250426AM1.png) 
- 最优epoch（229）模型在test数据集(only tusimple + llamas)的表现：
  `{'F1': 0.9329367689535849, 'Precision': 0.9267087097324016, 'Recall': 0.9392491074513417, 'FPS': 1000.0, 'TP': 16311, 'FP': 1290, 'FN': 1055}`
- 最新test数据集：
  `{'F1': 0.8330017243666268, 'Precision': 0.8175427430786588, 'Recall': 0.8490566037735849, 'FPS': 1000.0, 'TP': 28260, 'FP': 6307, 'FN': 5024}`
- valid数据集：
  `{'F1': 0.8138870741950537, 'Precision': 0.7932615339834085, 'Recall': 0.8356138131922886, 'FPS': 1000.0, 'TP': 21802, 'FP': 5682, 'FN': 4289}`
- 分场景test性能(test是上述混合数据集;高速场景是tusimple+llamas;其他场景来自openlane的test):
```JSON
{
    "test": {
        "F1": 0.8329894476212933,
        "Precision": 0.8175190928025919,
        "Recall": 0.8490566037735849,
        "FPS": 1000.0,
        "TP": 28260,
        "FP": 6308,
        "FN": 5024
    },
    "test_curve_case": {
        "F1": 0.7558838341447037,
        "Precision": 0.7476753453772582,
        "Recall": 0.7642745603910652,
        "FPS": 1000.0,
        "TP": 11257,
        "FP": 3799,
        "FN": 3472
    },
    "test_extreme_weather_case": {
        "F1": 0.6124369172868018,
        "Precision": 0.6099846651899813,
        "Recall": 0.6149089659910684,
        "FPS": 1000.0,
        "TP": 3580,
        "FP": 2289,
        "FN": 2242
    },
    "test_night_case": {
        "F1": 0.6326634259595154,
        "Precision": 0.5790941692123788,
        "Recall": 0.6971538215542054,
        "FPS": 1000.0,
        "TP": 4360,
        "FP": 3169,
        "FN": 1894
    },
    "test_intersection_case": {
        "F1": 0.5701706772812254,
        "Precision": 0.5334612346236316,
        "Recall": 0.6123056994818653,
        "FPS": 1000.0,
        "TP": 9454,
        "FP": 8268,
        "FN": 5986
    },
    "test_up_down_case": {
        "F1": 0.6828587381351201,
        "Precision": 0.6571735626007523,
        "Recall": 0.7106333527019175,
        "FPS": 1000.0,
        "TP": 3669,
        "FP": 1914,
        "FN": 1494
    },
    "test_merge_split_case": {
        "F1": 0.6781688069797078,
        "Precision": 0.6781023328759067,
        "Recall": 0.678235294117647,
        "FPS": 1000.0,
        "TP": 3459,
        "FP": 1642,
        "FN": 1641
    },
    "test_highway_case": {
        "F1": 0.9331083593102066,
        "Precision": 0.926879154593489,
        "Recall": 0.9394218588045606,
        "FPS": 1000.0,
        "TP": 16314,
        "FP": 1287,
        "FN": 1052
    }
}
```
- **分析与总结**:
  - 与实验14对比，通过增加训练的epoch和Tmax，评价指标提高了2-4个点。
  - 高速公路性能回到了仅有tusimple数据集时候的水平。
  - 分析本实验的评价指标变化情况，前1/3就达到了与最终性能大体相当的结果。
- 弯道mistakes原因分析：
  - 预测错误；
  - 标签底部线性延伸并不总是符合实际情况；
  - 漏标；
  - 标注与推理的y范围偏差较大；
  - iou判断匹配与draw的算法是否不一致？

## 16. 2025 April 29 AM1, 用前一个实验的模型评价新的数据集；
- 数据集： 按新的规则，重新生成数据集。Valid和test数据集由于重新随机分割会变动。
- Valid性能:
` {'F1': 0.7261798911690437, 'Precision': 0.819736630801349, 'Recall': 0.6517908446657729, 'FPS': 1000.0, 'TP': 20418, 'FP': 4490, 'FN': 10908}`
- test:
```JSON
{
    "test": {
        "F1": 0.7599900514010943,
        "Precision": 0.8174603174603174,
        "Recall": 0.7100697134004648,
        "FPS": 1000.0,
        "TP": 27501,
        "FP": 6141,
        "FN": 11229
    },
    "test_curve_case": {
        "F1": 0.6111871962190177,
        "Precision": 0.7558950514779144,
        "Recall": 0.5129823296069239,
        "FPS": 1000.0,
        "TP": 11380,
        "FP": 3675,
        "FN": 10804
    },
    "test_extreme_weather_case": {
        "F1": 0.5050927487352446,
        "Precision": 0.6380368098159509,
        "Recall": 0.41799709724238027,
        "FPS": 1000.0,
        "TP": 3744,
        "FP": 2124,
        "FN": 5213
    },
    "test_night_case": {
        "F1": 0.5619106699751861,
        "Precision": 0.6015407092575376,
        "Recall": 0.5271796065650098,
        "FPS": 1000.0,
        "TP": 4529,
        "FP": 3000,
        "FN": 4062
    },
    "test_intersection_case": {
        "F1": 0.4553115006652506,
        "Precision": 0.5503639749449806,
        "Recall": 0.38825636942675157,
        "FPS": 1000.0,
        "TP": 9753,
        "FP": 7968,
        "FN": 15367
    },
    "test_up_down_case": {
        "F1": 0.5042050730000673,
        "Precision": 0.6711445459430414,
        "Recall": 0.4037715517241379,
        "FPS": 1000.0,
        "TP": 3747,
        "FP": 1836,
        "FN": 5533
    },
    "test_merge_split_case": {
        "F1": 0.6477200036553048,
        "Precision": 0.6947657322093707,
        "Recall": 0.6066415611092092,
        "FPS": 1000.0,
        "TP": 3544,
        "FP": 1557,
        "FN": 2298
    },
    "test_highway_case": {
        "F1": 0.9329367689535849,
        "Precision": 0.9267087097324016,
        "Recall": 0.9392491074513417,
        "FPS": 1000.0,
        "TP": 16311,
        "FP": 1290,
        "FN": 1055
    }
}
```
- ==分析与总结==
  - 由于更新了openlane的标签，valid和test数据集的F1显著下降。新的标注质量比之前的高，而且标注的车道线更多。可以理解为考试要求提高了。

## 17. 2025 April 29 AM2, 
- 数据集： 去掉tusimple的数据，openlane使用更新标签后的数据集进行初步训练. 整个数据集最大车道线数量为4。
- anchors文件： 同上。
- epochs: 20. T_max: 20* 10814
- 训练参数修改：置信度conf_threshold 0.5；nms_thres 50；nms_topk 4；
- 训练时长： 预期4个小时左右；
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250429AM2.png)； 对pred采用3次样条插值.
- 最优epoch（TODO）模型在valid数据集：
  `{'F1': 0.779599462224783, 'Precision': 0.8248660631812303, 'Recall': 0.7390426377118644, 'FPS': 1000.0, 'TP': 22325, 'FP': 4740, 'FN': 7883}`
- 分场景test性能(test是上述混合数据集;高速场景是tusimple+llamas;其他场景来自openlane的test):
```JSON
{
    "test": {
        "F1": 0.7669065538647475,
        "Precision": 0.8108108108108109,
        "Recall": 0.7275127679533058,
        "FPS": 1000.0,
        "TP": 20940,
        "FP": 4886,
        "FN": 7843
    },
    "test_curve_case": {
        "F1": 0.7348672394543955,
        "Precision": 0.7960353349458408,
        "Recall": 0.6824287774972954,
        "FPS": 1000.0,
        "TP": 15139,
        "FP": 3879,
        "FN": 7045
    },
    "test_extreme_weather_case": {
        "F1": 0.6228037842486792,
        "Precision": 0.6923917497609616,
        "Recall": 0.5659260913252205,
        "FPS": 1000.0,
        "TP": 5069,
        "FP": 2252,
        "FN": 3888
    },
    "test_night_case": {
        "F1": 0.6367218282111898,
        "Precision": 0.6643055906906147,
        "Recall": 0.6113374461645908,
        "FPS": 1000.0,
        "TP": 5252,
        "FP": 2654,
        "FN": 3339
    },
    "test_intersection_case": {
        "F1": 0.5610144764897318,
        "Precision": 0.6430666323642912,
        "Recall": 0.497531847133758,
        "FPS": 1000.0,
        "TP": 12498,
        "FP": 6937,
        "FN": 12622
    },
    "test_up_down_case": {
        "F1": 0.6273394391034757,
        "Precision": 0.6644173996867092,
        "Recall": 0.5941810344827586,
        "FPS": 1000.0,
        "TP": 5514,
        "FP": 2785,
        "FN": 3766
    },
    "test_merge_split_case": {
        "F1": 0.7082429501084598,
        "Precision": 0.7502872462657986,
        "Recall": 0.670660732625813,
        "FPS": 1000.0,
        "TP": 3918,
        "FP": 1304,
        "FN": 1924
    },
    "test_highway_case": {
        "F1": 0.9302163779308426,
        "Precision": 0.961317227494967,
        "Recall": 0.9010648335355169,
        "FPS": 1000.0,
        "TP": 6685,
        "FP": 269,
        "FN": 734
    }
}
```
- 变更test_parameters参数，max lanes设为4,重新评估：
  - 在当前数据集上，最佳参数还是0.5, 45~50

## 18. 2025 April 30 AM1, 
- 数据集： 去掉tusimple的数据，openlane使用更新标签后的数据集进行初步训练. 整个数据集最大车道线数量为4。
- anchors文件： 同上。
- epochs: 40. T_max: 40* 10814
- 训练参数修改：置信度conf_threshold 0.5；nms_thres 50；nms_topk 4；
- 训练时长： 预期4个小时左右；
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250430AM1.png)； 对pred采用3次样条插值.
- 最优epoch（39）模型在valid数据集：
  `{'F1': 0.7861322003204011, 'Precision': 0.8292799412196914, 'Recall': 0.7472523834745762, 'FPS': 1000.0, 'TP': 22573, 'FP': 4647, 'FN': 7635}`
- 分场景test性能(test是上述混合数据集;高速场景是llamas;其他场景来自openlane的test):
```JSON
{
    "test": {
        "F1": 0.7654052091841367,
        "Precision": 0.8013453937366981,
        "Recall": 0.7325504638154466,
        "FPS": 1000.0,
        "TP": 21085,
        "FP": 5227,
        "FN": 7698
    },
    "test_curve_case": {
        "F1": 0.7414985590778097,
        "Precision": 0.7934827302631579,
        "Recall": 0.6959069599711504,
        "FPS": 1000.0,
        "TP": 15438,
        "FP": 4018,
        "FN": 6746
    },
    "test_extreme_weather_case": {
        "F1": 0.6181265206812653,
        "Precision": 0.6790057463584124,
        "Recall": 0.5672658256112537,
        "FPS": 1000.0,
        "TP": 5081,
        "FP": 2402,
        "FN": 3876
    },
    "test_night_case": {
        "F1": 0.6480038364704472,
        "Precision": 0.6680262019527871,
        "Recall": 0.6291467815155395,
        "FPS": 1000.0,
        "TP": 5405,
        "FP": 2686,
        "FN": 3186
    },
    "test_intersection_case": {
        "F1": 0.5653820744329443,
        "Precision": 0.6326761951700345,
        "Recall": 0.5110270700636943,
        "FPS": 1000.0,
        "TP": 12837,
        "FP": 7453,
        "FN": 12283
    },
    "test_up_down_case": {
        "F1": 0.6309910526015843,
        "Precision": 0.6696504173218821,
        "Recall": 0.596551724137931,
        "FPS": 1000.0,
        "TP": 5536,
        "FP": 2731,
        "FN": 3744
    },
    "test_merge_split_case": {
        "F1": 0.7006004122233176,
        "Precision": 0.7351890163626105,
        "Recall": 0.6691201643272852,
        "FPS": 1000.0,
        "TP": 3909,
        "FP": 1408,
        "FN": 1933
    },
    "test_highway_case": {
        "F1": 0.9297928541637703,
        "Precision": 0.9599540691832926,
        "Recall": 0.9014692007009031,
        "FPS": 1000.0,
        "TP": 6688,
        "FP": 279,
        "FN": 731
    }
}
```

## 18. 2025 May AM1, 在实验17的基础上，iou阈值改为0.4
- Review valid的mistakes，发现有很多FP其实在图像上看还不错。比如，标注与预测的y值范围不一样。
- Valid性能：
```JSON
 {'F1': 0.8206548241031, 'Precision': 0.8658312509187124, 'Recall': 0.7799589512711864, 'FPS': 1000.0, 'TP': 23561, 'FP': 3651, 'FN': 6647}
``` 
- test性能：
```JSON
{
    "test": {
        "F1": 0.8081831218573581,
        "Precision": 0.8462327986010796,
        "Recall": 0.7734079143939131,
        "FPS": 1000.0,
        "TP": 22261,
        "FP": 4045,
        "FN": 6522
    },
    "test_curve_case": {
        "F1": 0.7884245917387127,
        "Precision": 0.8436986019736842,
        "Recall": 0.7399477100613054,
        "FPS": 1000.0,
        "TP": 16415,
        "FP": 3041,
        "FN": 5769
    },
    "test_extreme_weather_case": {
        "F1": 0.6784671532846714,
        "Precision": 0.7452893224642523,
        "Recall": 0.6226415094339622,
        "FPS": 1000.0,
        "TP": 5577,
        "FP": 1906,
        "FN": 3380
    },
    "test_night_case": {
        "F1": 0.7157844253941611,
        "Precision": 0.7379480840543882,
        "Recall": 0.6949132813409382,
        "FPS": 1000.0,
        "TP": 5970,
        "FP": 2120,
        "FN": 2621
    },
    "test_intersection_case": {
        "F1": 0.6237525609675502,
        "Precision": 0.6983179598480738,
        "Recall": 0.5635748407643312,
        "FPS": 1000.0,
        "TP": 14157,
        "FP": 6116,
        "FN": 10963
    },
    "test_up_down_case": {
        "F1": 0.694289296705802,
        "Precision": 0.7368739414468909,
        "Recall": 0.6563577586206897,
        "FPS": 1000.0,
        "TP": 6091,
        "FP": 2175,
        "FN": 3189
    },
    "test_merge_split_case": {
        "F1": 0.7475109875325141,
        "Precision": 0.7851893725268513,
        "Recall": 0.7132831222184184,
        "FPS": 1000.0,
        "TP": 4167,
        "FP": 1140,
        "FN": 1675
    },
    "test_highway_case": {
        "F1": 0.9413318504101209,
        "Precision": 0.9718673747667576,
        "Recall": 0.9126566922765872,
        "FPS": 1000.0,
        "TP": 6771,
        "FP": 196,
        "FN": 648
    }
}
```

## 19. 2025 May 13 PM1, 加入修订后的tusimple数据集 
- 数据集： 加入修订后的tusimple数据集
- anchors文件： 同上。
- epochs: 40. T_max: 40* 10814
- 最优性能(Valid数据集): ![Metrics](./img/tusimple_metrics_plot-20250513PM1.png)； 
- epoch35在Valid性能：
```JSON
{'F1': 0.8239286664892201, 'Precision': 0.8663588021270641, 'Recall': 0.7854605430093885, 'FPS': 1000.0, 'TP': 24764, 'FP': 3820, 'FN': 6764}
``` 
- 在test性能（不包含tusimple的test数据集）：
```JSON
{
    "test": {
        "F1": 0.8145559135674367,
        "Precision": 0.8555640535372849,
        "Recall": 0.7772991001632908,
        "FPS": 1000.0,
        "TP": 22373,
        "FP": 3777,
        "FN": 6410
    },
    "test_curve_case": {
        "F1": 0.7960050896694116,
        "Precision": 0.8515075247829883,
        "Recall": 0.7472953479985576,
        "FPS": 1000.0,
        "TP": 16578,
        "FP": 2891,
        "FN": 5606
    },
    "test_extreme_weather_case": {
        "F1": 0.6766770432765672,
        "Precision": 0.7464314570428225,
        "Recall": 0.6188455956235347,
        "FPS": 1000.0,
        "TP": 5543,
        "FP": 1883,
        "FN": 3414
    },
    "test_night_case": {
        "F1": 0.7123518440527044,
        "Precision": 0.7372353673723536,
        "Recall": 0.689093237108602,
        "FPS": 1000.0,
        "TP": 5920,
        "FP": 2110,
        "FN": 2671
    },
    "test_intersection_case": {
        "F1": 0.6264638816940072,
        "Precision": 0.7056505909929679,
        "Recall": 0.5632563694267516,
        "FPS": 1000.0,
        "TP": 14149,
        "FP": 5902,
        "FN": 10971
    },
    "test_up_down_case": {
        "F1": 0.7071723280393375,
        "Precision": 0.7436110780934269,
        "Recall": 0.6741379310344827,
        "FPS": 1000.0,
        "TP": 6256,
        "FP": 2157,
        "FN": 3024
    },
    "test_merge_split_case": {
        "F1": 0.7450767841011743,
        "Precision": 0.7888293802601377,
        "Recall": 0.7059226292365628,
        "FPS": 1000.0,
        "TP": 4124,
        "FP": 1104,
        "FN": 1718
    },
    "test_highway_case": {
        "F1": 0.940954510516386,
        "Precision": 0.9769297736506094,
        "Recall": 0.9075347081816957,
        "FPS": 1000.0,
        "TP": 6733,
        "FP": 159,
        "FN": 686
    }
}
```


## TODO List
- 把openlane的标签数据可视化为视频，方便人工检查；
- 转换openlane的test数据集，评估特殊场景下的性能；--Done
- 更多epochs的训练--DONE