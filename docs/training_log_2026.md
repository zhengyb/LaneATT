# Training Log 2026

这里记录2026年的训练记录。

## 1. 2025年模型在carla数据集上的表现
### 测试模型： './onnx/LaneATT_r18_tusimple-0513.onnx'
- 备注： 不是产品使用的模型, conf_threshold=0.5
- 测试工具： lanedet_on_onnx.py
- 评估指标：
```
Total Metrics: {
    "TP": 12188,
    "FP": 7523,
    "FN": 17596,
    "Precision": 0.6183349398812845,
    "Recall": 0.4092130002686006,
    "F1": 0.49249419133245786,
    "FPS": 1000.0
}
```

### 测试模型： './onnx/LaneATT_test-0529RGB.sim.onnx'
- 备注: 产品中使用的模型, conf_threshold=0.3,
- 测试工具： lanedet_on_onnx_cpu.py
- 评估指标：
```JSON
Total Metrics: {
    "TP": 14800,
    "FP": 10948,
    "FN": 14984,
    "Precision": 0.5748019263632127,
    "Recall": 0.4969110932044051,
    "F1": 0.5330260030252828,
    "FPS": 1000.0
}
```

### 测试模型： ‘experiments/laneatt_r18_tusimple/models/model_0033.pt’
- 备注: 上述RGB模型对应的pt模型,  conf_threshold: 0.3; nms_thres: 50.; nms_topk: 4
- 测试工具： python main.py test --exp_name laneatt_r18_tusimple --epoch 33
- 评估指标：
```
All Metrics: {
    "test": {
        "F1": 0.5742436412315931,
        "Precision": 0.6130101749171144,
        "Recall": 0.5400886381950041,
        "FPS": 1000.0,
        "TP": 16086,
        "FP": 10155,
        "FN": 13698
    },
    "F1": 0.5742436412315931,
    "Precision": 0.6130101749171144,
    "Recall": 0.5400886381950041,
    "FPS": 1000.0,
    "TP": 16086,
    "FP": 10155,
    "FN": 13698
}



```


## 2. Train on carla & test on carla

```
Best test F1: 0.9090 @ epoch 43 (Precision=0.9361, Recall=0.8834)
Best val F1: 0.8976 @ epoch 43 (Precision=0.9307, Recall=0.8668)

```
- Hk Metrics on epoch 43
```
All Metrics: {
    "test": {
        "F1": 0.908983393288326,
        "Precision": 0.9360600832233837,
        "Recall": 0.8834291187739464,
        "FPS": 1000.0,
        "TP": 36892,
        "FP": 2520,
        "FN": 4868
    },
    "test_H1": {
        "F1": 0.9188790560471976,
        "Precision": 0.943939393939394,
        "Recall": 0.8951149425287356,
        "FPS": 1000.0,
        "TP": 6230,
        "FP": 370,
        "FN": 730
    },
    "test_H2": {
        "F1": 0.9239026542166018,
        "Precision": 0.9460924559554283,
        "Recall": 0.9027298850574713,
        "FPS": 1000.0,
        "TP": 6283,
        "FP": 358,
        "FN": 677
    },
    "test_H3": {
        "F1": 0.9181563007829813,
        "Precision": 0.9448160535117057,
        "Recall": 0.8929597701149425,
        "FPS": 1000.0,
        "TP": 6215,
        "FP": 363,
        "FN": 745
    },
    "test_H4": {
        "F1": 0.9153664998528113,
        "Precision": 0.9382920941460471,
        "Recall": 0.8935344827586207,
        "FPS": 1000.0,
        "TP": 6219,
        "FP": 409,
        "FN": 741
    },
    "test_H5": {
        "F1": 0.8979652238253792,
        "Precision": 0.9257055682684974,
        "Recall": 0.8718390804597701,
        "FPS": 1000.0,
        "TP": 6068,
        "FP": 487,
        "FN": 892
    },
    "test_H6": {
        "F1": 0.8791323859386687,
        "Precision": 0.9168486739469579,
        "Recall": 0.844396551724138,
        "FPS": 1000.0,
        "TP": 5877,
        "FP": 533,
        "FN": 1083
    }
}
root@zyb-CORSAIR-VENGEANCE-i8100:/app# 


```


## 3. 未见过的中间安装高度测试(Town04)
- Town04, H7/8/9， test_1.json, test_2.json, val_1.json, val_2.json
- model: epoch 43
- All lane lines metrics:
```
    "test": {
        "F1": 0.9123661525377865,
        "Precision": 0.9368103516710328,
        "Recall": 0.8891651597347799,
        "FPS": 1000.0,
        "TP": 11801,
        "FP": 796,
        "FN": 1471
    }

```
- LR lane lines only metrics: (因为模型不输出lane line名称，Precision不针对LR，这里主要关注Recall)
```
    "test": {
        "F1": 0.6600114386731141,
        "Precision": 0.5038501230451695,
        "Recall": 0.9564496684749849,
        "FPS": 1000.0,
        "TP": 6347,
        "FP": 6250,
        "FN": 289
    }

```

## 4. 未见过的地图Town01 + 未见过的中间高度H7/8/9
- Town01, H7/8/9， train_1 ~ 5(这个地图是双向单车道)
- model: epoch 43
- All lane lines metrics:
```
    "test": {
        "F1": 0.4677184144239692,
        "Precision": 0.441,
        "Recall": 0.497883149872989,
        "FPS": 1000.0,
        "TP": 7056,
        "FP": 8944,
        "FN": 7116
    }

```
- LR lane lines only metrics: (因为模型不输出lane line名称，Precision不针对LR，这里主要关注Recall)
```
    "test": {
        "F1": 0.4122749127934383,
        "Precision": 0.2733125,
        "Recall": 0.8387034906022248,
        "FPS": 1000.0,
        "TP": 4373,
        "FP": 11627,
        "FN": 841
    }

```

## 4. 未见过的地图Town06 + 未见过的中间高度H7/8/9(tusimple-0331)
- Town01, H7/8/9，train 1~11, test1-2, val 1-2
- epoch: 43
- 所有车道线：
```
    "test": {
        "F1": 0.8472511433297656,
        "Precision": 0.8571428571428571,
        "Recall": 0.8375851321713033,
        "FPS": 1000.0,
        "TP": 43536,
        "FP": 7256,
        "FN": 8442
    }
```
- LR only：
```
    "test": {
        "F1": 0.6430930284162849,
        "Precision": 0.48498808890988915,
        "Recall": 0.9541405221163529,
        "FPS": 1000.0,
        "TP": 24634,
        "FP": 26159,
        "FN": 1184
    }

```
- 在carla的未见过地图和高度上，泛化性能不错。

## 5. Town04 + 90度HFOV相机，H1～H9全高度
- 数据集: tusimple-0401, 全数据集，test 1~5, val 1~7, train 1 ~ 44
- epoch: 43
- All lanes: (F1下降2%)
```
    "test": {
        "F1": 0.8812033783367239,
        "Precision": 0.9148069926696106,
        "Recall": 0.8499810172034452,
        "FPS": 1000.0,
        "TP": 185822,
        "FP": 17305,
        "FN": 32797
    }

```
- LR only(下降2个点)
```
    "test": {
        "F1": 0.6492428318856177,
        "Precision": 0.4988305889579248,
        "Recall": 0.9295210424370488,
        "FPS": 1000.0,
        "TP": 21115,
        "FP": 21214,
        "FN": 1601
    }

```
