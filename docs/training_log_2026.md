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
        "F1": 0.5739937527889336,
        "Precision": 0.6127434167905187,
        "Recall": 0.5398536126779478,
        "FPS": 1000.0,
        "TP": 16079,
        "FP": 10162,
        "FN": 13705
    },
    "F1": 0.5739937527889336,
    "Precision": 0.6127434167905187,
    "Recall": 0.5398536126779478,
    "FPS": 1000.0,
    "TP": 16079,
    "FP": 10162,
    "FN": 13705
}


```