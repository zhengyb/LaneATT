# Debug log


## Training Results on Llamas dataset

```JSON
[
    {
        "epoch": 20,
        "metrics": {"TP": 68038, "FP": 1834, "FN": 5997, "Precision": 0.9737520036638424, "Recall": 0.9189977713243737, "F1": 0.9455829111856963}
    },
    {
        "epoch": 30,
        "metrics": {"TP": 65316, "FP": 1876, "FN": 8719, "Precision": 0.9720800095249434, "Recall": 0.8822313770513946, "F1": 0.9249789346229829}
    }
    {
        "epoch": 40,
        "metrics": {"TP": 67875, "FP": 1799, "FN": 6160, "Precision": 0.9741797514137268, "Recall": 0.9167961099479975, "F1": 0.9446172473540281}
    },
    {
        "epoch": 50,
        "metrics": {"TP": 68357, "FP": 1844, "FN": 5678, "Precision": 0.973732567912138, "Recall": 0.9233065442020666, "F1": 0.9478493579966166}
    },
    {
        "epoch": 60,
        "metrics": {"TP": 64570, "FP": 1872, "FN": 9465, "Precision": 0.9718250504199151, "Recall": 0.8721550617950969, "F1": 0.9192963972749987}
    },
    {
        "epoch": 70,
        "metrics": {"TP": 68088, "FP": 1914, "FN": 5947, "Precision": 0.9726579240593126, "Recall": 0.9196731275747957, "F1": 0.9454237452876691}
    },
    {
        "epoch": 80,
        "metrics": {"TP": 67503, "FP": 1774, "FN": 6532, "Precision": 0.9743926555711131, "Recall": 0.9117714594448572, "F1": 0.9420425365635816}
    },
    {
        "epoch": 90,
        "metrics": {"TP": 67197, "FP": 2276, "FN": 6838, "Precision": 0.967239071293884, "Recall": 0.9076382791922739, "F1": 0.93649134543022}
    },
    {
        "epoch": 100,
        "metrics": {"TP": 67273, "FP": 1906, "FN": 6762, "Precision": 0.9724482863296665, "Recall": 0.9086648206929155, "F1": 0.9394751909729496}
    }
]
```


## 模型泛化能力的影响因素
- anchor mask频率文件
  模型初始化时，会用anchor_mask频率文件来过滤anchor。该文件是通过对训练集中的anchor进行统计得到的。
  如果训练集和测试集的anchor分布差异较大，那么模型在测试集上的表现会较差。用于生成该文件的训练数据集应该具备良好的多样性。
- 其他


## 如何重新生成锚文件
- 准备好training数据集;
- 修改data/laneatt_xxx_yyy.yml文件，去掉其中`anchors_freq_path: 'data/tusimple_anchors_freq.pt'`这一行；
- 运行`python utils/gen_anchor_mask.py --cfg ./cfgs/laneatt_tusimple_resnet18.yml --output tusimple_250418_anchors_mask.pt`

## 如何重新训练
- 准备好数据集
- 修改`./cfgs/laneatt_tusimple_resnet18.yml`，选择训练参数
- 运行`python main.py train --exp_name laneatt_r18_tusimple --cfg ./cfgs/laneatt_tusimple_resnet18.yml `

## 训练后如何生成指标图
- 运行`python utils/viz_metrics.py experiments/laneatt_r18_tusimple/results`
```
Metrics plot saved to: experiments/laneatt_r18_tusimple/results/tusimple_metrics_plot.png
```