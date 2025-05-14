# 如何运行onnx模型


- 下载仓库代码https://github.com/zhengyb/LaneATT.git，并checkout dev分支；（或者只下载lanedet_on_onnx_cpu.py文件）
- 网盘下载校验数据集和onnx模型文件;
- 参考lanedet_on_onnx_cpu.py的说明，创建python虚拟环境，并安装依赖；
- 修改lanedet_on_onnx_cpu.py中的校验数据集标签文件路径和模型路径；
- 运行python lanedet_on_onnx_cpu.py, 预期输出如下：
```bash
(onnx_infer) zyb@zyb-CORSAIR-VENGEANCE-i8100:/data/LaneATT$ python lanedet_on_onnx_cpu.py 
Available ONNX Runtime providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
Using device: cpu
Validate onnx model
/ 999
Metrics:
{'F1': 0.8249113475177304, 'Precision': 0.8671947809878844, 'Recall': 0.7865595942519019, 'FPS': 1000.0, 'TP': 1861, 'FP': 285, 'FN': 505}
Validate onnx model done

```