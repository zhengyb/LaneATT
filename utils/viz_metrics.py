#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Visualize metrics from json files
# JSON file directory structure:
# - <metrics_dir_root>/
#   - epoch_0001/
#     - val_metrics.json
#   - epoch_0002/
#     - val_metrics.json
#   - ...
#
# Each val_metrics.json file contains a dictionary with the following format:
# {"TP": 67273, "FP": 1906, "FN": 6762, "Precision": 0.9724482863296665, "Recall": 0.9086648206929155, "F1": 0.9394751909729496}

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def find_epoch_dirs(root_dir):
    """查找所有包含val_metrics.json的epoch目录并按数字排序"""
    epoch_dirs = []
    for d in Path(root_dir).iterdir():
        if d.is_dir() and d.name.startswith("epoch_"):
            metric_file = d / "val_metrics.json"
            if metric_file.exists():
                epoch_dirs.append(d)
    # 按epoch数字排序
    epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
    return epoch_dirs

def plot_metrics_tusimple(epochs, metrics, max_f1, args):
    """绘制指标曲线图"""
    plt.figure(figsize=(12, 6))
    # 绘制三条曲线
    plt.plot(epochs, metrics['Accuracy'], label='Accuracy', marker='*', markersize=3)
    plt.plot(epochs, metrics['FP'], label='FP', marker='o', markersize=3)
    plt.plot(epochs, metrics['FN'], label='FN', marker='o', markersize=3)
    
    # 标注最大值
    plt.scatter(max_f1['epoch'], max_f1['value'], color='red', zorder=5, 
                label=(f'Max Accuracy: {max_f1["value"]:.4f} @ Epoch {max_f1["epoch"]}\n'
                       f'FP: {max_f1["FP"]:.4f}\n'
                       f'FN: {max_f1["FN"]:.4f}'))
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs[::1])  # 每5个epoch显示一个刻度
    plt.tight_layout()
    # 保存图片到指标目录
    save_path = Path(args.metrics_dir) / "tusimple_metrics_plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Metrics plot saved to: {save_path}")    


def plot_metrics_llamas(epochs, metrics, max_f1, args):
    """绘制指标曲线图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制三条曲线
    plt.plot(epochs, metrics['Precision'], label='Precision', marker='o', markersize=3)
    plt.plot(epochs, metrics['Recall'], label='Recall', marker='o', markersize=3)
    plt.plot(epochs, metrics['F1'], label='F1 Score', marker='*', markersize=3)
    
    # 标注最大值
    plt.scatter(max_f1['epoch'], max_f1['value'], color='red', zorder=5, 
                label=(f'Max F1: {max_f1["value"]:.4f} @ Epoch {max_f1["epoch"]}\n'
                       f'Precision: {max_f1["precision"]:.4f}\n'
                       f'Recall: {max_f1["recall"]:.4f}'))
    
    plt.title('Validation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs[::1])  # 每5个epoch显示一个刻度
    plt.tight_layout()
    # 保存图片到指标目录
    save_path = Path(args.metrics_dir) / "llamas_metrics_plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Metrics plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('metrics_dir', type=str, help='Root directory containing epoch folders')
    args = parser.parse_args()

    # 收集数据
    llamas_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    tusimple_metrics = {'Accuracy': [], 'FP': [], 'FN': []}
    llamas_max_f1 = {'value': 0, 'epoch': 0, 'precision': 0, 'recall': 0}
    tusimple_max_f1 = {'value': 0, 'epoch': 0, 'FP': 0, 'FN': 0}
    epochs = []

    dataset_type = "llamas"
    metrics = llamas_metrics
    max_metrics = llamas_max_f1
    if "tusimple" in args.metrics_dir:
        dataset_type = "tusimple"
        metrics = tusimple_metrics
        max_metrics = tusimple_max_f1

    for epoch_dir in find_epoch_dirs(args.metrics_dir):
        epoch_num = int(epoch_dir.name.split("_")[1])
        print(epoch_num)
        with open(epoch_dir / "val_metrics.json") as f:
            data = json.load(f)
        print(data)
        print(metrics)
        # 记录当前epoch数据
        epochs.append(epoch_num)
        for k in metrics:
            metrics[k].append(data[k])
        
        # 更新最大F1值
        if dataset_type == "llamas":
            main_metric = data['F1']
        else:
            main_metric = data['Accuracy']    
        if main_metric > max_metrics['value']:
            max_metrics['value'] = main_metric
            max_metrics['epoch'] = epoch_num
        for k in max_metrics.keys():
            if k == 'value' or k == 'epoch':
                continue
            max_metrics[k] = data[k]

    # 绘制图表
    if dataset_type == "llamas":
        plot_metrics_llamas(epochs, metrics, max_metrics, args)
    else:
        plot_metrics_tusimple(epochs, metrics, max_metrics, args)

if __name__ == '__main__':
    main()
