import random
import argparse
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import trange

# add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import Config
from lib.models.matching import match_proposals_with_targets



def get_anchors_use_frequency(cfg, split='train', t_pos=15., t_neg=20.):
    model = cfg.get_model()
    anchors_frequency = torch.zeros(len(model.anchors), dtype=torch.int32)
    nb_unmatched_targets = 0
    dataset = cfg.get_dataset(split)
    # 遍历数据集中的每个样本
    for idx in trange(len(dataset)):
        # 获取当前样本的标注信息
        _, targets, _ = dataset[idx]
        # 假设只处理类别1的目标
        targets = targets[targets[:, 1] == 1]
        n_targets = len(targets)
        if n_targets == 0:
            continue
        targets = torch.tensor(targets)
        # 核心匹配逻辑
        positives_mask, _, _, target_indices = match_proposals_with_targets(model,
                                                                            model.anchors, # 模型预定义的锚框
                                                                            targets,      # 当前图片的真实目标框
                                                                            t_pos=t_pos, # 正样本IOU阈值（默认30%）
                                                                            t_neg=t_neg) # 负样本IOU阈值（默认35%）
        # 计算匹配的目标框数量
        n_matches = len(set(target_indices.tolist()))
        # 计算未匹配的目标框数量
        nb_unmatched_targets += n_targets - n_matches
        assert (n_targets - n_matches) >= 0
        # 累加当前图片的正样本锚点使用次数
        anchors_frequency += positives_mask

    return anchors_frequency


def save_mask(cfg_path, output_path):
    cfg = Config(cfg_path)
    frequency = get_anchors_use_frequency(cfg, split='train', t_pos=30., t_neg=35.)
    torch.save(frequency, output_path)


def map_freq_to_binary(freq, threshold=0.5):
    # freq: [0, 1]
    # convert to binary
    binary = (freq * 255).astype(np.uint8)
    # convert to binary
    binary = np.where(binary >= (threshold * 255), np.uint8(255), np.uint8(0)) 
    # convert to gray
    #return cv2.cvtColor(freq, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def map_freq_to_rgb(freq):
    # freq: [0, 1]
    freq = np.array(freq * 255, dtype=np.uint8)
    # convert to rgb
    return cv2.applyColorMap(freq, cv2.COLORMAP_JET)

def draw_mask(anchors_freq_path, output_path, topk=1000):
    print(f"anchors_freq_path: {anchors_freq_path}")
    # load frequency
    anchors_mask = torch.load(anchors_freq_path).cpu().numpy()

    # normalize the anchors_mask
    min_freq = anchors_mask.min()
    max_freq = anchors_mask.max()
    mean_freq = anchors_mask.mean()
    anchors_mask = (anchors_mask - min_freq) / (max_freq - min_freq)  # 防止除零

    print(f"min_freq: {min_freq:.4f}, max_freq: {max_freq:.4f}, mean_freq: {mean_freq:.4f}")

    # get the threshold of the anchors_mask
    #sorted_anchors_mask = torch.argsort(anchors_mask, descending=True)
    #threshold = float(anchors_mask[sorted_anchors_mask[1000]])
    #print(f"threshold: {threshold}")

    # Get the threshold of the anchors_mask
    sorted_indices = np.argsort(-anchors_mask)  # 降序排列索引
    if len(sorted_indices) < topk:
        raise ValueError("锚点数量不足{}个，当前数量为{}".format(topk, len(sorted_indices)))

    threshold = float(anchors_mask[sorted_indices[topk-1]])  # 第1000个元素的索引是999
    print(f"Normalized threshold: {threshold:.4f}")  # 增加小数精度显示
    above_threshold = np.sum(anchors_mask > threshold)  # 统计严格大于阈值的数量
    at_threshold = np.sum(anchors_mask == threshold)    # 统计等于阈值的数量

    print(f"阈值: {threshold:.4f}")
    print(f"超过阈值的锚点数: {above_threshold} (严格大于)")
    print(f"等于阈值的锚点数: {at_threshold}")
    print(f"总有效锚点数: {above_threshold + at_threshold} (>=阈值)")
    print(f"anchors_mask.shape: {anchors_mask.shape}")

    # 原始分割代码
    left_anchors_mask = anchors_mask[:432].reshape(72, 6)
    bottom_anchors_mask = anchors_mask[432:432+1920].reshape(128, 15)
    right_anchors_mask = anchors_mask[432+1920:].reshape(72, 6)

    # 新增填充逻辑
    # 统一列数为15（与bottom_anchors_mask一致）
    left_padded = np.pad(left_anchors_mask, 
                        pad_width=((0,0), (0,15-6)),  # 行不填充，列右侧填充9个0
                        mode='constant', 
                        constant_values=0)
    
    right_padded = np.pad(right_anchors_mask,
                         pad_width=((0,0), (0,15-6)),
                         mode='constant',
                         constant_values=0)

    # 验证维度
    print(f"填充后维度: left {left_padded.shape}, bottom {bottom_anchors_mask.shape}, right {right_padded.shape}")
    # 输出: (72,15), (128,15), (72,15)

    # 安全拼接
    anchors_mask = np.concatenate([left_padded, bottom_anchors_mask, right_padded], axis=0)

    # transpose
    anchors_mask = anchors_mask.transpose(1, 0)

    print(anchors_mask.shape) # (15, 272)

    # 改进后的resize逻辑
    # 获取原始尺寸
    orig_height, orig_width = anchors_mask.shape[:2]
    
    # 参数化缩放因子（可配置）
    width_scale = 3    # 宽度放大3倍
    height_scale = 3  # 高度放大16倍
    
    # 计算目标尺寸
    resized_w = orig_width * width_scale
    resized_h = orig_height * height_scale
    
    # 根据数据类型选择插值方法
    if anchors_mask.dtype == np.uint8:
        interpolation = cv2.INTER_NEAREST  # 适用于二值/分类数据
    else:
        interpolation = cv2.INTER_LINEAR   # 适用于连续值
        
    # 执行resize并保持宽高比
    anchors_mask = cv2.resize(
        anchors_mask, 
        (int(resized_w), int(resized_h)), 
        interpolation=interpolation
    )
    
    print(f"Resized from {orig_width}x{orig_height} to {resized_w}x{resized_h}")

    # convert to RGB
    #anchors_mask = map_freq_to_rgb(anchors_mask)
    anchors_mask = map_freq_to_binary(anchors_mask, threshold)
   
    cv2.imwrite(output_path, anchors_mask)
    pass

def normalize_anchors_mask(anchors_mask):
    min_freq = anchors_mask.min()
    max_freq = anchors_mask.max()
    mean_freq = anchors_mask.mean()
    anchors_mask = (anchors_mask - min_freq) / (max_freq - min_freq)  # 防止除零
    return anchors_mask

def merge_mask(anchors_freq_path1, anchors_freq_path2, anchors_freq_path3, output_path):
    anchors_mask1_np = torch.load(anchors_freq_path1).cpu().numpy()
    anchors_mask2_np = torch.load(anchors_freq_path2).cpu().numpy()
    anchors_mask3_np = torch.load(anchors_freq_path3).cpu().numpy()

    # normalize the anchors_mask
    nor_anchors_mask1_np = normalize_anchors_mask(anchors_mask1_np)
    nor_anchors_mask2_np = normalize_anchors_mask(anchors_mask2_np)
    nor_anchors_mask3_np = normalize_anchors_mask(anchors_mask3_np)

    sum_anchors_mask = nor_anchors_mask1_np * 0.1 + nor_anchors_mask2_np * 2.0+ nor_anchors_mask3_np * 0.1

    sum_anchors_mask = (sum_anchors_mask * 1000).astype(np.int32)

    sum_anchors_mask_ts = torch.Tensor(sum_anchors_mask)
    torch.save(sum_anchors_mask_ts, output_path)


def view_mask(cfg_path, output_path):
    cfg = Config(cfg_path)
    model = cfg.get_model()
    img = model.draw_anchors(img_w=512, img_h=288)
    #cv2.imshow('anchors', img)
    # save image
    cv2.imwrite('anchors.png', img)
    #cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute anchor frequency for later use as anchor mask")
    parser.add_argument("--output", help="Output path (e.g., `anchors_mask.pt`", required=True)
    parser.add_argument("--cfg", help="Config file (e.g., `config.yml`")
    args = parser.parse_args()

    return args


def main_save_mask():
    args = parse_args()
    # Fix seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_mask(args.cfg, args.output)

if __name__ == '__main__':
    mask_path = './data/llamas_anchors_freq.pt'
    output_path = './data/llamas_anchors_mask.png'

    mask_path_list = [
        ['./data/llamas_anchors_freq.pt', './data/llamas_anchors_mask.png'],
        ['./data/tusimple_anchors_freq.pt', './data/tusimple_anchors_mask.png'],
        ['./data/culane_anchors_freq.pt', './data/culane_anchors_mask.png'],
    ]

    for mask_path, output_path in mask_path_list:
        draw_mask(mask_path, output_path)

    merge_mask(mask_path_list[0][0], mask_path_list[1][0], mask_path_list[2][0], './data/sum_anchors_mask.pt')
    draw_mask('./data/sum_anchors_mask.pt', './data/sum_anchors_mask.png')

    #view_mask('./experiments/laneatt_r18_llamas/config.yaml', None)