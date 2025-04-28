import argparse

import cv2
import torch
import random
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import Config


DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a dataset")
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--split",
                        choices=["train", "test", "val"],
                        default='train',
                        help="Dataset split to visualize")
    parser.add_argument("--video_path", default=None,
                        help="Path to video file")
    args = parser.parse_args()

    return args


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    args = parse_args()
    cfg = Config(args.cfg)
    train_dataset = cfg.get_dataset(args.split)
    if args.video_path is None:
        args.video_path = f"{train_dataset.name}_{args.split}.mp4"

    # 获取一帧确定视频尺寸
    img, _, _ = train_dataset.draw_annotation(0)
    height, width = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.video_path, fourcc, 2, (width, height))

    for idx in range(len(train_dataset)):
        if DEBUG and idx >= 30:
            print('调试：已处理30帧，提前结束。')
            break
        img, _, _ = train_dataset.draw_annotation(idx)
        img_path = train_dataset.annotations[idx]['path'] if hasattr(train_dataset, 'annotations') and 'path' in train_dataset.annotations[idx] else ''
        if DEBUG:
            print(f'DEBUG idx={idx}, img_path={img_path}')
        img_path = f"{idx}:{img_path}"
        # 在顶部写上路径
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        # 分割长路径为多行
        def split_text(text, max_len=40):
            return [text[i:i+max_len] for i in range(0, len(text), max_len)]
        lines = split_text(img_path, 55)
        # 计算单行高度
        sample_line = lines[0] if lines else ''
        (text_width, text_height), _ = cv2.getTextSize(sample_line, font, font_scale, thickness)
        line_height = text_height + 5
        # 画底色
        cv2.rectangle(img, (0, 0), (width, line_height * len(lines) + 5), (0, 0, 0), -1)
        # 写多行
        for i, line in enumerate(lines):
            cv2.putText(img, line, (5, line_height * (i + 1)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        # cv2.imshow('sample', img)
        # cv2.waitKey(1)
        video_writer.write(img)
    video_writer.release()
    print(f"Video saved to {args.video_path}")


if __name__ == "__main__":
    main()
