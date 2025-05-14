#!/bin/python
"""
将LLAMAS数据集转换为TuSimple数据集
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.datasets.tusimple import TuSimple


def sample_tusimple():
    new_dataset_dir = "datasets/sampled_tusimple"
    old_dataset_dir = "datasets/TUSimple/tusimple/"
    # validation set
    sample_num = 1000
    tusimple_dataset = TuSimple(split="val", max_lanes=4, root=old_dataset_dir)
    tusimple_dataset.random_sample_annotations(sample_num, new_dataset_dir)
    # training set
    sample_num = 200
    tusimple_dataset = TuSimple(split="train", max_lanes=4, root=old_dataset_dir)
    tusimple_dataset.random_sample_annotations(sample_num, new_dataset_dir)

if __name__ == "__main__":
    sample_tusimple()