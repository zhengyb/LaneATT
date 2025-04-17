#!/bin/python
"""
将LLAMAS数据集转换为TuSimple数据集
"""

import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.datasets.tusimple import TuSimple
from lib.datasets.llamas import LLAMAS


def convert_llamas_to_tusimple(split='test', llamas_root_dir=None, tusimple_dir=None, copy_images=False, sample_interval=10):
    """
    将LLAMAS数据集转换为TuSimple数据集
    """
    print(f"Converting {split} set of LLAMAS to TuSimple format")
    llamas_dataset = LLAMAS(split=split, max_lanes=4, root=llamas_root_dir)
    llamas_dataset.convert_annotations_to_tusimple_format(tusimple_dir, copy_images=copy_images, sample_interval=sample_interval)
    print(f"Converted {split} set of LLAMAS to TuSimple format\n\n")

    # TODO: valid convertation

if __name__ == "__main__":
    llamas_root_dir = "datasets/llamas/"
    tusimple_test_dir = "datasets/TUSimple/tusimple-test/"
    convert_llamas_to_tusimple(split='val', 
                               llamas_root_dir=llamas_root_dir, 
                               tusimple_dir=tusimple_test_dir,
                               copy_images=False,
                               sample_interval=10)
    
    llamas_root_dir = "datasets/llamas/"
    tusimple_train_dir = "datasets/TUSimple/tusimple/"
    convert_llamas_to_tusimple(split='train', 
                               llamas_root_dir=llamas_root_dir, 
                               tusimple_dir=tusimple_train_dir,
                               copy_images=False,
                               sample_interval=10)