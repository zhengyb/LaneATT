import os
import argparse
from pathlib import Path

def print_tree(path, level_limits, current_level=0):
    if current_level >= len(level_limits):
        return
    
    entries = sorted(os.listdir(path))
    print("│   " * current_level + "├── " + os.path.basename(path) + "/")
    
    for i, entry in enumerate(entries[:level_limits[current_level]]):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            print_tree(full_path, level_limits, current_level+1)
        else:
            print("│   " * (current_level+1) + "├── " + entry)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='打印目录树结构')
    parser.add_argument('--path', type=str, default='./', 
                      help='目标目录路径（默认为当前目录）')
    args = parser.parse_args()
    
    target_path = Path(args.path)
    print_tree(target_path, level_limits=[5, 5, 3, 2]) 