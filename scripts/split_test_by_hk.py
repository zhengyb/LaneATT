#!/usr/bin/env python3
"""Split CARLA test annotations by camera height Hk.

Reads all *_test*.json files from datasets/carla_tusimple/, groups annotations
by Hk (H1-H6 extracted from raw_file path), and writes per-Hk output files.

Usage:
    python scripts/split_test_by_hk.py
    python scripts/split_test_by_hk.py --input-dir datasets/carla_tusimple --max-lines 1000

Output:
    datasets/carla_tusimple/H1_test_1.json
    datasets/carla_tusimple/H2_test_1.json
    ...
    datasets/carla_tusimple/H6_test_1.json
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Split CARLA test data by camera height Hk")
    parser.add_argument("--input-dir", default="datasets/carla_tusimple",
                        help="Directory containing annotation files")
    parser.add_argument("--max-lines", type=int, default=1000,
                        help="Max lines per output file (default: 1000)")
    return parser.parse_args()


def extract_hk(raw_file):
    """Extract Hk value (e.g. 'H1') from raw_file path like .../tusimple/H1/images/..."""
    m = re.search(r'/tusimple/(H\d+)/', raw_file)
    return m.group(1) if m else None


def main():
    args = parse_args()
    input_dir = args.input_dir

    # Find all test annotation files
    pattern = os.path.join(input_dir, '*_test*.json')
    test_files = sorted(glob.glob(pattern))

    # Exclude our own output files (H*_test_*.json)
    test_files = [f for f in test_files if not re.match(r'H\d+_test', os.path.basename(f))]

    if not test_files:
        print("No test files found matching: {}".format(pattern))
        return 1

    print("Input files ({}):\n  {}".format(len(test_files), '\n  '.join(test_files)))

    # Read all annotations and group by Hk
    hk_groups = defaultdict(list)
    unknown = []

    for fpath in test_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                hk = extract_hk(data['raw_file'])
                if hk:
                    hk_groups[hk].append(line)
                else:
                    unknown.append(line)

    if unknown:
        print("\nWARNING: {} annotations with unknown Hk".format(len(unknown)))

    # Write per-Hk output files
    print("\nOutput:")
    total_written = 0
    for hk in sorted(hk_groups.keys()):
        lines = hk_groups[hk]
        num_files = math.ceil(len(lines) / args.max_lines)
        for file_idx in range(num_files):
            start = file_idx * args.max_lines
            end = min(start + args.max_lines, len(lines))
            chunk = lines[start:end]

            out_name = "{}_test_{}.json".format(hk, file_idx + 1)
            out_path = os.path.join(input_dir, out_name)
            with open(out_path, 'w') as f:
                f.write('\n'.join(chunk) + '\n')

            print("  {} ({} records)".format(out_name, len(chunk)))
            total_written += len(chunk)

    print("\nSummary:")
    for hk in sorted(hk_groups.keys()):
        print("  {}: {} records".format(hk, len(hk_groups[hk])))
    print("  Total: {}".format(total_written))

    return 0


if __name__ == '__main__':
    exit(main())
