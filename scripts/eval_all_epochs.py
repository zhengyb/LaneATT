#!/usr/bin/env python3
"""Evaluate all epoch checkpoints on test and/or val datasets.

Usage:
    # Evaluate all 100 epochs on both test and val
    python scripts/eval_all_epochs.py --exp_name laneatt_r18_tusimple --splits test,val

    # Evaluate a specific range, skip already-evaluated epochs
    python scripts/eval_all_epochs.py --exp_name laneatt_r18_tusimple --start 1 --end 50 --skip-existing

    # Use a different config file
    python scripts/eval_all_epochs.py --exp_name laneatt_r18_tusimple --cfg cfgs/laneatt_tusimple_resnet18.yml
"""
import argparse
import csv
import json
import logging
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate all epoch checkpoints")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--cfg", default=None, help="Config file (default: use experiment's config)")
    parser.add_argument("--start", type=int, default=1, help="Start epoch (default: 1)")
    parser.add_argument("--end", type=int, default=None, help="End epoch (default: last checkpoint)")
    parser.add_argument("--splits", default="test,val", help="Comma-separated splits to evaluate (default: test,val)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip epochs that already have results")
    return parser.parse_args()


def find_available_epochs(models_dir):
    """Scan models directory and return sorted list of available epoch numbers."""
    pattern = re.compile(r'model_(\d+)\.pt')
    epochs = []
    for f in os.listdir(models_dir):
        m = pattern.match(f)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def has_existing_result(results_dir, epoch, split):
    """Check if metrics JSON already exists for this epoch/split."""
    metrics_path = os.path.join(results_dir, 'epoch_{:04d}'.format(epoch), '{}_metrics.json'.format(split))
    return os.path.exists(metrics_path)


def load_existing_metrics(results_dir, epoch, split):
    """Load existing metrics from a previous evaluation run."""
    metrics_path = os.path.join(results_dir, 'epoch_{:04d}'.format(epoch), '{}_metrics.json'.format(split))
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(',')]

    # Setup experiment and config (same pattern as main.py)
    exp = Experiment(args.exp_name, mode='test')
    cfg_path = args.cfg if args.cfg else exp.cfg_path
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)

    device = torch.device('cuda')
    runner = Runner(cfg, exp, device)

    # Determine epoch range
    available = find_available_epochs(exp.models_dirpath)
    if not available:
        print("No checkpoints found in {}".format(exp.models_dirpath))
        return 1

    start_epoch = args.start
    end_epoch = args.end if args.end else max(available)
    epochs = [e for e in available if start_epoch <= e <= end_epoch]
    print("Found {} checkpoints, evaluating epochs {}-{} ({} epochs)".format(
        len(available), start_epoch, end_epoch, len(epochs)))
    print("Splits: {}".format(', '.join(splits)))
    print("Skip existing: {}".format(args.skip_existing))
    print()

    # Collect all metrics: list of (epoch, split, metrics_dict)
    all_results = []

    for i, epoch in enumerate(epochs):
        for split in splits:
            on_val = (split == 'val')
            tag = "[{}/{}] Epoch {:4d} / {}".format(i + 1, len(epochs), epoch, split)

            # Skip if already evaluated
            if args.skip_existing and has_existing_result(exp.results_dirpath, epoch, split):
                metrics = load_existing_metrics(exp.results_dirpath, epoch, split)
                if metrics:
                    all_results.append((epoch, split, metrics))
                    print("{} -- SKIPPED (existing) F1={:.4f}".format(tag, metrics.get('F1', 0)))
                    continue

            print("{} -- evaluating...".format(tag))
            metrics = runner._eval(epoch, on_val=on_val)
            all_results.append((epoch, split, metrics))
            print("{} -- F1={:.4f}  Precision={:.4f}  Recall={:.4f}".format(
                tag, metrics.get('F1', 0), metrics.get('Precision', 0), metrics.get('Recall', 0)))

    # Write summary CSV
    if all_results:
        csv_path = os.path.join(exp.exp_dirpath, 'eval_summary.csv')
        # Determine all metric keys from the first result
        metric_keys = sorted({k for _, _, m in all_results for k in m.keys()})
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'split'] + metric_keys)
            for epoch, split, metrics in sorted(all_results, key=lambda x: (x[0], x[1])):
                row = [epoch, split] + [metrics.get(k, '') for k in metric_keys]
                writer.writerow(row)
        print("\nSummary saved to: {}".format(csv_path))

        # Print best epochs per split
        print("\n" + "=" * 60)
        for split in splits:
            split_results = [(e, m) for e, s, m in all_results if s == split and 'F1' in m]
            if split_results:
                best_epoch, best_metrics = max(split_results, key=lambda x: x[1]['F1'])
                print("Best {} F1: {:.4f} @ epoch {} (Precision={:.4f}, Recall={:.4f})".format(
                    split, best_metrics['F1'], best_epoch,
                    best_metrics['Precision'], best_metrics['Recall']))
        print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
