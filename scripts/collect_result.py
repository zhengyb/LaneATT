#!/usr/bin/env python3
"""Collect ablation experiment results into a structured JSON file.

Usage:
    python3 scripts/collect_result.py <exp_name> --group A2 --phase 0
    python3 scripts/collect_result.py <exp_name> --group B  --phase 1 --peak-vram-mb 6200

Output:
    experiments/<exp_name>/ablation_result.json
"""
import argparse
import json
import math
import os
import re
import yaml
from collections import defaultdict
from datetime import datetime

# Maximum reasonable per-epoch duration: 2 hours.
# Anything longer is a sign of multi-session log contamination.
MAX_EPOCH_DURATION_SEC = 7200


def parse_training_log(log_path):
    """Parse log_train.txt, return per-epoch statistics.

    Handles multi-session logs (from --resume) by capping per-epoch duration
    and deduplicating epoch data (last session wins).
    """
    # Collect all iterations keyed by epoch.
    # If a log has multiple sessions (resume), the same epoch may appear
    # with different timestamps; we keep only the LAST block of iterations
    # for each epoch.
    raw_epochs = defaultdict(list)  # epoch -> list of (ts, loss, cls, reg, lr)

    pattern = re.compile(
        r'\[([^\]]+)\].*Epoch \[(\d+)/(\d+)\] - Iter \[(\d+)/(\d+)\] - '
        r'Loss: ([\d.eE+nan\-inf]+) - cls_loss: ([\d.eE+nan\-inf]+) - '
        r'reg_loss: ([\d.eE+nan\-inf]+) - batch_positives: [\d.]+ - lr: ([\d.eE+\-]+)'
    )

    # Track session boundaries: a "Beginning training session" line resets
    session_boundary_pattern = re.compile(r'Beginning training session')
    current_session_start = {}  # epoch -> index in raw_epochs[ep]

    with open(log_path) as f:
        for line in f:
            if session_boundary_pattern.search(line):
                # Mark that a new session started; any epoch data after this
                # supersedes earlier data for the same epoch
                current_session_start.clear()

            m = pattern.search(line)
            if m:
                ts, epoch_s, _, iter_s, max_iter_s, \
                    loss_s, cls_s, reg_s, lr_s = m.groups()
                ep = int(epoch_s)
                it = int(iter_s)

                # If this is iter 0 for this epoch in a new session,
                # discard previous data for this epoch
                if it == 0 and ep not in current_session_start:
                    raw_epochs[ep] = []
                    current_session_start[ep] = True

                raw_epochs[ep].append((ts, float(loss_s), float(cls_s),
                                       float(reg_s), float(lr_s)))

    result = []
    for ep in sorted(raw_epochs.keys()):
        entries = raw_epochs[ep]
        if not entries:
            continue

        losses = [e[1] for e in entries]
        cls_losses = [e[2] for e in entries]
        reg_losses = [e[3] for e in entries]
        lrs = [e[4] for e in entries]
        ts_list = [e[0] for e in entries]

        # Duration from first to last iteration timestamp
        duration_sec = None
        try:
            fmt = '%Y-%m-%d %H:%M:%S,%f'
            t0 = datetime.strptime(ts_list[0], fmt)
            t1 = datetime.strptime(ts_list[-1], fmt)
            d = round((t1 - t0).total_seconds())
            # Cap at MAX_EPOCH_DURATION_SEC to filter multi-session artifacts
            if 0 <= d <= MAX_EPOCH_DURATION_SEC:
                duration_sec = d
        except (ValueError, IndexError):
            pass

        # Check for NaN / Inf
        has_nan = any(math.isnan(v) or math.isinf(v) for v in losses)

        avg = lambda vals: round(sum(vals) / len(vals), 5)
        result.append({
            'epoch': ep,
            'loss_avg': avg(losses) if not has_nan else None,
            'cls_loss_avg': avg(cls_losses) if not has_nan else None,
            'reg_loss_avg': avg(reg_losses) if not has_nan else None,
            'lr_start': lrs[0],
            'lr_end': lrs[-1],
            'num_iters': len(losses),
            'duration_sec': duration_sec,
            'has_nan': has_nan
        })

    return result


def load_config(exp_dir):
    """Load experiment config.yaml as dict."""
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_eval_metrics(exp_dir):
    """Load all evaluation metrics from results/ subdirectories."""
    results_dir = os.path.join(exp_dir, 'results')
    if not os.path.exists(results_dir):
        return {}

    metrics = {}
    for epoch_dir in sorted(os.listdir(results_dir)):
        m = re.match(r'epoch_(\d+)', epoch_dir)
        if not m:
            continue
        ep = int(m.group(1))
        epoch_path = os.path.join(results_dir, epoch_dir)
        for fname in os.listdir(epoch_path):
            if fname.endswith('_metrics.json'):
                split = fname.replace('_metrics.json', '')
                with open(os.path.join(epoch_path, fname)) as f:
                    metrics.setdefault(ep, {})[split] = json.load(f)

    return metrics


def check_oscillation(epoch_data, start_ep=1, end_ep=5, threshold=1.05):
    """Check epochs [start_ep, end_ep] for loss_avg increase > threshold."""
    relevant = sorted(
        [e for e in epoch_data
         if start_ep <= e['epoch'] <= end_ep and e['loss_avg'] is not None],
        key=lambda x: x['epoch']
    )

    hits = []
    for i in range(len(relevant) - 1):
        curr, nxt = relevant[i], relevant[i + 1]
        if curr['loss_avg'] <= 0:
            continue
        ratio = nxt['loss_avg'] / curr['loss_avg']
        if ratio > threshold:
            hits.append({
                'from_epoch': curr['epoch'],
                'to_epoch': nxt['epoch'],
                'loss_from': curr['loss_avg'],
                'loss_to': nxt['loss_avg'],
                'ratio': round(ratio, 4)
            })

    return len(hits) > 0, hits


def fmt_duration(total_sec):
    """Format seconds into a human-readable string."""
    if total_sec is None:
        return 'N/A'
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    if h > 0:
        return f'{h}h {m}m {s}s'
    return f'{m}m {s}s'


def main():
    parser = argparse.ArgumentParser(description='Collect ablation results into JSON')
    parser.add_argument('exp_name', help='Experiment name (directory under experiments/)')
    parser.add_argument('--group', default='', help='Group label, e.g. A1, B, C')
    parser.add_argument('--phase', type=int, default=-1, help='Phase number 0-3')
    parser.add_argument('--peak-vram-mb', type=int, default=None,
                        help='Peak GPU VRAM in MB (manual, from nvidia-smi)')
    args = parser.parse_args()

    exp_dir = os.path.join('experiments', args.exp_name)
    if not os.path.exists(exp_dir):
        print(f'Error: {exp_dir} not found')
        return 1

    # --- Parse training log ---
    log_path = os.path.join(exp_dir, 'log_train.txt')
    if os.path.exists(log_path):
        epoch_data = parse_training_log(log_path)
        diverged = any(e['has_nan'] for e in epoch_data)
        status = 'diverged' if diverged else 'completed'
    else:
        epoch_data = []
        status = 'no_log'

    # --- Load config ---
    cfg = load_config(exp_dir)

    # --- Load evaluation metrics ---
    eval_metrics = load_eval_metrics(exp_dir)

    # --- Oscillation check ---
    is_osc, osc_detail = check_oscillation(epoch_data)

    # --- Total training duration ---
    total_sec = None
    if epoch_data:
        durations = [e['duration_sec'] for e in epoch_data if e['duration_sec'] is not None]
        if durations:
            total_sec = sum(durations)

    # --- Warmup epochs from config ---
    warmup_epochs = cfg.get('warmup_epochs', 0)

    # --- Build result ---
    result = {
        'experiment': {
            'name': args.exp_name,
            'group': args.group,
            'phase': args.phase
        },
        'config': {
            'batch_size': cfg.get('batch_size'),
            'lr': (cfg.get('optimizer') or {}).get('parameters', {}).get('lr'),
            'warmup_epochs': warmup_epochs,
            'epochs': cfg.get('epochs'),
            'T_max': (cfg.get('lr_scheduler') or {}).get('parameters', {}).get('T_max')
        },
        'environment': {
            'iters_per_epoch': epoch_data[0]['num_iters'] if epoch_data else None,
            'peak_vram_mb': args.peak_vram_mb
        },
        'training': {
            'status': status,
            'total_duration_sec': total_sec,
            'epochs': epoch_data
        },
        'evaluation': {str(k): v for k, v in eval_metrics.items()},
        'judgment': {
            'oscillation': is_osc,
            'oscillation_detail': osc_detail
        }
    }

    # --- Write JSON ---
    out_path = os.path.join(exp_dir, 'ablation_result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'Saved: {out_path}')

    # --- Print summary ---
    if epoch_data:
        last = epoch_data[-1]
        print(f'\n  status        : {status}')
        print(f'  epochs        : {len(epoch_data)}')
        print(f'  loss_avg(ep1) : {epoch_data[0]["loss_avg"]}')
        if len(epoch_data) >= 5:
            print(f'  loss_avg(ep5) : {epoch_data[4]["loss_avg"]}')
        print(f'  loss_avg(ep{last["epoch"]:<2}): {last["loss_avg"]}')
        print(f'  oscillation   : {"YES" if is_osc else "NO"}')
        for d in osc_detail:
            print(f'    ep{d["from_epoch"]}->{d["to_epoch"]}: '
                  f'{d["loss_from"]:.5f} -> {d["loss_to"]:.5f} (x{d["ratio"]})')
        print(f'  total time    : {fmt_duration(total_sec)}')

    return 0


if __name__ == '__main__':
    exit(main())
