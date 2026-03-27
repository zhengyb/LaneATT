#!/usr/bin/env python3
"""Generate ablation experiment config files from a base config.

Phase 0 (lr search):
    python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml

Phase 1-3 (batch size scaling, after lr_base is determined):
    python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml --lr-base 0.0003

Output:
    cfgs/ablation/ablation_*.yml
"""
import argparse
import math
import os
import sys

import yaml


# ---------------------------------------------------------------------------
# Experiment definitions from hyperparameter_tuning.md
# ---------------------------------------------------------------------------

ABLATION_EPOCHS = 15
VAL_EVERY = 999        # Disable mid-training validation (avoids checkpoint mismatch crash)
CHECKPOINT_INTERVAL = 1

PHASE0_LRS = [0.0001, 0.0003, 0.001]
PHASE0_NAMES = ['ablation_bs8_lr0001', 'ablation_bs8_lr0003', 'ablation_bs8_lr001']
PHASE0_GROUPS = ['A1', 'A2', 'A3']

# (name, group, batch_size, lr_multiplier, lr_scaling, warmup_epochs)
PHASE1_DEFS = [
    ('ablation_bs16_linear', 'B', 16, 2.0, 'linear', 0),
    ('ablation_bs16_sqrt', 'C', 16, math.sqrt(2), 'sqrt', 0),
]
PHASE2_DEFS = [
    ('ablation_bs16_linear_warmup', 'D', 16, 2.0, 'linear', 3),
    ('ablation_bs16_sqrt_warmup', 'E', 16, math.sqrt(2), 'sqrt', 3),
]
PHASE3_DEFS = [
    ('ablation_bs32_linear', 'F', 32, 4.0, 'linear', 0),
    ('ablation_bs32_sqrt', 'G', 32, 2.0, 'sqrt', 0),
    ('ablation_bs32_linear_warmup', 'H', 32, 4.0, 'linear', 5),
]


def load_base_config(path):
    """Load base YAML config with FullLoader (supports !!python/tuple)."""
    with open(path) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_dataset_size(cfg):
    """Instantiate the train dataset and return its length."""
    # Add project root to path so lib.* imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from lib.config import Config as LaneConfig

    # Write a temp config to load through the project's Config class
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp_path = tmp.name
    try:
        lcfg = LaneConfig(tmp_path)
        ds = lcfg.get_dataset('train')
        return len(ds)
    finally:
        os.unlink(tmp_path)


def compute_tmax(dataset_size, batch_size, epochs):
    return epochs * math.ceil(dataset_size / batch_size)


def build_ablation_cfg(base_cfg, batch_size, lr, epochs, tmax, warmup_epochs=0):
    """Create an ablation config dict by overriding the base config."""
    cfg = dict(base_cfg)  # shallow copy of top level

    cfg['batch_size'] = batch_size
    cfg['epochs'] = epochs
    cfg['val_every'] = VAL_EVERY
    cfg['model_checkpoint_interval'] = CHECKPOINT_INTERVAL

    cfg['optimizer'] = {
        'name': 'Adam',
        'parameters': {'lr': round(lr, 10)}
    }
    cfg['lr_scheduler'] = {
        'name': 'CosineAnnealingLR',
        'parameters': {'T_max': tmax}
    }

    if warmup_epochs > 0:
        cfg['warmup_epochs'] = warmup_epochs

    return cfg


def write_config(cfg, path, header_comment=''):
    """Write config dict to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        if header_comment:
            for line in header_comment.strip().split('\n'):
                f.write(f'# {line}\n')
            f.write('\n')
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation config files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Phase 0 only (lr search):
  python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml

  # All phases (after lr_base determined):
  python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml --lr-base 0.0003

  # Skip dataset loading, provide N manually:
  python3 scripts/generate_configs.py --base cfgs/laneatt_tusimple_resnet18.yml -N 100000
""")
    parser.add_argument('--base', required=True, help='Path to base config YAML')
    parser.add_argument('--lr-base', type=float, default=None,
                        help='lr_base from Phase 0. If omitted, only Phase 0 configs are generated.')
    parser.add_argument('-N', type=int, default=None,
                        help='Dataset size (skip auto-detection)')
    parser.add_argument('--outdir', default='cfgs/ablation',
                        help='Output directory (default: cfgs/ablation)')
    args = parser.parse_args()

    base_cfg = load_base_config(args.base)

    # --- Determine dataset size ---
    if args.N:
        N = args.N
        print(f'Using provided dataset size: N = {N}')
    else:
        print('Loading dataset to determine size (this may take a moment)...')
        N = get_dataset_size(base_cfg)
        print(f'Dataset size: N = {N}')

    print()

    # --- Phase 0: lr search at bs=8 ---
    print('=== Phase 0: lr search (bs=8) ===')
    generated = []
    for name, group, lr in zip(PHASE0_NAMES, PHASE0_GROUPS, PHASE0_LRS):
        bs = 8
        tmax = compute_tmax(N, bs, ABLATION_EPOCHS)
        cfg = build_ablation_cfg(base_cfg, bs, lr, ABLATION_EPOCHS, tmax)
        path = os.path.join(args.outdir, f'{name}.yml')
        comment = f'Ablation {group}: bs={bs}, lr={lr}, T_max={tmax}'
        write_config(cfg, path, comment)
        generated.append((group, name, bs, lr, tmax, 0))
        print(f'  {group} -> {path}  (bs={bs}, lr={lr}, T_max={tmax})')

    # --- Phase 1-3: batch size scaling (requires --lr-base) ---
    if args.lr_base is None:
        print('\n--lr-base not provided. Phase 1-3 configs not generated.')
        print('Run Phase 0 experiments first, then re-run with --lr-base <value>.')
    else:
        lr_base = args.lr_base
        for phase_num, phase_name, defs in [
            (1, 'Phase 1: bs=16 scaling', PHASE1_DEFS),
            (2, 'Phase 2: bs=16 + warmup', PHASE2_DEFS),
            (3, 'Phase 3: bs=32 scaling', PHASE3_DEFS),
        ]:
            print(f'\n=== {phase_name} (lr_base={lr_base}) ===')
            for name, group, bs, mult, scaling, warmup_ep in defs:
                lr = lr_base * mult
                tmax = compute_tmax(N, bs, ABLATION_EPOCHS)
                cfg = build_ablation_cfg(base_cfg, bs, lr, ABLATION_EPOCHS, tmax, warmup_ep)
                path = os.path.join(args.outdir, f'{name}.yml')
                warmup_str = f', warmup={warmup_ep}ep' if warmup_ep else ''
                comment = (f'Ablation {group}: bs={bs}, lr={lr:.6f} '
                           f'(lr_base={lr_base} x{mult:.3f} {scaling}), '
                           f'T_max={tmax}{warmup_str}')
                write_config(cfg, path, comment)
                generated.append((group, name, bs, lr, tmax, warmup_ep))
                print(f'  {group} -> {path}  (bs={bs}, lr={lr:.6f}{warmup_str}, T_max={tmax})')

    # --- Summary ---
    print(f'\n--- Summary ---')
    print(f'Dataset size N = {N}')
    print(f'Generated {len(generated)} config(s) in {args.outdir}/')
    print(f'\n{"Group":<6} {"Name":<35} {"bs":>3} {"lr":>10} {"T_max":>10} {"warmup":>7}')
    print('-' * 75)
    for group, name, bs, lr, tmax, warmup in generated:
        wu = f'{warmup}ep' if warmup else '-'
        print(f'{group:<6} {name:<35} {bs:>3} {lr:>10.6f} {tmax:>10} {wu:>7}')

    return 0


if __name__ == '__main__':
    exit(main())
