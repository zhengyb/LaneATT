#!/usr/bin/env python3
"""Run ablation experiments: train -> test -> collect, with phase-level orchestration.

Usage:
    # Run a single experiment (train + test + collect)
    python3 scripts/run_ablation.py run ablation_bs8_lr0003 --group A2 --phase 0

    # Run all experiments in a phase
    python3 scripts/run_ablation.py phase0
    python3 scripts/run_ablation.py phase1 --lr-base 0.0003
    python3 scripts/run_ablation.py phase2 --lr-base 0.0003
    python3 scripts/run_ablation.py phase3 --lr-base 0.0003 --bs16-winner ablation_bs16_linear
"""
import argparse
import json
import math
import os
import subprocess
import sys


ABLATION_EPOCHS = 15
CFG_DIR = 'cfgs/ablation'

PHASE0_EXPERIMENTS = [
    ('ablation_bs8_lr0001', 'A1'),
    ('ablation_bs8_lr0003', 'A2'),
    ('ablation_bs8_lr001', 'A3'),
]

PHASE1_EXPERIMENTS = [
    ('ablation_bs16_linear', 'B'),
    ('ablation_bs16_sqrt', 'C'),
]

PHASE2_EXPERIMENTS = [
    ('ablation_bs16_linear_warmup', 'D'),
    ('ablation_bs16_sqrt_warmup', 'E'),
]

PHASE3_EXPERIMENTS = [
    ('ablation_bs32_linear', 'F'),
    ('ablation_bs32_sqrt', 'G'),
    ('ablation_bs32_linear_warmup', 'H'),
]


def run_cmd(cmd, desc=None):
    """Run a shell command, print it, and return the exit code."""
    if desc:
        print(f'\n{"="*60}')
        print(f'  {desc}')
        print(f'{"="*60}')
    print(f'$ {cmd}')
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'WARNING: command exited with code {result.returncode}')
    return result.returncode


def check_config_exists(exp_name):
    cfg_path = os.path.join(CFG_DIR, f'{exp_name}.yml')
    if not os.path.exists(cfg_path):
        print(f'Error: config not found: {cfg_path}')
        print(f'Run scripts/generate_configs.py first.')
        return None
    return cfg_path


def run_single(exp_name, group, phase, epoch=ABLATION_EPOCHS):
    """Run train -> test -> collect for a single experiment."""
    cfg_path = check_config_exists(exp_name)
    if cfg_path is None:
        return 1

    # Train
    rc = run_cmd(
        f'python main.py train --exp_name {exp_name} --cfg {cfg_path}',
        f'Training {group}: {exp_name}'
    )
    if rc != 0:
        print(f'Training failed for {exp_name}, skipping test and collect.')
        return rc

    # Test
    run_cmd(
        f'python main.py test --exp_name {exp_name} --epoch {epoch}',
        f'Testing {group}: {exp_name} @ epoch {epoch}'
    )

    # Collect
    run_cmd(
        f'python3 scripts/collect_result.py {exp_name} --group {group} --phase {phase}',
        f'Collecting results for {group}: {exp_name}'
    )

    return 0


def run_compare_pick_best(experiments):
    """Run compare_results.py --pick-best and return the winner name."""
    names = ' '.join(name for name, _ in experiments)
    print(f'\n{"="*60}')
    print(f'  Comparing Phase 0 results (pick best)')
    print(f'{"="*60}')
    cmd = f'python3 scripts/compare_results.py --pick-best {names}'
    print(f'$ {cmd}')
    subprocess.run(cmd, shell=True)

    # Also try to extract the winner programmatically
    try:
        sys.path.insert(0, 'scripts')
        from compare_results import load_result, get_max_epoch, get_loss_avg, pick_best
        results = []
        for name, _ in experiments:
            r = load_result(name)
            if r:
                results.append(r)
        if results:
            max_ep = max(get_max_epoch(r) for r in results)
            winner_name, _ = pick_best(results, max_ep)
            return winner_name
    except Exception as e:
        print(f'Warning: could not auto-detect winner: {e}')
    return None


def run_compare_baseline(baseline, experiments):
    """Run compare_results.py --baseline and return list of (name, verdict)."""
    names = ' '.join(name for name, _ in experiments)
    print(f'\n{"="*60}')
    print(f'  Comparing against baseline: {baseline}')
    print(f'{"="*60}')
    cmd = f'python3 scripts/compare_results.py --baseline {baseline} {names}'
    print(f'$ {cmd}')
    subprocess.run(cmd, shell=True)

    # Extract verdicts programmatically
    verdicts = []
    try:
        sys.path.insert(0, 'scripts')
        from compare_results import load_result, get_max_epoch, judge_vs_baseline
        bl = load_result(baseline)
        if bl:
            max_ep = get_max_epoch(bl)
            for name, group in experiments:
                r = load_result(name)
                if r:
                    v = judge_vs_baseline(r, bl, max_ep)
                    verdicts.append((name, group, v))
    except Exception as e:
        print(f'Warning: could not auto-extract verdicts: {e}')
    return verdicts


def cmd_run(args):
    """Run a single experiment."""
    return run_single(args.exp_name, args.group, args.phase, args.epoch)


def cmd_phase0(args):
    """Run Phase 0: lr search."""
    print('Phase 0: lr search at bs=8')
    print(f'Experiments: {", ".join(n for n, _ in PHASE0_EXPERIMENTS)}')

    for name, group in PHASE0_EXPERIMENTS:
        rc = run_single(name, group, phase=0)
        if rc != 0:
            print(f'\n{name} failed. Continuing with remaining experiments...')

    winner = run_compare_pick_best(PHASE0_EXPERIMENTS)
    if winner:
        print(f'\n>>> Phase 0 winner: {winner}')
        print(f'>>> Next step: run scripts/generate_configs.py --base <base.yml> --lr-base <lr>')
        print(f'>>>            then: python3 scripts/run_ablation.py phase1 --lr-base <lr>')

    return 0


def cmd_phase1(args):
    """Run Phase 1: bs=16 scaling."""
    if args.lr_base is None:
        print('Error: --lr-base required for Phase 1')
        return 1

    # Check configs exist
    for name, _ in PHASE1_EXPERIMENTS:
        if check_config_exists(name) is None:
            return 1

    # Determine baseline (Phase 0 winner)
    baseline = _find_phase0_winner()
    if not baseline:
        print('Error: no Phase 0 results found. Run phase0 first.')
        return 1
    print(f'Phase 1: bs=16 scaling (baseline={baseline})')

    for name, group in PHASE1_EXPERIMENTS:
        run_single(name, group, phase=1)

    verdicts = run_compare_baseline(baseline, PHASE1_EXPERIMENTS)

    # Decision logic
    on_par = [(n, g) for n, g, v in verdicts if v == 'on_par']
    if on_par:
        winner = on_par[0]
        print(f'\n>>> Phase 1 winner: {winner[1]} ({winner[0]}) — verdict: on_par')
        print(f'>>> Can proceed to Phase 3. Or run Phase 2 for additional data.')
    else:
        print(f'\n>>> No on_par result in Phase 1.')
        osc = [(n, g) for n, g, v in verdicts if v == 'oscillation']
        slow = [(n, g) for n, g, v in verdicts if v == 'slow']
        if osc:
            print(f'>>> B oscillated — run Phase 2 (D: linear+warmup)')
        if slow:
            print(f'>>> Found slow convergence — run Phase 2 (E: sqrt+warmup)')
        if not osc and not slow:
            print(f'>>> Results marginal — run Phase 2 for both D and E.')
        print(f'>>> Command: python3 scripts/run_ablation.py phase2 --lr-base {args.lr_base}')

    return 0


def cmd_phase2(args):
    """Run Phase 2: bs=16 + warmup."""
    if args.lr_base is None:
        print('Error: --lr-base required for Phase 2')
        return 1

    for name, _ in PHASE2_EXPERIMENTS:
        if check_config_exists(name) is None:
            return 1

    baseline = _find_phase0_winner()
    if not baseline:
        print('Error: no Phase 0 results found.')
        return 1
    print(f'Phase 2: bs=16 + warmup (baseline={baseline})')

    for name, group in PHASE2_EXPERIMENTS:
        run_single(name, group, phase=2)

    verdicts = run_compare_baseline(baseline, PHASE2_EXPERIMENTS)

    on_par = [(n, g) for n, g, v in verdicts if v == 'on_par']
    if on_par:
        winner = on_par[0]
        print(f'\n>>> Phase 2 winner: {winner[1]} ({winner[0]}) — verdict: on_par')
        print(f'>>> Can proceed to Phase 3.')
    else:
        print(f'\n>>> No on_par result in Phase 2 either.')
        print(f'>>> Recommendation: stay at bs=8, or try bs=12 manually.')

    return 0


def cmd_phase3(args):
    """Run Phase 3: bs=32 scaling."""
    if args.lr_base is None:
        print('Error: --lr-base required for Phase 3')
        return 1

    for name, _ in PHASE3_EXPERIMENTS:
        if check_config_exists(name) is None:
            return 1

    if args.bs16_winner:
        bs16_baseline = args.bs16_winner
    else:
        bs16_baseline = _find_bs16_winner()
    if not bs16_baseline:
        print('Error: no bs=16 winner found. Specify --bs16-winner.')
        return 1

    print(f'Phase 3: bs=32 scaling (baseline={bs16_baseline})')

    for name, group in PHASE3_EXPERIMENTS:
        run_single(name, group, phase=3)

    verdicts = run_compare_baseline(bs16_baseline, PHASE3_EXPERIMENTS)

    on_par = [(n, g) for n, g, v in verdicts if v == 'on_par']
    if on_par:
        winner = on_par[0]
        print(f'\n>>> Phase 3 winner: {winner[1]} ({winner[0]}) — verdict: on_par')
        print(f'>>> Adopt bs=32 for production training.')
    else:
        print(f'\n>>> No on_par result at bs=32.')
        print(f'>>> Recommendation: use bs=16 winner ({bs16_baseline}) for production.')

    return 0


def _find_phase0_winner():
    """Find Phase 0 winner from existing ablation_result.json files."""
    try:
        sys.path.insert(0, 'scripts')
        from compare_results import load_result, get_max_epoch, pick_best
        results = []
        for name, _ in PHASE0_EXPERIMENTS:
            r = load_result(name)
            if r:
                results.append(r)
        if not results:
            return None
        max_ep = max(get_max_epoch(r) for r in results)
        winner_name, _ = pick_best(results, max_ep)
        return winner_name
    except Exception:
        return None


def _find_bs16_winner():
    """Find best bs=16 experiment from Phase 1 and Phase 2 results."""
    try:
        sys.path.insert(0, 'scripts')
        from compare_results import load_result, get_max_epoch, judge_vs_baseline
        baseline_name = _find_phase0_winner()
        if not baseline_name:
            return None
        bl = load_result(baseline_name)
        if not bl:
            return None
        max_ep = get_max_epoch(bl)

        all_bs16 = PHASE1_EXPERIMENTS + PHASE2_EXPERIMENTS
        for name, group in all_bs16:
            r = load_result(name)
            if r and judge_vs_baseline(r, bl, max_ep) == 'on_par':
                return name
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', help='Command')
    sub.required = True

    # run
    p_run = sub.add_parser('run', help='Run a single experiment')
    p_run.add_argument('exp_name', help='Experiment name')
    p_run.add_argument('--group', required=True, help='Group label (A1, B, etc.)')
    p_run.add_argument('--phase', type=int, required=True, help='Phase number 0-3')
    p_run.add_argument('--epoch', type=int, default=ABLATION_EPOCHS, help='Epoch to test')
    p_run.set_defaults(func=cmd_run)

    # phase0
    p0 = sub.add_parser('phase0', help='Run Phase 0: lr search')
    p0.set_defaults(func=cmd_phase0)

    # phase1
    p1 = sub.add_parser('phase1', help='Run Phase 1: bs=16 scaling')
    p1.add_argument('--lr-base', type=float, help='lr_base from Phase 0')
    p1.set_defaults(func=cmd_phase1)

    # phase2
    p2 = sub.add_parser('phase2', help='Run Phase 2: bs=16 + warmup')
    p2.add_argument('--lr-base', type=float, help='lr_base from Phase 0')
    p2.set_defaults(func=cmd_phase2)

    # phase3
    p3 = sub.add_parser('phase3', help='Run Phase 3: bs=32 scaling')
    p3.add_argument('--lr-base', type=float, help='lr_base from Phase 0')
    p3.add_argument('--bs16-winner', default=None,
                    help='bs=16 winner experiment name (auto-detected if omitted)')
    p3.set_defaults(func=cmd_phase3)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    exit(main())
