#!/usr/bin/env python3
"""Compare ablation experiment results against a baseline.

Reads ablation_result.json from each experiment, applies quantified judgment
criteria, and outputs a comparison table with verdicts.

Usage:
    python scripts/compare_results.py --baseline ablation_bs8_lr0003 \\
        ablation_bs16_linear ablation_bs16_sqrt

    # Phase 0: compare lr candidates (no baseline, pick best)
    python scripts/compare_results.py --pick-best \\
        ablation_bs8_lr0001 ablation_bs8_lr0003 ablation_bs8_lr001

Output:
    Formatted comparison table to stdout.
    Optional: --save <path> to write JSON summary.
"""
import argparse
import json
import os
import sys


# --- Judgment thresholds (from hyperparameter_tuning.md) ---
LOSS_ON_PAR_THRESHOLD = 1.20   # loss_avg(15) <= baseline * 1.20
F1_ON_PAR_THRESHOLD = 0.95     # F1(15) >= baseline * 0.95
LOSS_SLOW_THRESHOLD = 1.30     # loss_avg(15) > baseline * 1.30
LOSS_CLOSE_THRESHOLD = 0.05    # two losses within 5% are "close"
OSCILLATION_THRESHOLD = 1.05   # epoch-over-epoch increase > 5%


def load_result(exp_name):
    path = os.path.join('experiments', exp_name, 'ablation_result.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_loss_avg(result, epoch):
    for e in result['training']['epochs']:
        if e['epoch'] == epoch:
            return e['loss_avg']
    return None


def get_f1(result, epoch):
    """Get F1 from evaluation data at given epoch."""
    ep_key = str(epoch)
    ev = result.get('evaluation', {}).get(ep_key)
    if ev is None:
        return None
    # evaluation may be {"test": {...}, "val": {...}} or flat {"F1": ...}
    if 'test' in ev:
        return ev['test'].get('F1')
    if 'val' in ev:
        return ev['val'].get('F1')
    return ev.get('F1')


def get_max_epoch(result):
    epochs = result.get('training', {}).get('epochs', [])
    if not epochs:
        return 15
    return max(e['epoch'] for e in epochs)


def judge_vs_baseline(result, baseline, max_epoch):
    """Return verdict string based on quantified criteria."""
    status = result['training']['status']
    if status == 'diverged':
        return 'diverged'
    if status == 'no_log':
        return 'no_data'

    is_osc = result['judgment']['oscillation']
    bl_loss = get_loss_avg(baseline, max_epoch)
    my_loss = get_loss_avg(result, max_epoch)
    bl_f1 = get_f1(baseline, max_epoch)
    my_f1 = get_f1(result, max_epoch)

    if my_loss is None or bl_loss is None or bl_loss <= 0:
        return 'no_data'

    loss_ratio = my_loss / bl_loss

    if is_osc:
        return 'oscillation'
    if loss_ratio > LOSS_SLOW_THRESHOLD:
        return 'slow'

    # on_par requires: loss <= 1.20x, F1 >= 0.95x, no oscillation
    f1_ok = True
    if my_f1 is not None and bl_f1 is not None and bl_f1 > 0:
        f1_ok = (my_f1 / bl_f1) >= F1_ON_PAR_THRESHOLD

    if loss_ratio <= LOSS_ON_PAR_THRESHOLD and f1_ok:
        return 'on_par'

    return 'marginal'


def pick_best(results, max_epoch):
    """Phase-0 logic: select lr_base from candidates."""
    candidates = []
    for r in results:
        name = r['experiment']['name']
        status = r['training']['status']
        is_osc = r['judgment']['oscillation']
        loss = get_loss_avg(r, max_epoch)
        lr = r['config']['lr']

        if status == 'diverged':
            candidates.append((name, lr, loss, 'diverged', False))
            continue
        if loss is None:
            candidates.append((name, lr, None, 'no_data', False))
            continue
        candidates.append((name, lr, loss, 'oscillation' if is_osc else 'stable', not is_osc))

    # Filter: prefer stable, then sort by loss
    stable = [(n, lr, l, s, ok) for n, lr, l, s, ok in candidates if ok and l is not None]
    if not stable:
        # All unstable: pick lowest loss anyway
        valid = [(n, lr, l, s, ok) for n, lr, l, s, ok in candidates if l is not None]
        if not valid:
            return None, candidates
        valid.sort(key=lambda x: x[2])
        return valid[0][0], candidates

    stable.sort(key=lambda x: x[2])

    # If top two are within 5%, pick higher lr
    if len(stable) >= 2:
        loss_0, loss_1 = stable[0][2], stable[1][2]
        if loss_0 > 0 and abs(loss_1 - loss_0) / loss_0 < LOSS_CLOSE_THRESHOLD:
            # Within 5%, pick higher lr
            if stable[1][1] > stable[0][1]:
                return stable[1][0], candidates

    return stable[0][0], candidates


def fmt(val, fmt_str, na='N/A'):
    if val is None:
        return na.rjust(len(fmt_str % 0)) if '%' in fmt_str else na
    return fmt_str % val


def cmd_compare(args):
    baseline = load_result(args.baseline)
    if baseline is None:
        print(f'Error: baseline "{args.baseline}" ablation_result.json not found')
        return 1

    max_epoch = get_max_epoch(baseline)
    bl_loss = get_loss_avg(baseline, max_epoch)
    bl_f1 = get_f1(baseline, max_epoch)

    print(f'\nBaseline: {args.baseline}')
    print(f'  loss_avg(ep{max_epoch}) = {bl_loss}')
    print(f'  F1(ep{max_epoch})       = {bl_f1}')
    print()

    # Table header
    cols = f'{"experiment":<35} {"bs":>3} {"lr":>10} {"loss_avg":>10} {"F1":>8} {"loss/bl":>8} {"osc":>4} {"verdict":>12}'
    print(cols)
    print('-' * len(cols))

    # Baseline row
    osc_bl = 'Y' if baseline['judgment']['oscillation'] else 'N'
    print(f'{args.baseline:<35} {baseline["config"]["batch_size"] or "?":>3} '
          f'{baseline["config"]["lr"] or 0:>10.6f} '
          f'{bl_loss or 0:>10.5f} {bl_f1 or 0:>8.4f} '
          f'{"1.000":>8} {osc_bl:>4} {"baseline":>12}')

    rows = []
    for exp_name in args.experiments:
        r = load_result(exp_name)
        if r is None:
            print(f'{exp_name:<35} {"--result file not found--":>60}')
            continue

        my_loss = get_loss_avg(r, max_epoch)
        my_f1 = get_f1(r, max_epoch)
        ratio = my_loss / bl_loss if (my_loss and bl_loss and bl_loss > 0) else None
        verdict = judge_vs_baseline(r, baseline, max_epoch)
        osc = 'X' if r['training']['status'] == 'diverged' else ('Y' if r['judgment']['oscillation'] else 'N')

        print(f'{exp_name:<35} {r["config"]["batch_size"] or "?":>3} '
              f'{r["config"]["lr"] or 0:>10.6f} '
              f'{my_loss or 0:>10.5f} {my_f1 or 0:>8.4f} '
              f'{ratio or 0:>8.3f} {osc:>4} {verdict:>12}')

        rows.append({
            'experiment': exp_name,
            'loss_avg': my_loss, 'f1': my_f1,
            'loss_ratio': round(ratio, 4) if ratio else None,
            'oscillation': r['judgment']['oscillation'],
            'verdict': verdict
        })

    print()
    print('Verdicts: on_par=pass | oscillation=epoch1-5 loss spike >5% | '
          'slow=loss >1.3x baseline | marginal=between thresholds | '
          'diverged=NaN/Inf')

    if args.save:
        summary = {
            'baseline': args.baseline,
            'max_epoch': max_epoch,
            'baseline_loss': bl_loss,
            'baseline_f1': bl_f1,
            'comparisons': rows
        }
        with open(args.save, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'\nSummary saved to {args.save}')

    return 0


def cmd_pick_best(args):
    results = []
    for exp_name in args.experiments:
        r = load_result(exp_name)
        if r is None:
            print(f'Warning: {exp_name} ablation_result.json not found, skipping')
            continue
        results.append(r)

    if not results:
        print('Error: no valid results to compare')
        return 1

    max_epoch = max(get_max_epoch(r) for r in results)
    best_name, candidates = pick_best(results, max_epoch)

    print(f'\nPhase 0 lr search — pick best (max_epoch={max_epoch})')
    print()
    print(f'{"experiment":<35} {"lr":>10} {"loss_avg":>10} {"status":>12} {"selected":>9}')
    print('-' * 80)
    for name, lr, loss, status, _ in candidates:
        sel = '<-- lr_base' if name == best_name else ''
        print(f'{name:<35} {lr or 0:>10.6f} {loss or 0:>10.5f} {status:>12} {sel:>9}')

    if best_name:
        best_r = next(r for r in results if r['experiment']['name'] == best_name)
        lr_base = best_r['config']['lr']
        print(f'\n  lr_base = {lr_base}')
        print(f'\n  Derived lr values for subsequent phases:')
        import math
        print(f'    B/D (x2)      = {lr_base * 2}')
        print(f'    C/E (xsqrt2)  = {lr_base * math.sqrt(2):.6f}')
        print(f'    F/H (x4)      = {lr_base * 4}')
        print(f'    G   (x2)      = {lr_base * 2}')

    return 0


def main():
    parser = argparse.ArgumentParser(description='Compare ablation results')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline', help='Baseline experiment name (phases 1-3)')
    group.add_argument('--pick-best', action='store_true',
                       help='Phase 0 mode: pick best lr from candidates')
    parser.add_argument('experiments', nargs='+', help='Experiment names to compare')
    parser.add_argument('--save', default=None, help='Save JSON summary to this path')
    args = parser.parse_args()

    if args.pick_best:
        return cmd_pick_best(args)
    else:
        return cmd_compare(args)


if __name__ == '__main__':
    exit(main())
