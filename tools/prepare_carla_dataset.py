#!/usr/bin/env python3
"""
Carla dataset preprocessor for LaneATT.

Reads annotation files from multiple datasets/tusimple-* directories,
rewrites raw_file paths to be relative to the datasets/ root,
and outputs new annotation files to a single output directory.

Original data is NOT modified. No images are copied or moved.

Usage:
    python tools/prepare_carla_dataset.py --datasets_dir datasets --output_dir datasets/carla_tusimple --verify
"""

import os
import sys
import json
import glob
import random
import argparse


def find_carla_datasets(datasets_dir):
    """Find all tusimple-* directories that contain tusimple_merged/."""
    datasets = []
    for entry in sorted(os.listdir(datasets_dir)):
        if not entry.startswith('tusimple-'):
            continue
        merged_dir = os.path.join(datasets_dir, entry, 'tusimple_merged')
        if os.path.isdir(merged_dir):
            datasets.append(entry)
    return datasets


def find_anno_files(merged_dir):
    """Find train/val/test annotation files in tusimple_merged/."""
    result = {'train': [], 'val': [], 'test': []}
    for fname in sorted(os.listdir(merged_dir)):
        if not fname.endswith('.json'):
            continue
        for split in result:
            if fname.startswith(split):
                result[split].append(fname)
                break
    return result


def process_anno_file(input_path, output_path, dataset_name):
    """Rewrite raw_file paths and write to output file.

    Returns the number of annotations processed.
    """
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            anno = json.loads(line)
            # Prepend dataset directory name to raw_file
            anno['raw_file'] = os.path.join(dataset_name, anno['raw_file'])
            fout.write(json.dumps(anno) + '\n')
            count += 1
    return count


def verify_paths(output_dir, datasets_dir, num_samples=10):
    """Randomly sample annotations from output files and check image paths exist."""
    all_files = glob.glob(os.path.join(output_dir, '*.json'))
    all_files = [f for f in all_files if not f.endswith('prepare_report.json')]

    all_annos = []
    for fpath in all_files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_annos.append((os.path.basename(fpath), json.loads(line)))

    if not all_annos:
        print("  No annotations to verify.")
        return

    samples = random.sample(all_annos, min(num_samples, len(all_annos)))
    ok = 0
    fail = 0
    for src_file, anno in samples:
        img_path = os.path.join(datasets_dir, anno['raw_file'])
        exists = os.path.isfile(img_path)
        status = "OK" if exists else "MISSING"
        if exists:
            ok += 1
        else:
            fail += 1
        print(f"  [{status}] {src_file} -> {img_path}")

    print(f"  Verification: {ok}/{ok + fail} paths exist.")
    if fail > 0:
        print("  WARNING: Some image paths are missing!")


def main():
    parser = argparse.ArgumentParser(description="Prepare Carla datasets for LaneATT")
    parser.add_argument('--datasets_dir', default='datasets',
                        help='Root directory containing tusimple-* datasets')
    parser.add_argument('--output_dir', default='datasets/carla_tusimple',
                        help='Output directory for generated annotation files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify sample image paths after processing')
    args = parser.parse_args()

    datasets_dir = args.datasets_dir
    output_dir = args.output_dir

    if not os.path.isdir(datasets_dir):
        print(f"Error: datasets directory '{datasets_dir}' not found.")
        sys.exit(1)

    # Find datasets
    datasets = find_carla_datasets(datasets_dir)
    if not datasets:
        print(f"No tusimple-* datasets with tusimple_merged/ found in '{datasets_dir}'.")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s): {datasets}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset
    report = {'datasets': {}, 'totals': {'train': 0, 'val': 0, 'test': 0}}

    for dataset_name in datasets:
        merged_dir = os.path.join(datasets_dir, dataset_name, 'tusimple_merged')
        anno_files = find_anno_files(merged_dir)

        dataset_stats = {'train': 0, 'val': 0, 'test': 0, 'files': []}

        for split, fnames in anno_files.items():
            for fname in fnames:
                input_path = os.path.join(merged_dir, fname)
                output_fname = f"{dataset_name}_{fname}"
                output_path = os.path.join(output_dir, output_fname)

                count = process_anno_file(input_path, output_path, dataset_name)
                dataset_stats[split] += count
                dataset_stats['files'].append(output_fname)
                print(f"  {output_fname}: {count} annotations")

        report['datasets'][dataset_name] = dataset_stats
        for split in ['train', 'val', 'test']:
            report['totals'][split] += dataset_stats[split]

    # Save report
    report_path = os.path.join(output_dir, 'prepare_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\nSummary:")
    print(f"  Train: {report['totals']['train']} annotations")
    print(f"  Val:   {report['totals']['val']} annotations")
    print(f"  Test:  {report['totals']['test']} annotations")
    print(f"  Total: {sum(report['totals'].values())} annotations")
    print(f"  Report saved to: {report_path}")

    # Verify
    if args.verify:
        print(f"\nVerifying image paths...")
        verify_paths(output_dir, datasets_dir)

    # Print usage hint
    print(f"\nUsage in LaneATT config:")
    print(f'  root: "{datasets_dir}"')
    print(f"  SPLIT_FILES entries:")
    for dataset_name in datasets:
        merged_dir = os.path.join(datasets_dir, dataset_name, 'tusimple_merged')
        anno_files = find_anno_files(merged_dir)
        for split, fnames in anno_files.items():
            for fname in fnames:
                rel = os.path.relpath(os.path.join(output_dir, f"{dataset_name}_{fname}"), datasets_dir)
                print(f'    "{split}": "{rel}"')


if __name__ == '__main__':
    main()
