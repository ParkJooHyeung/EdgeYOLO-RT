#!/usr/bin/env python3
"""Run experiment presets from scripts/experiments.yaml and collect baseline metrics.

Usage:
  python scripts/run_experiments.py --preset default

This runner invokes `scripts/baseline_yolov8.py` for each preset and saves results under
`results/<preset>_baseline.csv`.
"""
import argparse
import os
import subprocess
import yaml

HERE = os.path.dirname(__file__)


def load_presets(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def summarize_csv(csv_path):
    import csv
    vals = {'pre': [], 'inf': [], 'post': [], 'total': []}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if not r or r[0] == '' or r[0].lower() == 'avg':
                    continue
                try:
                    vals['pre'].append(float(r[1]))
                    vals['inf'].append(float(r[2]))
                    vals['post'].append(float(r[3]))
                    vals['total'].append(float(r[4]))
                except Exception:
                    continue
    except FileNotFoundError:
        return None
    summary = {}
    for k, arr in vals.items():
        summary[k] = sum(arr) / len(arr) if arr else None
    return summary


def run_baseline(preset_name, cfg, override_max=None):
    out_dir = os.path.join(HERE, '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'{preset_name}_baseline.csv')
    imgsz = cfg.get('imgsz', [640, 640])
    max_images = override_max if override_max is not None else cfg.get('max_images', 100)
    cmd = ["./venv/bin/python", os.path.join(HERE, 'baseline_yolov8.py'),
           '--model', os.path.join(HERE, '..', 'yolov8n.pt'),
           '--images', os.path.join(HERE, '..', 'images'),
           '--max_images', str(max_images),
           '--size', str(imgsz[0]), str(imgsz[1]),
           '--out', out_csv]
    print('Running:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print('Baseline run failed (this can happen if dependencies are missing).')
    # Summarize results if CSV produced
    summary = summarize_csv(out_csv)
    if summary is not None:
        summary_path = os.path.join(out_dir, f'{preset_name}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"preset: {preset_name}\n")
            f.write(f"max_images: {max_images}\n")
            for k, v in summary.items():
                f.write(f"{k}_ms_avg: {v if v is not None else 'N/A'}\n")
        print('Wrote summary to', summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default=None, help='Preset name from scripts/experiments.yaml')
    args = parser.parse_args()

    presets = load_presets(os.path.join(HERE, 'experiments.yaml'))
    if args.preset:
        if args.preset not in presets:
            print('Preset not found:', args.preset)
            return
        run_baseline(args.preset, presets[args.preset])
    else:
        for name, cfg in presets.items():
            print('\n=== Running preset:', name)
            run_baseline(name, cfg)


if __name__ == '__main__':
    main()
