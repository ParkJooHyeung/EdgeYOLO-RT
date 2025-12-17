#!/usr/bin/env python3
"""
Baseline inference script for YOLOv8 with detailed timing.

Measures per-image Pre-process, Inference, Post-process times
and writes `baseline_result.csv` with per-image and average metrics.

Usage:
  python scripts/baseline_yolov8.py --model yolov8n.pt --images ./images --max_images 1000

Note: install required packages in a virtualenv before running.
"""
import argparse
import glob
import os
import time
import csv
from collections import defaultdict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


def preprocess_image(img, size=(640, 640)):
    # Letterbox resize keeping aspect ratio, then BGR->RGB, return HWC uint8 image
    h0, w0 = img.shape[:2]
    r = min(size[0] / h0, size[1] / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    dy = (size[0] - nh) // 2
    dx = (size[1] - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw, :] = resized
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return img_rgb


def postprocess_predictions(preds):
    # preds: whatever ultralytics returns; this function should be adapted if needed
    boxes = []
    try:
        for p in preds:
            # ultralytics returns results with .boxes.xyxy, .boxes.conf, .boxes.cls
            if hasattr(p, 'boxes'):
                for box in p.boxes:
                    xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy
                    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf)
                    cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls)
                    boxes.append({'xyxy': xyxy, 'conf': conf, 'cls': cls})
    except Exception:
        # Fallback: try iterating top-level preds
        pass
    return boxes


def measure_gpu_mem_mb():
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(0) / 1024 / 1024)
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to YOLOv8 .pt model')
    parser.add_argument('--images', required=True, help='Folder or glob of images')
    parser.add_argument('--out', default='baseline_result.csv', help='CSV output file')
    parser.add_argument('--max_images', type=int, default=1000)
    parser.add_argument('--size', type=int, nargs=2, default=(640, 640))
    args = parser.parse_args()

    if YOLO is None:
        print('Error: ultralytics not installed. Install in your venv: pip install ultralytics')
        return

    model = YOLO(args.model)

    # collect image paths
    if os.path.isdir(args.images):
        img_paths = sorted(glob.glob(os.path.join(args.images, '*.*')))
    else:
        img_paths = sorted(glob.glob(args.images))
    img_paths = img_paths[:args.max_images]

    rows = []
    totals = defaultdict(float)
    count = 0

    for p in img_paths:
        count += 1
        # Preprocess
        t0 = time.perf_counter()
        img = cv2.imread(p)
        input_img = preprocess_image(img, size=tuple(args.size))
        t1 = time.perf_counter()

        # Inference
        inf_start = time.perf_counter()
        # ultralytics accepts numpy arrays too; we time the predict call as inference
        preds = model.predict(source=[input_img], imgsz=args.size, verbose=False)
        inf_end = time.perf_counter()

        # Postprocess
        post_start = time.perf_counter()
        boxes = postprocess_predictions(preds)
        post_end = time.perf_counter()

        preprocess_ms = (t1 - t0) * 1000.0
        inference_ms = (inf_end - inf_start) * 1000.0
        postprocess_ms = (post_end - post_start) * 1000.0
        total_ms = preprocess_ms + inference_ms + postprocess_ms

        gpu_mem = measure_gpu_mem_mb()

        rows.append([os.path.basename(p), f'{preprocess_ms:.3f}', f'{inference_ms:.3f}', f'{postprocess_ms:.3f}', f'{total_ms:.3f}', gpu_mem if gpu_mem is not None else 'N/A'])

        totals['pre'] += preprocess_ms
        totals['inf'] += inference_ms
        totals['post'] += postprocess_ms
        totals['total'] += total_ms

        if count % 20 == 0:
            print(f'Processed {count}/{len(img_paths)} images...')

    # write CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms', 'gpu_mem_mb'])
        writer.writerows(rows)
        # write averages
        if count > 0:
            writer.writerow([])
            writer.writerow(['avg', f"{totals['pre']/count:.3f}", f"{totals['inf']/count:.3f}", f"{totals['post']/count:.3f}", f"{totals['total']/count:.3f}", ''])

    print(f'Done. Results written to {args.out}')


if __name__ == '__main__':
    main()
