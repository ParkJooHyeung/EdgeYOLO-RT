#!/usr/bin/env python3
"""Optimize YOLO ONNX model: simplify ONNX and optionally build TensorRT engine via trtexec.

Usage:
  python scripts/optimize_model.py --onnx yolov8n.onnx --out-dir optimized --trt --fp16
"""
import argparse
import os
import shutil
import subprocess

def simplify_onnx(src, dst):
    try:
        import onnx
        from onnxsim import simplify
    except Exception as e:
        print('ONNX simplifier not available:', e)
        return False

    print(f'Loading ONNX model from {src}...')
    model = onnx.load(src)
    print('Simplifying ONNX model...')
    model_simp, check = simplify(model)
    if not check:
        print('Simplified model failed checks')
        return False
    onnx.save(model_simp, dst)
    print(f'Saved simplified ONNX to {dst}')
    return True

def build_trt(onnx_path, out_dir, fp16=False, int8=False):
    trt_file = os.path.join(out_dir, 'model.trt')
    cmd = ['trtexec', f'--onnx={onnx_path}', f'--saveEngine={trt_file}', '--workspace=2048']
    if fp16:
        cmd.append('--fp16')
    if int8:
        cmd.append('--int8')
    print('Running trtexec:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print('trtexec not found in PATH — skipping TensorRT build')
        return False
    except subprocess.CalledProcessError as e:
        print('trtexec failed:', e)
        return False
    print(f'TensorRT engine saved to {trt_file}')
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out-dir', default='optimized')
    parser.add_argument('--trt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.basename(args.onnx)
    simp_path = os.path.join(args.out_dir, f'simplified_{base}')

    if simplify_onnx(args.onnx, simp_path):
        print('ONNX simplification complete')
    else:
        print('ONNX simplification skipped or failed — copying original ONNX')
        shutil.copy(args.onnx, simp_path)

    if args.trt:
        built = build_trt(simp_path, args.out_dir, fp16=args.fp16, int8=args.int8)
        if not built:
            print('TensorRT build was not completed')

if __name__ == '__main__':
    main()
