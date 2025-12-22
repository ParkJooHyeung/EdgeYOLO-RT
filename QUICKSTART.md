# Quick Start

This file shows quick environment setup and running the baseline script.

1) Create and activate virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
# Install torch matching your CUDA version, e.g. for CUDA 12.2:
# pip install --index-url https://download.pytorch.org/whl/cu122 torch torchvision
```

3) Run baseline

```bash
python scripts/baseline_yolov8.py --model yolov8n.pt --images ./images --max_images 1000 --out baseline_result_1000.csv
```

4) Export ONNX

```bash
python - << 'PY'
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='onnx', opset=12)
PY
```

5) TensorRT

Install `trtexec`/TensorRT from NVIDIA apt repo or use an NVIDIA Docker image with TensorRT.

6) Model optimization (optional)

```bash
# Simplify ONNX and optionally build TensorRT engine (requires onnxsim/trtexec)
python scripts/optimize_model.py --onnx yolov8n.onnx --out-dir optimized --trt --fp16

# If you only want to simplify ONNX:
python scripts/optimize_model.py --onnx yolov8n.onnx --out-dir optimized
```

7) Experiments

Simple experiment presets are provided in `scripts/experiments.yaml` for quick sweeps.
