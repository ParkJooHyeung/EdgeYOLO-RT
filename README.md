# 🚀 EdgeYOLO-RT: 초경량 실시간 안전 감지 시스템
> **High-Performance Object Detection Inference Engine using TensorRT & C++**

![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-green?logo=nvidia)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-yellow)

## 📖 1. Project Overview (프로젝트 개요)
본 프로젝트는 **임베디드/엣지 환경**에서의 실시간 객체 탐지를 위해, 기존 Python/PyTorch 기반의 YOLOv8 모델을 **C++와 TensorRT**를 사용하여 최적화한 추론 엔진 구현 프로젝트입니다.

산업 현장의 안전 모니터링(안전모 미착용, 위험 구역 침범 등)을 가정하여, 제한된 하드웨어 자원 내에서 **FPS(초당 프레임)를 극대화**하고 **GPU 메모리 사용량을 최소화**하는 것을 목표로 하였습니다.

### **🎯 Key Achievements**
- **속도 향상:** Python(PyTorch) 대비 **약 3배 가속** (30 FPS → **95 FPS**)
- **메모리 최적화:** **INT8 Quantization** 적용을 통해 모델 사이즈 **4배 감소** 및 런타임 메모리 절감
- **Low-Level 구현:** Python 종속성을 제거하고, **CUDA Memory Management** 및 **NMS(Non-Maximum Suppression)** 알고리즘을 C++로 직접 구현

---

## 📊 2. Performance Benchmark (성능 분석)
> *Tested on NVIDIA GeForce RTX 3060 Laptop GPU / Input Resolution: 640x640*

### **⚡ Inference Speed & Memory Usage**
가장 중요한 최적화 성과 지표입니다. TensorRT FP16 및 INT8 적용 시의 성능 변화를 비교했습니다.

| Environment | Precision | Latency (ms) | Throughput (FPS) | GPU Memory (MB) | Improvement |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Python (PyTorch)** | FP32 | 33.2 ms | 30.1 FPS | 1,250 MB | Baseline |
| **C++ (TensorRT)** | FP16 | 12.5 ms | 80.0 FPS | 850 MB | **2.6x Faster** |
| **C++ (TensorRT)** | **INT8** | **10.5 ms** | **95.2 FPS** | **620 MB** | **3.1x Faster** |

![Benchmark Graph](https://via.placeholder.com/800x400?text=Insert+Performance+Graph+Here)

---

## 🛠 3. Technical Highlights (기술적 핵심 내용)

### **A. TensorRT Engine Optimization**
- **ONNX Parsing:** PyTorch(`ultralytics`) 모델을 ONNX로 변환 후, `trtexec` 및 C++ API를 통해 엔진 빌드.
- **Precision Calibration:** FP16(Half Precision) 및 INT8 양자화를 적용하여 연산 효율 증대.
- **Dynamic Shapes:** 고정된 Input Dimension을 사용하여 메모리 재할당 오버헤드 제거.

### **B. Efficient Memory Management (C++ & CUDA)**
- **Buffer Reuse:** 매 프레임마다 `cudaMalloc`을 호출하지 않고, 초기화 단계에서 Input/Output Device 메모리를 한 번만 할당하여 재사용.
- **Zero-Copy Consideration:** Host(CPU)와 Device(GPU) 간의 데이터 전송(`cudaMemcpy`) 최소화 전략 수립.

### **C. Custom Post-Processing**
- YOLOv8의 Output Tensor 구조를 분석하여 Decoding 로직 구현.
- **NMS(Non-Maximum Suppression)** 알고리즘을 C++ `std::sort`와 `IoU` 계산 함수로 직접 구현하여 중복 박스 제거 최적화.

---

## 💻 4. Environment & Prerequisites

### **H/W & S/W Requirements**
# 🚀 EdgeYOLO-RT: 초경량 실시간 안전 감지 시스템
> **High-Performance Object Detection Inference Engine using TensorRT & C++**

![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-green?logo=nvidia)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-yellow)

## 📖 1. Project Overview (프로젝트 개요)
본 프로젝트는 **임베디드/엣지 환경**에서의 실시간 객체 탐지를 위해, 기존 Python/PyTorch 기반의 YOLOv8 모델을 **C++와 TensorRT**를 사용하여 최적화한 추론 엔진 구현 프로젝트입니다.

산업 현장의 안전 모니터링(안전모 미착용, 위험 구역 침범 등)을 가정하여, 제한된 하드웨어 자원 내에서 **FPS(초당 프레임)를 극대화**하고 **GPU 메모리 사용량을 최소화**하는 것을 목표로 하였습니다.

### **🎯 Key Achievements**
- **속도 향상:** Python(PyTorch) 대비 **약 3배 가속** (30 FPS → **95 FPS**)
- **메모리 최적화:** **INT8 Quantization** 적용을 통해 모델 사이즈 **4배 감소** 및 런타임 메모리 절감
- **Low-Level 구현:** Python 종속성을 제거하고, **CUDA Memory Management** 및 **NMS(Non-Maximum Suppression)** 알고리즘을 C++로 직접 구현

---

## 📊 2. Performance Benchmark (성능 분석)
> *Tested on NVIDIA GeForce RTX 3060 Laptop GPU / Input Resolution: 640x640*

### **⚡ Inference Speed & Memory Usage**
가장 중요한 최적화 성과 지표입니다. TensorRT FP16 및 INT8 적용 시의 성능 변화를 비교했습니다.

| Environment | Precision | Latency (ms) | Throughput (FPS) | GPU Memory (MB) | Improvement |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Python (PyTorch)** | FP32 | 33.2 ms | 30.1 FPS | 1,250 MB | Baseline |
| **C++ (TensorRT)** | FP16 | 12.5 ms | 80.0 FPS | 850 MB | **2.6x Faster** |
| **C++ (TensorRT)** | **INT8** | **10.5 ms** | **95.2 FPS** | **620 MB** | **3.1x Faster** |

![Benchmark Graph](https://via.placeholder.com/800x400?text=Insert+Performance+Graph+Here)

---

## 🛠 3. Technical Highlights (기술적 핵심 내용)

### **A. TensorRT Engine Optimization**
- **ONNX Parsing:** PyTorch(`ultralytics`) 모델을 ONNX로 변환 후, `trtexec` 및 C++ API를 통해 엔진 빌드.
- **Precision Calibration:** FP16(Half Precision) 및 INT8 양자화를 적용하여 연산 효율 증대.
- **Dynamic Shapes:** 고정된 Input Dimension을 사용하여 메모리 재할당 오버헤드 제거.

### **B. Efficient Memory Management (C++ & CUDA)**
- **Buffer Reuse:** 매 프레임마다 `cudaMalloc`을 호출하지 않고, 초기화 단계에서 Input/Output Device 메모리를 한 번만 할당하여 재사용.
- **Zero-Copy Consideration:** Host(CPU)와 Device(GPU) 간의 데이터 전송(`cudaMemcpy`) 최소화 전략 수립.

### **C. Custom Post-Processing**
- YOLOv8의 Output Tensor 구조를 분석하여 Decoding 로직 구현.
- **NMS(Non-Maximum Suppression)** 알고리즘을 C++ `std::sort`와 `IoU` 계산 함수로 직접 구현하여 중복 박스 제거 최적화.

---

## 💻 4. Environment & Prerequisites

### **H/W & S/W Requirements**
* **OS:** Ubuntu 20.04 LTS
* **Compiler:** GCC 9.4.0 / CMake 3.10+
* **CUDA:** 11.8
* **TensorRT:** 8.6.1
* **OpenCV:** 4.5.5 (w/ contrib)

---

## 🚀 5. How to Run (실행 방법)

### **Step 1: Convert Model (Python)**
PyTorch 모델을 ONNX로 변환합니다.
```bash
cd python
pip install ultralytics
python export.py --weights yolov8s.pt --dest model.onnx
```