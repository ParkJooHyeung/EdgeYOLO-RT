# Curriculum Progress

아래는 사용자가 제시한 커리큘럼을 기준으로 현재 진행 상태를 정리한 체크리스트입니다.

## Phase 1. 환경 구축 및 Baseline 설정 (Python)
- [x] Step 1-1: CUDA/cuDNN/TensorRT 버전 매칭 (설치/확인 완료)
- [x] Step 1-2: Python Baseline 코드 작성 (`scripts/baseline_yolov8.py`)
- [x] 산출물: `baseline_result_1000.csv` 생성 (images 1000장)

## Phase 2. 모델 변환 (ONNX Bridge)
- [x] Step 2-1: .pt → .onnx 변환 (`yolov8n.onnx`, opset=12)
- [ ] Step 2-2: 모델 구조 확인 (Netron에서 Input/Output 노드 확인)

## Phase 3. TensorRT 엔진 빌드 및 CLI 테스트
- [x] `trtexec` / TensorRT 설치 확인 (trtexec 사용 가능)
- [ ] Step 3-1: FP32 vs FP16 엔진 빌드 및 속도/용량 기록
- [ ] Step 3-2: 레이어 프로파일링 (`--dumpProfile`)

## Phase 4. C++ Inference Engine 구현 (핵심 파트)
- [ ] Step 4-1: 프로젝트 세팅 (CMakeLists, 라이브러리 링크)
- [ ] Step 4-2: TensorRT Logger & Runtime 초기화
- [ ] Step 4-3: GPU 메모리 할당 및 버퍼 재사용 설계
- [ ] Step 4-4: Pre-processing (OpenCV + CUDA)
- [ ] Step 4-5: Inference Execution (executeV2 / enqueueV2)
- [ ] Step 4-6: Post-processing (NMS, 좌표 검증)

## Phase 5. 양자화 (INT8) (Advanced)
- [ ] Step 5-1: Calibration Dataset 준비 (100~500장)
- [ ] Step 5-2: INT8 Calibration 및 엔진 생성
- [ ] Step 5-3: 정확도(mAP) 검증

## Phase 6. 최종 벤치마킹 및 결과물 정리
- [ ] Step 6-1: 비교 그래프 작성
- [ ] Step 6-2: 데모 영상 (Side-by-Side)
- [ ] Step 6-3: README 최종 정리 (STAR 기법)

> 현재까지 Phase 1 (환경 구축 및 Baseline)과 Phase 2(ONNX 변환 일부), TensorRT 설치를 완료했습니다.
