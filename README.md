# LIMO 자율주행 (인하대 캡스톤 - 써드임팩트)

Agilex **LIMO** 플랫폼 위에서 카메라 영상만으로 조향을 예측하는 imitation learning 모델과,
라이다/카메라 기반 미션 판단 로직을 결합해 신호등 대기 → 라바콘 구간 주행 → 로타리 →
보행자 정지 → 터널/트럭 추종 → 자동 주차까지 하나의 코스를 완주하는 자율주행 스택입니다.

## 팀원 (3명)

- 인천대 임베디드시스템공학과 이원종
- 인하대 전기공학과 성경수
- 인천대 전자공학과 임애진

## 교육 및 대회 개요

[교육 및 대회 개요](https://docs.google.com/document/d/1hYNgvPgOFEsmw6BR12WKt5NZ7-oVlr5PEMmsoVmZoYw/edit?tab=t.xgsp502b8pry#heading=h.4pdv5a5ce13l)

## 시연 영상

[![데모 영상](https://img.youtube.com/vi/hux1rPf8BKY/0.jpg)](https://youtu.be/hux1rPf8BKY)

위 썸네일을 클릭하면 전체 코스 시연 영상을 볼 수 있습니다.

## 수상

인하대학교 미래자동차사업단 주관 **2026 미래모빌리티 AI 자율주행 대회** 금상

<img src="docs/%EC%9D%B8%ED%95%98%EB%8C%80%20%EA%B8%88%EC%83%81.png" width="360" alt="2026 미래모빌리티 AI 자율주행 대회 금상" />

## 실행 환경

- **ROS2 Humble**
- Python 3.10
- 학습(train.py / train_go_stop.py): Google Colab **T4 GPU**
- 실차 추론: Jetson Orin nano + TensorRT (PyTorch `.pth` → ONNX → `.engine`)
- 하드웨어: LIMO 베이스(`limo_base`), Orbbec 카메라(`orbbec_camera`), YDLidar(`ydlidar_ros2_driver`)
  - 이 세 패키지는 용량/버전 문제로 이 저장소가 아닌 `wego_ws`에 별도로 설치되어 있고,
    `jetracer_teleop.launch.py`가 실행 시점에 그 워크스페이스를 직접 source합니다.

## 저장소 구조

패키지는 `jetracer_ros2` 하나뿐입니다.

```
src/jetracer_ros2/
├── jetracer_ros2/        # 노드 코드
│   ├── utils/            # 공용 유틸 (ackermann 변환 등)
│   └── *.py
├── launch/                # launch 파일
├── params/                # 파라미터 (params.yaml)
├── scripts/                # 학습 / 데이터 전처리 / TRT 변환 스크립트 (ROS 노드 아님)
└── jetracer_models/       # 학습된 가중치 예시 (.pth / .onnx)
```

> `jetracer_dataset/`(수집한 원본 이미지)와 `jetracer_engines/`(Jetson에서 변환한 `.engine`
> 파일)는 용량 문제로 저장소에 올리지 않았습니다. `.gitignore`에서도 제외되어 있습니다.

## 주행 파이프라인

전체 미션은 `mission_manager_node` 하나가 상태머신으로 지휘합니다. `/cmd_vel`에 실제로
발행하는 것도(주차 단계 제외) 이 노드뿐이고, 나머지 미션 노드들은 필요한 구간에서만
`mission_manager_node`가 subprocess로 켜고 끕니다.

1. **신호등 대기** — `mission_traffic_light_node`가 카메라로 신호등(초록/빨강)을 인식할 때까지 정지
2. **라바콘 구간 / 로타리 주행** — `mission_inference_node`(imitation learning, go_stop 모델)가
   카메라 영상만으로 조향을 예측해 라바콘 사이 코스와 로타리 곡선 구간을 주행

   <p float="left">
     <img src="docs/%EB%9D%BC%EB%B0%94%EC%BD%98%20%ED%9A%8C%ED%94%BC%20%EC%A3%BC%ED%96%89%20.jpg" width="48%" alt="라바콘 회피 주행" />
     <img src="docs/%EB%A1%9C%ED%83%80%EB%A6%AC%20%EC%A3%BC%ED%96%89.jpg" width="48%" alt="로타리 주행" />
   </p>

3. **보행자 e-stop** — `mission_people_estop_node`가 라이다로 전방 장애물을 감지하면 선제 감속 후 정지, 사라지면 재출발
4. **터널 / 트럭 추종** — `mission_tunnel_node`가 조도 급락으로 터널 진입을 감지하면 감속하고,
   `mission_track_following_node`가 라이다로 좌측 트럭과의 거리를 유지하며 추종
5. **자동 주차** — `mission_parking_node`가 라이다로 오른쪽 벽을 따라 접근한 뒤, 정지선에서 주차 시퀀스를 실행

각 단계 전환과 속도 재조정 로직의 자세한 흐름은
[`mission_manager_node.py`](src/jetracer_ros2/jetracer_ros2/mission_manager_node.py) 상단 주석에 정리되어 있습니다.

## 노드 구성

### 주행 / 추론

| 노드 | 설명 |
|---|---|
| `data_collection_node` | 조이스틱으로 직접 주행하며 (이미지, 조향) 데이터를 수집. 파일명에 조향값(cx)을 인코딩해 저장 |
| `inference_node` | 구간(section)별 TensorRT 엔진으로 조향만 추론해 `/cmd_vel`로 직접 발행 (기본 주행용) |
| `inference_go_stop_node` | 조향 + 정지판단을 동시에 하는 go_stop 모델 추론 노드. 정지 확정 시 `/parking_start` 발행 |
| `mission_inference_node` | `inference_go_stop_node`를 재사용하되 `/cmd_vel` 대신 `inference/cmd_vel`로 발행 (미션 파이프라인 전용) |

### 미션 상태머신

| 노드 | 설명 |
|---|---|
| `mission_manager_node` | 전체 미션 단계를 순서대로 진행시키는 상태머신. `/cmd_vel` 발행 주체 |

### 미션별 센서 / 판단 노드 (각각 `debug_` 버전은 RViz로 인식 결과를 시각화)

| 노드 | 설명 |
|---|---|
| `mission_traffic_light_node` | ROI에서 HSV 색상 기준으로 신호등 색(red/green) 판정 |
| `mission_people_estop_node` | 라이다로 전방 보행자/장애물 감지 → stop/clear 발행 |
| `mission_tunnel_node` | 카메라 평균 밝기로 터널 진입(조도 급락) 감지 |
| `mission_track_following_node` | 라이다로 좌측 트럭과의 거리 감시 → stop/clear 발행 |
| `mission_parking_node` | 우측 벽 추종으로 접근 후 주차 시퀀스 실행 (Phase 1: 접근/정지, Phase 2: 주차) |
| `debug_mission_*_node` | 위 노드들을 그대로 재사용하되 판정 근거(ROI, FOV 등)를 RViz용 토픽으로 추가 발행하는 디버그용 |

### 학습 / 전처리 스크립트 (`scripts/`, ROS 노드 아님 — 서버/Jetson에서 직접 실행)

| 스크립트 | 설명 |
|---|---|
| `train.py` | 구간별 조향 회귀 모델(ResNet18) 학습. Colab **T4**에서 실행 |
| `train_go_stop.py` | 조향 + 정지판단을 동시에 뽑는 멀티태스크 모델 학습. Colab **T4**에서 실행 |
| `balance_dataset.py` | 수집한 데이터셋의 조향값 분포 확인 및 직진 편중 다운샘플링 |
| `onnx_to_trt.py` | 학습된 `.onnx`를 Jetson에서 TensorRT `.engine`으로 변환 |

## 실행 방법

```bash
# 터미널 1: 하드웨어 드라이버 (limo_base, camera, lidar)
ros2 launch jetracer_ros2 jetracer_teleop.launch.py

# 터미널 2: 미션 파이프라인
ros2 launch jetracer_ros2 all_in_node.launch.py
```

데이터 수집 → 학습 → 배포 흐름:

```bash
# 1) 조이스틱으로 주행하며 구간별 데이터 수집 (Jetson)
ros2 run jetracer_ros2 data_collection_node --ros-args -p section_id:=1

# 2) 조향값 분포 확인 / 다운샘플링 (선택)
python balance_dataset.py --data_dir ~/jetracer_dataset --section 1

# 3) 학습 (Colab, T4 GPU)
python train.py --data_dir ~/jetracer_dataset --section all --epochs 50
python train_go_stop.py --data_dir ~/jetracer_dataset --epochs 50

# 4) TensorRT 엔진 변환 (Jetson)
python onnx_to_trt.py --onnx_dir ./models --output_dir ~/jetracer_engines
```

## 발표자료

[캡스톤디자인 최종발표.pdf](docs/%EC%BA%A1%EC%8A%A4%ED%86%A4%EB%94%94%EC%9E%90%EC%9D%B8_%EC%B5%9C%EC%A2%85%EB%B0%9C%ED%91%9C.pdf)
