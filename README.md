# LIMO 자율주행: Autorace 2026 (팀 써드임팩트)

WeGo-Robotics × 국민대학교 **Autorace 2026** 출전용 ROS2 자율주행 스택입니다.
Agilex LIMO 위에서 카메라와 LiDAR로 미션 8종을 수행하며 트랙 한 바퀴를 주행합니다.

> **기반 코드**: 인하대 미래자동차사업단 여름 학점교류 **2026 미래모빌리티 AI 자율주행 대회 금상** 코드
> ([jjong102/limo_inha_capstone](https://github.com/jjong102/limo_inha_capstone))를 가져와
> Autorace 규정에 맞게 수정했습니다. 여름 대회의 imitation learning 주행은 라인 추종 + 이벤트 기반 상태머신 구조로 바꿨습니다.

## 팀원

- 인하대 소프트웨어융합공학과 조성윤
- 인하대 전기공학과 성경수
- 인천대 전자공학과 임애진
- 인천대 컴퓨터공학부 박세은

브랜치 규칙과 작업 방법은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고합니다.

## 실행 환경

- ROS2 Humble / Ubuntu 22.04 / Python 3.10
- Jetson Orin Nano (JetPack 6.x) + TensorRT
- 하드웨어 드라이버: `limo_base`, `orbbec_camera`, `ydlidar_ros2_driver`
  - 이 레포에는 없고, 차량의 `~/wego_ws`에 설치되어 있습니다.
  - `jetracer_teleop.launch.py`가 실행할 때 해당 워크스페이스를 source합니다.
- 조향 모드: **아커만** (motion_mode=1) 기준

## Autorace 2026 미션 (규정집 초안 기준)

진행 순서(시계방향): START → M5 → M6 → M3 → M7 → M2 → M8 → M1 → M4

| # | 미션 | 배점 | 주요 센서 | 담당 |
|---|---|---|---|---|
| M1 | 신호등 정지·발진 (방해 광고 모니터) | 15 | 카메라 | 박세은 |
| M2 | 갈림길 A/B 표지판 | 10 | 카메라 | 성경수 |
| M3 | 보행자 표지판 정지 (3초) | 10 | 카메라 | 박세은 |
| M4 | T자 주차 | 15 | 카메라 + LiDAR | 임애진 |
| M5 | 서행 (방지턱) | 10 | 카메라 | 임애진 |
| M6 | 서행 유도선 (지그재그) | 15 | 카메라 | 성경수 |
| M7 | 정적 장애물 회피 (3차선) | 10 | LiDAR | 조성윤 |
| M8 | 라바콘 회피 | 15 | LiDAR | 조성윤 |

## 저장소 구조

```
src/jetracer_ros2/
├── jetracer_ros2/      # 노드 코드
│   └── utils/          # 공용 유틸 (ackermann 변환 등)
├── launch/             # launch 파일
├── params/             # 파라미터 (params.yaml)
├── scripts/            # 학습 / 데이터 전처리 / TRT 변환 스크립트 (ROS 노드 아님)
└── test/               # 단위 테스트 (lane_pipeline)
```

## 노드 구성

이름 규칙: 처음부터 끝까지 켜져 있는 노드는 접두어 없음, `mission_manager_node`가 특정 미션 구간에서만 켜고 끄는 노드는 `mission_` 접두어.
`/cmd_vel`은 `mission_manager_node`만 발행합니다.

### 항상 실행 (`all_in_node.launch.py`)

| 노드 | 역할 | 쓰는 미션 |
|---|---|---|
| `mission_manager_node` | 전체 상태머신. 미션 순서 진행, 주행 명령 선택, 속도·정지 제어, `/cmd_vel` 단독 발행 | 전체 |
| `lane_following_node` | 차선 검출(`lane_pipeline.py`) → 조향. 모드: `center` / `left` / `right` / `heading_hold`, 지그재그 구간 자동 감지 | 기본 주행, M2, M5, M6, M7 |
| `stop_line_node` | 카메라 하단 가로선(흰/노랑 선택) 검출, 앞바퀴→정지선 거리 추정 (`reference_wheel`로 뒷바퀴 기준 전환 가능). 사각지대는 odom으로 추적 | M1, M3 |
| `yolo_detection_node` | YOLO 하나로 신호등 색, A/B, 보행자, 서행, 주차 표지판 인식. 광고 모니터 영역 제외 | M1, M2, M3, M4, M5 |

### 미션 구간에서만 실행 (manager가 켜고 끔)

| 노드 | 역할 | 미션 |
|---|---|---|
| `mission_obstacle_avoid_node` | LiDAR로 앞의 정차 LIMO 감지, 차선 변경 방향 결정 | M7 |
| `mission_cone_avoid_node` | LiDAR로 양쪽 라바콘 사이 경로 추종 (차선 무시) | M8 |
| `mission_traffic_light_node` | YOLO 신호등 결과 + 정지선 거리로 신호 상태 확정 | M1 |
| `mission_parking_node` | LiDAR로 빈칸 탐색 → 전진 → 회전 후진 → 직진 후진 (odom 기준 T자 주차) | M4 |

### 별도 노드 없이 `mission_manager_node`에서 처리

- **M2 갈림길**: YOLO `sign_a` / `sign_b` → `lane_following_node` 모드를 `left` / `right`로 전환
- **M3 보행자**: YOLO `pedestrian` → 정지 후 대기 (`pedestrian_hold_sec`)
- **M5 서행**: YOLO `slow_sign` → `slow_speed`로 감속

### 개발·디버그용

| 노드 / 파일 | 역할 |
|---|---|
| `debug_autorace_node` | 미션 상태와 인식 결과를 RViz 마커(`/autorace/debug_markers`)로 표시 |
| `lane_debug.launch.py` | `lane_following_node`만 단독 실행 (차선 검출 튜닝용) |
| `data_collection_node`, `scripts/*` | 이미지 수집, 학습, TRT 변환 (YOLO 데이터용으로 활용) |

## 토픽 인터페이스

노드끼리 연결되는 토픽 이름과 메시지 타입입니다. **이름을 바꾸려면 팀에서 먼저 합의하고 이 표부터 수정합니다.**

규칙:

- 메시지는 커스텀 msg 없이 기본 타입(`std_msgs`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`)만 사용
- 미션 관련 토픽은 `mission/...`, 미션별 토픽은 `mission/<미션>/...` (예: `mission/parking/cmd_vel`)
- 예외: `inference/cmd_vel`(여름 코드와 manager 인터페이스 호환), `perception/detections`(YOLO 원본 출력)
- 디버그 화면용 토픽은 `/<노드>/debug/...`

### 센서·하드웨어 (wego_ws 드라이버)

| 토픽 | 타입 | 비고 |
|---|---|---|
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 카메라 내부 파라미터 (`lane_following_node`) |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR |
| `/odom` | `nav_msgs/Odometry` | 주행 거리·헤딩 (`mission_parking_node`) |
| `/cmd_vel` | `geometry_msgs/Twist` | 실제 구동 명령. **`mission_manager_node`만 발행** |

### 인식

| 토픽 | 타입 | 발행 | 구독 | 내용 |
|---|---|---|---|---|
| `mission/yolo_event` | `std_msgs/String` | `yolo_detection_node` | manager | `confirm_frames` 연속으로 보인 객체. **클래스마다 따로 세서 각각 발행**하므로 한 화면에 여러 종류가 보여도 묻히지 않음. 확정되는 순간 1회 발행, 사라졌다 다시 보이면 다시 발행 (아래 값) |
| `perception/detections` | `std_msgs/String` (JSON) | `yolo_detection_node` | (없음, 디버그용) | 한 프레임에 보인 객체 전부 |
| `mission/stop_line_distance` | `std_msgs/Float32` | `stop_line_node` | manager, traffic_light, lane(M6) | **앞바퀴**→정지선 거리 [m] (정지선 앞에 서야 하므로 앞바퀴 기준. `reference_wheel: rear`로 전환 가능). 선이 카메라 사각지대로 사라져도 odom으로 계속 추정, 선을 지나치거나 없으면 `-1.0` |
| `mission/stop_line_detected` | `std_msgs/Bool` | `stop_line_node` | (없음) | 거리 값이 유효한지 (화면 검출 또는 odom 추적) |

`mission/yolo_event` 값: `traffic_red`, `traffic_yellow`, `traffic_green`, `sign_a`, `sign_b`, `pedestrian`, `slow_sign`, `parking_sign`

YOLO 학습 클래스 이름은 위 값 그대로 쓰는 것을 권장합니다. (`red`, `a`, `slow`, `parking` 등 일부 별칭도 자동 변환됨)

`perception/detections` 형식:

```json
{"stamp_ns": 1730000000000000000, "detections": [{"label": "sign_a", "confidence": 0.91, "bbox": [300, 160, 340, 200]}]}
```

### 주행 명령 (→ `mission_manager_node`가 하나 선택)

| 토픽 | 타입 | 발행 | 사용 구간 |
|---|---|---|---|
| `inference/cmd_vel` | `geometry_msgs/Twist` | `lane_following_node` | M8, M4 이외 전부 |
| `mission/cone/cmd_vel` | `geometry_msgs/Twist` | `mission_cone_avoid_node` | M8 |
| `mission/parking/cmd_vel` | `geometry_msgs/Twist` | `mission_parking_node` | M4 |

### 상태·모드

| 토픽 | 타입 | 발행 | 구독 | 값 |
|---|---|---|---|---|
| `mission/active` | `std_msgs/String` | manager | debug | 현재 미션 (`M5` … `M4`) |
| `mission/lane_mode` | `std_msgs/String` | manager | `lane_following_node` | `center` / `left` / `right` / `heading_hold` |
| `mission/traffic_light_state` | `std_msgs/String` | `mission_traffic_light_node` | manager | `red` / `yellow` / `green` |
| `mission/obstacle_avoid/state` | `std_msgs/String` | `mission_obstacle_avoid_node` | manager | `change_left` / `change_right` / `stop` / `clear` |
| `mission/cone/state` | `std_msgs/String` | `mission_cone_avoid_node` | manager | `tracking` / `lost` / `emergency_stop` |
| `mission/parking/start` | `std_msgs/Bool` | manager | `mission_parking_node` | `true`면 주차 시작 |
| `mission/parking/state` | `std_msgs/String` | `mission_parking_node` | manager | `waiting` / `searching` / `advance` / `reverse_turn` / `reverse_straight` / `complete` |

### 수동 명령 (테스트용)

`mission/event` (`std_msgs/String`)로 manager에 직접 명령을 보낼 수 있습니다.

| 값 | 동작 |
|---|---|
| `NEXT`, `M번호_DONE` (예: `M5_DONE`) | 다음 미션으로 넘어감 |
| `LANE_CENTER` / `LANE_LEFT` / `LANE_RIGHT` / `HEADING_HOLD` | 라인 추종 모드 변경 |
| `ESTOP` / `ESTOP_CLEAR` | 비상 정지 / 해제 |

```bash
ros2 topic pub --once mission/event std_msgs/String "data: 'NEXT'"
```

### 디버그 (RViz / rqt_image_view)

`/lane/debug/compressed`, `/stop_line/debug/compressed`, `/stop_line/debug/row_ratio`, `/yolo/debug/compressed`, `/autorace/debug_markers`, `/lane/model`, `/lane/mode_state`, `/lane/zigzag`

## 빌드 및 실행

```bash
cd ~/autorace_ws
colcon build --symlink-install
source install/setup.bash

# 터미널 1: 하드웨어 드라이버 (limo_base, camera, lidar)
ros2 launch jetracer_ros2 jetracer_teleop.launch.py

# 터미널 2: 미션 파이프라인
ros2 launch jetracer_ros2 all_in_node.launch.py

# 차선 검출만 따로 테스트할 때
ros2 launch jetracer_ros2 lane_debug.launch.py
```

모든 파라미터(속도, 정지 거리, ROI 등)는 `params/params.yaml`에 있습니다. 실차 보정은 이 파일에서 합니다.
