# LIMO 자율주행: Autorace 2026 (팀 써드임팩트)

WeGo-Robotics × 국민대학교 **Autorace 2026** 출전용 ROS2 자율주행 스택입니다.
Agilex LIMO 위에서 카메라와 LiDAR로 미션 8종을 수행하며 트랙 한 바퀴를 주행합니다.

> **기반 코드**: 인하대 미래자동차사업단 여름 학점교류 **2026 미래모빌리티 AI 자율주행 대회 금상** 코드
> ([jjong102/limo_inha_capstone](https://github.com/jjong102/limo_inha_capstone))를 가져와
> Autorace 규정에 맞게 수정하는 중입니다. 아래 "노드 구성"은 여름 대회 기준이며, 수정되는 대로 갱신합니다.

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

| # | 미션 | 배점 | 주요 센서 | 담당 | 상태 |
|---|---|---|---|---|---|
| M1 | 신호등 정지·발진 (방해 광고 모니터) | 15 | 카메라 | 박세은 | 여름 코드 수정 예정 |
| M2 | 갈림길 A/B 표지판 | 10 | 카메라 | 성경수 | 신규 |
| M3 | 보행자 표지판 정지 (3초) | 10 | 카메라 | 박세은 | 신규 |
| M4 | T자 주차 | 15 | 카메라 + LiDAR | 임애진 | 신규 (여름 주차 노드 참고) |
| M5 | 서행 (방지턱) | 10 | 카메라 | 임애진 | 신규 |
| M6 | 서행 유도선 (지그재그) | 15 | 카메라 | 성경수 | 신규 |
| M7 | 정적 장애물 회피 (3차선) | 10 | LiDAR | 조성윤 | 신규 (트럭 추종 노드 참고) |
| M8 | 라바콘 회피 | 15 | LiDAR | 조성윤 | 신규 |

기본 주행은 여름 대회의 imitation learning 조향 대신 **차선 검출 기반 라인 추종**으로 바꿀 예정입니다.

## 저장소 구조

```
src/jetracer_ros2/
├── jetracer_ros2/      # 노드 코드
│   └── utils/          # 공용 유틸 (ackermann 변환 등)
├── launch/             # launch 파일
├── params/             # 파라미터 (params.yaml)
└── scripts/            # 학습 / 데이터 전처리 / TRT 변환 스크립트 (ROS 노드 아님)
```

## 노드 구성

이름 규칙: 처음부터 끝까지 켜져 있는 노드는 접두어 없음, `mission_manager_node`가 특정 구간에서만 켜고 끄는 노드는 `mission_` 접두어.
`/cmd_vel`은 `mission_manager_node`만 발행하고, 주행 노드는 `inference/cmd_vel`로 발행합니다.

### 신규 노드

| 노드 | 역할 | 담당 미션 |
|---|---|---|
| `lane_following_node` | 차선 검출 → 조향, `inference/cmd_vel` 발행. 모드: 중앙 / 왼쪽선 / 오른쪽선 / 헤딩 유지 | 기본 주행, M2, M6, M7 차선 변경 |
| `stop_line_node` | 카메라 하단 정지선(가로선) 검출, 남은 거리 발행 | M1, M3 |
| `yolo_detection_node` | YOLO 하나로 신호등 색, A/B, 보행자, 서행, 주차 표지판 인식 | M1, M2, M3, M4, M5 |
| `mission_obstacle_avoid_node` | LiDAR로 정차 LIMO 감지, 차선 변경 방향 결정 | M7 |
| `mission_cone_avoid_node` | LiDAR로 라바콘 사이 경로 생성 (차선 무시) | M8 |

### 기존 노드 (여름 대회 코드)

| 노드 | 여름 대회 역할 | 처리 |
|---|---|---|
| `mission_manager_node` | 전체 상태머신, `/cmd_vel` 단일 발행 | 유지. 단계를 M5→M6→M3→M7→M2→M8→M1→M4, 이벤트 기반으로 재작성 |
| `mission_traffic_light_node` | HSV 적/녹 신호등 인식 | 유지. YOLO 결과 + 정지선 기반으로 수정 (황색 추가, 광고 모니터 배제) |
| `mission_parking_node` | 벽 추종 + 평행주차 | 이름 유지, M4 T자 주차로 재작성 |
| `data_collection_node`, `scripts/*` | 데이터 수집·학습·TRT 변환 | 유지. YOLO 데이터 수집 / TRT 변환에 사용 |
| `debug_*_node` | 판정 결과 RViz 시각화 | 유지. 신규 노드에도 같은 방식으로 추가 |
| `mission_track_following_node` | LiDAR 좌측 트럭 거리 유지 | M7 작성 시 참고 후 삭제 |
| `inference_node`, `inference_go_stop_node`, `mission_inference_node` | IL 조향 추론 | 삭제 (`lane_following_node`로 대체) |
| `mission_tunnel_node`, `mission_people_estop_node` | 터널 감지, LiDAR 보행자 정지 | 삭제 |

### 별도 노드 없이 `mission_manager_node`에서 처리

- **M2 갈림길**: YOLO의 A/B 결과로 `lane_following_node` 모드를 왼쪽선 / 오른쪽선으로 전환
- **M3 보행자**: YOLO 결과 + 정지선으로 정지 후 3초 대기
- **M5 서행**: 속도만 절반 (≤0.2 m/s)
- **M6 지그재그**: `lane_following_node`를 헤딩 유지 모드로 전환

## 토픽 인터페이스

노드끼리 연결되는 토픽 이름과 메시지 타입입니다. **이름을 바꾸려면 팀에서 먼저 합의하고 이 표부터 수정합니다.**

규칙:

- 메시지는 커스텀 msg 없이 기본 타입(`std_msgs`, `geometry_msgs`, `sensor_msgs`)만 사용
- 항상 켜진 인식 노드 출력은 `perception/...`, 미션 노드 상태는 `mission/...`
- 주행 명령은 각 노드가 `<출처>/cmd_vel`로 보내고, `mission_manager_node`가 단계에 맞는 것 하나만 골라 `/cmd_vel`로 내보냄

### 센서·하드웨어 (wego_ws 드라이버)

| 토픽 | 타입 | 비고 |
|---|---|---|
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR |
| `/odom` | `nav_msgs/Odometry` | 주행 거리·헤딩 (M6 헤딩 유지용, 차량에서 토픽 이름 확인 필요) |
| `/cmd_vel` | `geometry_msgs/Twist` | 실제 구동 명령. **`mission_manager_node`만 발행** |

### 인식

| 토픽 | 타입 | 발행 | 구독 | 내용 |
|---|---|---|---|---|
| `perception/detections` | `std_msgs/String` (JSON) | `yolo_detection_node` | manager, traffic_light, parking | 아래 형식 참고 |
| `perception/stop_line_distance` | `std_msgs/Float32` | `stop_line_node` | manager, traffic_light | 정지선까지 거리 [m], 안 보이면 `-1.0` |

`perception/detections` 형식 (한 프레임에 보인 객체 전부):

```json
[{"label": "sign_A", "conf": 0.91, "x": 320, "y": 180, "w": 40, "h": 40}]
```

YOLO 클래스 이름 (`label`): `traffic_red`, `traffic_yellow`, `traffic_green`, `sign_A`, `sign_B`, `pedestrian`, `slow`, `parking`

### 주행 명령 (→ `mission_manager_node`가 선택)

| 토픽 | 타입 | 발행 | 사용 구간 |
|---|---|---|---|
| `inference/cmd_vel` | `geometry_msgs/Twist` | `lane_following_node` | 기본 주행, M2, M5, M6, M7 |
| `cone_avoid/cmd_vel` | `geometry_msgs/Twist` | `mission_cone_avoid_node` | M8 |
| `parking/cmd_vel` | `geometry_msgs/Twist` | `mission_parking_node` | M4 |

### 상태·모드

| 토픽 | 타입 | 발행 | 구독 | 값 |
|---|---|---|---|---|
| `mission/lane_mode` | `std_msgs/String` | manager | `lane_following_node` | `center` / `left` / `right` / `heading` |
| `mission/traffic_light_state` | `std_msgs/String` | `mission_traffic_light_node` | manager | `red` / `yellow` / `green` |
| `mission/obstacle_avoid_state` | `std_msgs/String` | `mission_obstacle_avoid_node` | manager | `left` / `right` (피할 방향) / `clear` |
| `mission/cone_avoid_state` | `std_msgs/String` | `mission_cone_avoid_node` | manager | `running` / `done` |
| `mission/parking_state` | `std_msgs/String` | `mission_parking_node` | manager | `running` / `done` |

## 빌드 및 실행

```bash
cd ~/autorace_ws
colcon build --symlink-install
source install/setup.bash

# 터미널 1: 하드웨어 드라이버 (limo_base, camera, lidar)
ros2 launch jetracer_ros2 jetracer_teleop.launch.py

# 터미널 2: 미션 파이프라인
ros2 launch jetracer_ros2 all_in_node.launch.py
```
