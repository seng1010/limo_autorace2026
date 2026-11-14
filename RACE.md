# race 브랜치 (대회용 통합 브랜치)

`m1-traffic-light`(매니저 + M1 + M3 + 정지선) + `common-yolo` 를 합치고, 조향 수정과 `simple_lane` 노드를 넣은 브랜치.
main 과 기존 브랜치는 건드리지 않았다. 차에서 확인된 것만 여기에 올린다.

## 1. 조향 수정 (모든 노드에 해당)

| | 최대 조향에서 실제 회전반경 |
|---|---|
| `limo_base.launch.py` + `/cmd_vel` (기존) | 1.0 ~ 1.2 m |
| `limo_base_ackerman.launch.py` + 변환 노드 | 0.50 ~ 0.53 m |

- 원인: `limo_driver.cpp` 436~453줄. 조향각을 28도로 자른 뒤 2.47로 나눠 보냄 -> 차가 받는 최대값 11.3도.
  명령을 2.6배로 키워도 회전반경이 그대로였다 (11/14 실차 측정, 카메라 배경 회전량 기준).
- 해결: 같은 패키지의 `limo_ackerman_base` 드라이버는 0.42rad(24도)까지 보낸다. 이 드라이버는
  `/limo/ack_cmd` (`ackermann_msgs/AckermannDrive`) 를 받으므로, `/cmd_vel` 을 그 형식으로 바꿔 주는
  변환 노드를 켠다: `ros2 run jetracer_ros2 simple_lane bridge` (하드웨어 launch 에 포함)
- 기존 노드 코드는 고칠 필요 없다. 단, 예전 드라이버 기준으로 맞춘 조향 관련 숫자(주차 시간, 콘 조향 세기 등)는 다시 확인.
- 변환 노드가 꺼져 있으면 차가 움직이지 않는다 (0.5초 넘게 `/cmd_vel` 이 없으면 정지 명령).

## 2. 켜는 순서

처음 한 번 (LIMO 에서 브랜치를 받은 뒤):

```bash
cd ~/limo_autorace2026 && colcon build --symlink-install && source install/setup.bash
```

터미널 2개면 된다.

```bash
# 터미널 1) 하드웨어: 새 드라이버 + 카메라 + 라이다 + 변환 노드
ros2 launch jetracer_ros2 jetracer_teleop.launch.py

# 터미널 2-a) 차선 주행만 시험 (매니저 없음)
ros2 launch jetracer_ros2 lane_only.launch.py rec:=true

# 터미널 2-b) 전체 (매니저 + simple_lane + 정지선 + YOLO)
ros2 launch jetracer_ros2 all_in_node.launch.py
```

- 변환 노드는 하드웨어 launch 가 같이 켠다. 따로 켤 필요 없다.
- `all_in_node.launch.py` 의 차선 노드는 기본이 `simple_lane` 이다. 팀 `lane_following_node` 로 돌리려면 `team_lane:=true`.
- 슬라이더 창은 화면이 있는 터미널(NoMachine)에서 켰을 때만 뜬다. 튜닝할 때는 손으로 켜는 게 편하다:
  `ros2 run jetracer_ros2 simple_lane drive rec`  (매니저와 함께면 뒤에 `--topic inference/cmd_vel`)
- 빌드 없이 파일 하나만 복사해서 `python3 simple_lane.py drive` 로 돌려도 똑같이 동작한다.
- odom 토픽은 새 드라이버에서도 `/odom` 이다.
- **실행 파일(launch) 변경은 아직 차에서 돌려 보지 않았다.** 처음 켤 때 드라이버·변환 노드가 뜨는지 확인할 것.

## 3. simple_lane.py 가 하는 일

- 노란 차선 사이 가운데를 따라가는 기본 주행 (`lane_following_node` 자리).
- 90도 코너: 곡선으로 판단된 상태에서 바깥 선이 정면 `turn_in_cm`(75) 안으로 들어오면 최대 조향을 걸고,
  odom 으로 `turn_deg`(60)도 돌면 풀어서 다시 차선을 따라간다. 도는 중 선이 안 보여도 멈추지 않는다.
- 11/14 밤 실차: 위쪽 직선 -> 90도 코너 -> 지그재그 -> 횡단보도 -> 90도 코너까지 이탈 없이 주행.
- 아직 없는 것: `mission/lane_mode`(왼쪽/오른쪽 차로 지시) 처리, 구간 자동 전환 이벤트.

## 4. 남은 일 (앞 구간부터 차에서 확인)

- [ ] 장애물 없이 한 바퀴: 3차선, 갈림길, 나머지 코너에서 기본 주행 확인
- [ ] 매니저 연결 + 구간 자동 전환 (M5, M6, M2, M8 은 지금 스스로 다음으로 넘어가지 않음)
- [ ] YOLO 모델 (`params.yaml` 의 `yolo_detection_node.model_path` 가 비어 있음)
- [ ] M3 -> M1 -> M2 -> M7 -> M8 -> M4 순서로 실차 확인

## 5. 구간 전환 + 서행 (section_node)

매니저는 M5·M6·M2·M8 을 스스로 넘기지 않는다. `section_node` 가 정지선(`stop_line_node`)과 odom 거리로
"이 구간이 끝났다"를 판단해 `mission/event` 로 `M5_DONE` 같은 신호를 보낸다. 팀 미션 코드는 그대로다.

- 정지선을 놓쳐도 정해진 거리(timeout)에서 넘어간다 -> 한 번 놓쳐도 뒤 미션이 밀리지 않는다.
- 서행 구간(M5 방지턱, M6 지그재그)에서는 `mission/speed_limit` 로 속도 상한(0.13 m/s)을 내고 매니저가 적용한다.
- `all_in_node.launch.py` 가 같이 켠다 (끄려면 `section:=false`).
- 설정: `params.yaml` 의 `section_node`. **지금 숫자는 도면 치수로 넣은 임시값**이다. 실차 기록으로 맞춘 뒤 쓸 것.
- 확인: `ros2 topic echo /section/debug` (미션, 간 거리, 서행 여부, 마지막 정지선).
  노드 로그에 정지선을 지날 때마다 `정지선: 누적 8.80 m (M5 시작 후 8.80 m)` 가 찍힌다 -> 이 값으로 설정을 맞춘다.
- `m3_trigger` 기본값을 `sign_or_line` 으로 바꿨다 (YOLO 없이도 횡단보도 정지선에서 선다).
- 시뮬레이션(ROS 없이 매니저 + section_node 연결)으로 M5 서행 -> M6 -> M3 정지·대기·출발 -> M7 순서를 확인했다. 차에서는 아직.
