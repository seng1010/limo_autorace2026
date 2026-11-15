# M8 콘 주행 통합본 — 구현, 검증, 배포

2026-11-15. 배포 브랜치: `feature/m8-cone-corridor`.

## 결론

긴 좌우 경계선/중앙경로와 여러 단계 복구를 **관측된 콘 사이 통과점 + 20 cm 진행 원호**로 대체했다. 차선 노드는 계속 실행한다. M8 진입이 허용된 상태에서 콘 명령을 받으면 매니저가 즉시 전달한다. 콘 내부에서 차선 명령을 섞지 않는다.

실차 성공 보고서는 아니다. bag의 고정 시점에 새 판단을 넣는 재생과 별도의 단순 폐루프 시뮬레이션을 구분했다. 명령 발생률을 통과 확률로 해석하지 않는다. 개발 Mac에서는 ROS2/Jetson 실행을 검증하지 못했다.

## 없앤 것과 남긴 것

- 제거: 전체 두 줄 완성이 필요한 경로 구성, 여러 단계 대체 원호, 동일 원호 충돌검사의 반복, 공용 YAML의 M8 설정, 매니저 안에 쌓인 콘 전용 분기.
- 유지: 최소 구동속도 0.10 m/s, 센서 유효성, 차체·콘 받침 충돌 검사, 사용자 ESTOP, 명령 타임스탬프, 출구에서 실제 관측 차선 확인.
- 통합: ROS와 외부 재생이 동일한 `Planner.update`를 사용한다.
- 짧은 유실: 명령을 맹목적으로 유지하지 않는다. 최근 관측 통과점/주변 점을 odom으로 좌표 변환하여 최대 0.5초·0.10 m 재사용하고, 현재 스캔으로 매번 진행 방향과 충돌을 확인한다. 예측만으로 기억 시간을 연장하지 않는다.
- `lost/emergency_stop` 문자열이 공통 `_blocked`를 덮어쓰지 않는다. 속도·각속도·스캔 시간이 묶인 명령을 읽고, 사용자 ESTOP는 별도로 유지한다.

## 계획기

1. beam 순서/거리 변화로 클러스터링한다. 충돌검사에는 클러스터가 아닌 원시 점도 남긴다.
2. 가까운 최대 24개 클러스터에서 횡방향 콘 쌍을 찾는다. 유효 폭, 가운데 다른 장애물, 진행 방향을 확인한다. 완성된 두 줄 전체는 요구하지 않는다.
3. 거리·방향·통과점 도착 방향으로 점수화한다. 단순 최근접 쌍의 대각선 오연결을 줄이고 상위 8개 후보를 평가한다.
4. 아커만 곡률 범위에서 20 cm 진행 원호를 샘플링한다. 차체 직사각형 모서리까지 원시 점과 충돌하지 않고 목표에 가까워지는 원호를 선택한다.
5. 같은 곡률의 충돌 결과는 스캔마다 한 번만 계산한다. 안전한 두 스캔 뒤 0.10 m/s, 장애물/스캔 유실이면 0을 발행한다.
6. odom이 없거나 오래되면 현재 스캔의 새 쌍만 사용한다. 후진, odom 점프, 시간 역행 시 기억을 지운다.

멀리까지 완벽한 경로가 나오길 기다리지 않고 짧게 진행해 재관측한다. 콘이 안 보이거나 차체/진행 원호가 막혔는데 강제 출발시키는 방식은 아니다.

## 제어권과 기존 주행

| 단계 | 동작 |
|---|---|
| 평상시 M2, M8 비활성 | 기존 race 주행 |
| ARM_M8/M2_DONE/NEXT 또는 지정 거리로 준비 | 차선 인식 유지, LiDAR 검사한 접근 차선 명령 |
| 준비 상태에서 콘 통과점 확보 | 첫 정지선 통과를 추가로 기다리지 않고 M8 콘 제어 |
| 콘 내부 | 콘 명령만 사용, 없거나 오래되거나 0이면 정지 |
| 서로 다른 출구 정지선 확인 | 새 관측 차선을 LiDAR로 검사하여 인계, 없으면 대기 |
| 출구 선 통과 + 차선 진행 거리 충족 | 기존 순서의 다음 미션 |

`Planner → mission/cone/safe_cmd_vel(TwistStamped) → M8Control → mission_manager_node /cmd_vel → 기존 cmdvel_to_ackermann → /limo/ack_cmd`

`mission/cone/cmd_vel`은 호환/관측용이다. 매니저는 시간이 포함된 safe_cmd_vel을 읽는다. 상태 문자열과 명령의 수신 순서가 달라 출발이 막히던 구조를 제거했다.

`angular.z = v × curvature`이다. 기존 race bridge/보정 배율을 콘에서 또 곱하지 않는다. 안쪽 바퀴 목표각과 /limo/ack_cmd의 보정된 중심각 명령은 서로 다른 값이며, 실제 조향 센서 측정값도 아니다. 예전 limo_base의 11도 제한 경로로 바꾸지 않는다.

### 정지선과 전체 주행의 전제

첫 선은 콘 진입의 필수 조건이 아니지만 **두 번째 선을 구별하는 기준**이다. 전체 인계 시험은 첫 정지선이 보이는 입구에서 시작한다. 콘 한가운데에서 `start:=M8`로 시작하면 콘 주행은 가능해도 첫 선 기준이 없어 자동 출구 인계가 완료되지 않을 수 있다.

전체 대회에서는 기본 설정대로 팀의 구간 판단이 M2에서 ARM_M8/M2_DONE을 전달한다. M8 시험 launch의 `start:=M2 motion:=true`에서는 먼저 `watch_entry`로 입구 선만 관측하고, 기존 race 차선 명령과 속도 제한을 그대로 사용한다. 앞 차축 기준 `entry_near_m` (기본 0.45m) 이내에서 입구 선이 일관되게 3회 관측되면 `approach`로 바뀌고 기존 LiDAR 접근 검사와 0.10 m/s 제한이 적용된다. 입구 선을 못 보면 자동으로 검사 모드에 들어가지 않으므로, 첫 실차 시험은 선이 보이는 접근 구간에서 진행하고 전환되지 않으면 차량을 정지한다. `motion:=false`는 일반 차선 명령이 나가지 않도록 기존 즉시 접근 가드를 유지한다.

## 파일 분리와 병합

| 파일 | 책임 |
|---|---|
| cone_corridor.py | ROS 없는 클러스터·통과점·원호·짧은 odom 기억 |
| mission_cone_avoid_node.py | 센서 수신, 단일 계획기, 시간 검사, 명령 발행 |
| m8_control.py | 매니저 인계·ESTOP·진입/출구 |
| m8_entry.py | 순수 명령/정지선 정책 |
| m8_observers.py | 기존 차선 결과와 해독된 영상에서 M8 관측 전달 |
| params/m8.yaml | 콘 설정 |
| launch/m8_cone.launch.py | 접근·콘·출구 시험과 bag 기록 |

공용 파일은 mission_manager_node의 생성/event/enter/tick, simple_lane의 결과 전달, stop_line_node의 관측 호출만 연결한다. 기존 params.yaml, 일반 정지선 검출기, 차선 계산/조향, bridge, 3차선 로직은 변경하지 않는다. 미래의 같은 줄 수정까지 무충돌을 보장하지 않지만 현재 병합 범위를 줄였다.

## 재생 결과와 부하

원본 bag은 수정하지 않았다. 14개 폴더 중 /scan이 있는 13개, **9,414 스캔**을 확인했다. bag_cone_05는 카메라/odom은 있으나 /scan 샘플이 없어 LiDAR 시험에서 제외했다.

| bag | 스캔 수 | 새 구현의 진행 명령 수 | 계획기 p95 ms |
|---|---:|---:|---:|
| bag_cone_01 | 475 | 168 | 7.96 |
| bag_cone_02 | 348 | 124 | 6.54 |
| bag_cone_03 | 324 | 129 | 6.85 |
| bag_cone_04 | 261 | 134 | 8.00 |
| bag_cone_test_01 | 314 | 273 | 6.01 |
| bag_cone_test_auto_controll_01 | 2340 | 1207 | 6.04 |
| bag_cone_test_direect_controll_01 | 134 | 85 | 6.13 |
| bag_cone_m8_test | 1011 | 292 | 5.39 |
| bag_cone_retest_01 | 1590 | 596 | 5.49 |
| bag_cone_test_auto_controll_02 | 310 | 219 | 7.68 |
| bag_cone_far_entry_20261115_173630 | 924 | 582 | 7.95 |
| bag_cone_two_lines_20261115_173116 | 694 | 369 | 4.56 |
| bag_m8_v12_20261115_192312 | 689 | 200 | 5.03 |

정리 직전 **미커밋 근거리 구현** 대비 bag_cone_04는 74→134, retest_01은 252→596, two_lines는 139→369개 진행 명령이었다. 이것은 원격 v12와의 동일 비교가 아니다. 마지막 v12 bag에 기록된 실제 /cmd_vel과 /limo/ack_cmd는 모두 정지였다. 새 200개는 같은 센서 시점에서 새 계획기/명령 정책을 실행한 결과다. 전체 기록에는 접근 전·후진·미션 종료 후도 포함되므로 이 비율은 성공률이 아니다.

시간은 **개발 Mac의 Python 코어** 기준이다. p95 약 4.6–8.0 ms, CPU 시간 p95 약 4.5–7.4 ms, 최대 wall time 약 36.7 ms였다. ROS 전송·다른 노드·Jetson 스케줄링 비용은 제외했다. GPU 추론은 없다. 실차는 `plan_ms/exit_ms/callback_ms/arrival_age_ms/output_age_ms`로 확인한다.

별도 단순 폐루프 시뮬레이션은 직선/좌우 곡선, 폭 0.60/0.72 m, 진입 각도/오프셋, ray 누락·잡음, 조향 변화율 제한을 조합했다. 점수 개선 전 21/24에서 개선 후 **24/24 완료, 충돌 0회**, 모델 받침까지 최소 여유 약 3.96 cm였다. 실제 마찰·조향·사각·받침 오차를 보장하지 않는다. 실차 90%/100% 성공으로 표기하지 않는다.

테스트는 계획기·원시 점 충돌·시간/odom 리셋·명령 인계·ESTOP·정지선과 기존 미션 회귀를 포함한다. 매니저 연결은 실제 M8Control을 가짜 ROS 메시지/전송으로 실행했다. 실제 DDS/launch/하드웨어 통합 검증은 별도로 필요하다.

조향 비교는 동일한 bridge 명령각 단위로 변환했다. 비교 가능한 움직임 샘플에서 기록 대비 절대 차이 중앙값은 auto_controll_02 약 4.62도, far_entry 약 7.91도, retest 약 13.11도다. 실패 주행/다른 제어기의 명령은 정답 조향이 아니므로 이를 그대로 모사하도록 튜닝하지 않았다. 성공 수동 기록 bag_cone_04에는 /limo/ack_cmd가 없어 정확한 명령각 차이를 계산하지 못한다. odom으로 추정한 곡률을 실제 조향 센서라고 대체 표기하지 않았다.

## 방법 선택과 출처

- [Coulter, Pure Pursuit (1992)](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm): 관측 목표에서 곡률을 구하는 기하 추종. 목표 쌍이 잘못되면 추종도 잘못되므로 쌍 방향/폭을 함께 확인한다.
- [Regulated Pure Pursuit (2023)](https://arxiv.org/abs/2305.20026), [Nav2 공식 설명](https://api.nav2.org/nav2-rolling/html/md_nav2_regulated_pure_pursuit_controller_README.html): 속도/전방 충돌 규제 원칙 참고. 전체 Nav2 RPP를 이식한 것은 아니다.
- [Dynamic Window Approach (1997)](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance): 가까운 실행 후보 평가 원칙 참고. LIMO에서는 제자리 회전이 아니라 아커만 곡률만 사용한다.
- [F1TENTH Follow the Gap](https://github.com/f1tenth/f1tenth_lab4_template): 매우 가볍지만 가장 넓은 빈 공간이 콘 코스 바깥일 수 있어 이번에는 콘 쌍을 목표로 선택했다.
- [Mapless Local Trajectory Planning (2024)](https://arxiv.org/abs/2401.17732), [FaSTTUBe](https://github.com/papalotis/ft-fsd-path-planning): 경계/그래프 활용은 충분한 관측에서 유리하나 이번 입구의 희소 관측에서 완성 경로를 요구하는 부담을 피했다.
- [공개 LIMO URDF](https://raw.githubusercontent.com/agilexrobotics/limo_ros2/humble/limo_description/urdf/limo_ackerman.xacro), [LIMO Pro 문서](https://github.com/agilexrobotics/limo_pro_doc/blob/master/Limo%20Pro%20user%20manual%28EN%29.md): 명목 장착 참고. laser x=0.103와 rear axle x=-0.10 차이 0.203 m는 초기값이며 실제 Pro 장착 검증을 대체하지 않는다.

외부 코드를 복사하지 않았다. EKF는 관측 모델/공분산 확인 전 추가하지 않았다. 지금의 짧은 기억에는 odom SE(2) 변환이 더 가볍고 원인이 명확하다.

## 주요 설정

`params/m8.yaml`:

- speed=0.10: 현재 0 또는 0.10만 발행. 실제 못 움직이는 0.1 미만을 진행으로 취급하지 않는다.
- max_steer_angle=0.42, minimum_turn_radius_m=0.50: 안쪽 바퀴/차량 기하 기준, 기존 bridge 유지.
- vehicle_half_width_m=0.12, front=0.29, rear=0.09: 뒤 차축 기준 차체 외곽. 실제 치수 확인 필요.
- cone_padding_m=0.045, minimum_clearance_m=0.025: 넓어지는 받침의 추가 여유와 분리 여유. 실제 받침이 더 넓으면 늘려야 한다.
- target_hold_sec=0.50, target_hold_distance_m=0.10: 최근 관측점 예측 상한.
- scan_timeout_sec=0.25, resume_scans=2: 오래된 명령 정지, 10 Hz일 때 복귀 확인 약 0.1초.
- motion_enabled=true: 통합 주행 기본 허용. 시험 launch는 기본 motion=false이며 실제 주행 시 명시적으로 true를 준다.

## LIMO 배포 — 일반 pull 대신 아래 사용

이력을 통합했으므로 기존 LIMO의 `git pull --ff-only`는 실패할 수 있다. 아래는 **로컬 수정이 있으면 중단**하고 기존 HEAD를 백업한다. reset --hard는 사용하지 않는다.

```bash
source ~/race_ws/install/setup.bash
m8_update() {
  cd ~/race_ws || return 1
  local m8_pkg m8_repo m8_backup
  m8_pkg=$(colcon list --packages-select jetracer_ros2 --paths-only) || return 1
  [ -n "$m8_pkg" ] || { echo "패키지 경로를 찾지 못함"; return 1; }
  m8_repo=$(git -C "$m8_pkg" rev-parse --show-toplevel) || return 1
  [ -z "$(git -C "$m8_repo" status --porcelain)" ] || {
    echo "로컬 수정이 있어 중단. 먼저 수정사항을 보존해 주세요."; return 1;
  }
  m8_backup="backup/m8-before-update-$(date +%Y%m%d_%H%M%S)"
  git -C "$m8_repo" branch "$m8_backup" || return 1
  git -C "$m8_repo" fetch origin || return 1
  git -C "$m8_repo" switch --detach origin/feature/m8-cone-corridor || return 1
  git -C "$m8_repo" branch -f feature/m8-cone-corridor origin/feature/m8-cone-corridor || return 1
  git -C "$m8_repo" switch feature/m8-cone-corridor || return 1
  git -C "$m8_repo" branch --set-upstream-to=origin/feature/m8-cone-corridor || return 1
  colcon build --symlink-install --packages-select jetracer_ros2 || return 1
  source ~/race_ws/install/setup.bash
}
m8_update
```

### 실행 — 하드웨어 한 번 + 미션 한 번

기존 all_in/매니저/콘/차선 실행을 종료한다. 이미 teleop 하드웨어가 살아 있으면 중복 실행하지 않는다. NoMachine에서:

터미널 1 — 하드웨어가 꺼져 있을 때만:

```bash
source ~/race_ws/install/setup.bash
ros2 launch jetracer_ros2 jetracer_teleop.launch.py
```

터미널 2 — 첫 정지선이 보이는 입구, 리모컨 코드 모드, 통로에 사람이 없는 상태:

```bash
source ~/race_ws/install/setup.bash
ros2 launch jetracer_ros2 m8_cone.launch.py motion:=true record:=true
```

차선·정지선·콘·매니저·bag 기록이 함께 실행된다. `~/bag_m8_local_날짜_시간`에 저장된다. 관측만은 motion:=false. 별도 bag record는 불필요하다. 차선 창에는 차선 제안이 표시되므로 실제 명령은 매니저 로그나 /cmd_vel로 확인한다.

### 실차 확인 목록

1. /cmd_vel 발행자는 매니저 하나, 구독은 기존 cmdvel_to_ackermann.
2. 아커만 모드 경고 없음, /scan 약 10 Hz, /odom 약 50 Hz.
3. 초반에는 phase=watch_entry와 기존 M2 차선 명령, 입구 검출 뒤 phase=approach와 검사된 source=lane, 콘 획득 뒤 phase=cone 및 source=cone와 v=0.100.
4. safe_cmd_vel과 /cmd_vel의 v/omega 일치.
5. 멈추면 no_gate/path_blocked/footprint_blocked/stale_cone/exit_lane 중 정확한 이유.
6. 출구 선 뒤 새 차선 미검출이면 대기, 관측 차선이 생기면 인계.
7. Jetson plan_ms/exit_ms/callback_ms와 센서→출력 age 확인.

첫 시험은 리모컨으로 즉시 정지할 수 있게 저속 1회 수행한다. 실제 받침·차체 치수와 조향 반경 검증이 여전히 필요하다.
