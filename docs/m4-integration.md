# M4 코드 통합 범위

M4 `fc6470c`를 main `d142030`에 통합했다. 미션 시작·완료와 매니저 연결은 후속 작업이다.

- `jetracer_ros2/mission_parking_node.py`: 기존 main의 매니저용 노드를 유지한다. 기존 `mission/parking/start`, `mission/parking/cmd_vel`, `mission/parking/state` 연결을 그대로 사용한다.
- `jetracer_ros2/parking_timed_test_node.py`: M4 브랜치의 `mission_parking_node.py`를 내용 변경 없이 별도 이름으로 보존했다. 직진/회전/진입 시간 1.7/1.8/0.1초, 직진·회전 속도 0.50m/s의 시험 버전이다.
- `jetracer_ros2/parking_test_node.py`: M4 원본 파일을 유지한다. 시간 1.5/2.5/1.5초, 속도 0.10/0.08m/s의 별도 시험 버전이다.
- `jetracer_ros2/parking_line_detection.py`: 정지선 접근 후 주차하는 시험 코드다.
- `mission_parking_avoid_node.py`: 패키지 디렉터리 바깥의 빈칸 탐색·회피 시험 코드 위치를 유지한다.

위 시험 노드들은 기본적으로 `/cmd_vel`에 직접 출력한다. 전체 주행 launch에 추가하거나 매니저와 동시에 실행하지 않는다. 이번 통합에서는 setup.py와 launch에 새 실행 항목을 등록하지 않았다.

후속 연결 시 채택할 주차 구현과 튜닝값을 정하고, 명령은 `mission/parking/cmd_vel`, 시작·완료는 기존 매니저 인터페이스로 연결한다. 실제 `/cmd_vel` 발행자는 매니저로 유지한다.

M4 브랜치에 함께 들어 있던 M5 시험 파일도 그대로 보존했지만 기존 서행 로직에 연결하지 않았다. 별도 M5 브랜치와 통합할 때 `m5-runner-node.py` 중복 및 판단 모듈 API를 정리해야 한다.
