"""M3 보행자 표지판 정지 — 판단 로직 (ROS 없음, mission_manager_node 가 매 주기 호출).

규정 (규정집 v3)
  - 보행자 표지판 인식 후 횡단보도 정지선 ±5cm 완전정지 (정지선 앞, 앞바퀴 기준.
    v3 III-아 '뒷바퀴'는 진입·통과·시간 카운트로 해석)
  - 완전정지 확인 시점부터 3초 이상 대기 후 발진, 15초 안에 출발
  - 실패: 정지 미이행, 오차 초과, 3초 미만 대기, 15초 초과

상태
  approach  평소 주행. 보행자 표지판을 본 뒤 sign_valid_distance_m 안에서 정지선이
            line_arm_distance_m 안으로 들어오면 stopping
  stopping  정지선까지 감속·정밀 정지 (StopApproachController). 정지 명령은 래치,
            정지선을 놓치면 그 자리에 정지
  hold      완전정지 확인 시각부터 hold_sec 동안 정지
  go        출발. 표지판·정지선 무시, done_distance_m 이동하면 done
  done      매니저가 다음 미션(M7)으로 넘김
  skipped   표지판을 못 보고 max_approach_distance_m 를 지나침 → M3 포기하고 다음 미션
            (M3 점수만 잃고 나머지 미션은 계속)

trigger
  'sign'         : 표지판을 본 뒤의 정지선에서만 정지 (기본, 대회용).
                   M3 구간에도 다른 흰 가로선(시작선·체크포인트 테이프 등)이 있어서
                   정지선만 보고 서면 엉뚱한 곳에 설 수 있음
  'sign_or_line' : 표지판 없이 정지선만 보여도 정지 (YOLO 없이 테스트할 때만)
  'line'         : 정지선만
"""

from dataclasses import dataclass
from typing import Optional

from jetracer_ros2.stop_approach import FullStopDetector, StopApproachController


@dataclass
class M3Config:
    trigger: str = 'sign'
    line_arm_distance_m: float = 0.80   # 정지선이 이 거리 안이면 stopping 시작
    sign_valid_distance_m: float = 1.5  # 표지판을 본 뒤 이 거리 안에 나온 정지선만 인정
    max_approach_distance_m: float = 1.5  # M3 시작 후 정지 못 하고 이만큼 가면 skipped (0 이하 = 끄기)
    hold_sec: float = 3.5               # 규정 3초 + 여유
    done_distance_m: float = 0.50       # 출발 후 이만큼 가면 M3 완료


@dataclass
class M3Decision:
    speed_limit: Optional[float] = None  # None = 평소, 0.0 = 정지
    done: bool = False
    skipped: bool = False                # done 이면서 정지는 못 함 (표지판 못 봄)
    state: str = 'approach'


class PedestrianStopMission:
    def __init__(self, config: M3Config, approach: StopApproachController,
                 full_stop: FullStopDetector):
        if config.trigger not in ('sign_or_line', 'sign', 'line'):
            raise ValueError("m3 trigger 는 'sign_or_line' | 'sign' | 'line'")
        self.config = config
        self.approach = approach
        self.full_stop = full_stop
        self.reset()

    def reset(self) -> None:
        self.state = 'approach'
        self.sign_seen = False
        self.sign_s: Optional[float] = None
        self.start_s: Optional[float] = None
        self.stop_latched = False
        self.hold_start: Optional[float] = None
        self.go_start_s: Optional[float] = None
        self.full_stop.reset()

    def _armed(self, stop_line_m: Optional[float], odom_s: float) -> bool:
        line_near = stop_line_m is not None and stop_line_m <= self.config.line_arm_distance_m
        trigger = self.config.trigger
        if trigger == 'sign':
            sign_recent = (self.sign_seen and self.sign_s is not None
                           and odom_s - self.sign_s <= self.config.sign_valid_distance_m)
            return sign_recent and line_near
        if trigger == 'line':
            return line_near
        return line_near            # sign_or_line: 표지판만 보고 선이 아직 멀면 계속 주행

    def update(self, now: float, stop_line_m: Optional[float], speed: Optional[float],
               odom_s: float, last_sign_s: Optional[float] = None) -> M3Decision:
        """last_sign_s: 보행자 표지판을 마지막으로 본 위치(odom 누적거리).
        매니저가 미션과 상관없이 기억해서 넘긴다 (M3 시작 전 M6 구간에서 먼저 보일 수 있음)."""
        if self.start_s is None:
            self.start_s = odom_s
        if last_sign_s is not None and self.state in ('approach', 'stopping'):
            self.sign_seen = True
            self.sign_s = last_sign_s

        if self.state == 'approach' and self._armed(stop_line_m, odom_s):
            self.state = 'stopping'

        limit_m = self.config.max_approach_distance_m
        if (self.state == 'approach' and limit_m > 0.0
                and odom_s - self.start_s >= limit_m):
            self.state = 'skipped'

        if self.state == 'stopping':
            # 한 번 정지 명령이 나가면 거리 값이 흔들려도 다시 움직이지 않는다 (래치).
            # 정지선을 놓치면(지나쳤거나 추적 실패) 그 자리에 선다 — 계속 달리는 것보다 낫다.
            if self.stop_latched or stop_line_m is None:
                limit = 0.0
            else:
                limit = self.approach.speed_limit(stop_line_m)
            if limit == 0.0:
                self.stop_latched = True
                confirmed = self.full_stop.update(now, speed)
                if confirmed is not None:
                    self.state = 'hold'
                    self.hold_start = confirmed
            else:
                self.full_stop.reset()
            if self.state == 'stopping':
                return M3Decision(speed_limit=limit, state='stopping')

        if self.state == 'hold':
            if now - self.hold_start >= self.config.hold_sec:
                self.state = 'go'
                self.go_start_s = odom_s
            else:
                return M3Decision(speed_limit=0.0, state='hold')

        if self.state == 'go':
            if odom_s - self.go_start_s >= self.config.done_distance_m:
                self.state = 'done'
            else:
                return M3Decision(state='go')

        if self.state == 'done':
            return M3Decision(done=True, state='done')
        if self.state == 'skipped':
            return M3Decision(done=True, skipped=True, state='skipped')
        return M3Decision(state=self.state)
