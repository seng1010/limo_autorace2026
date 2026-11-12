"""M3 보행자 표지판 정지 — 판단 로직 (ROS 없음, mission_manager_node 가 매 주기 호출).

규정 (규정집 v3)
  - 보행자 표지판 인식 후 횡단보도 정지선 ±5cm 완전정지 (정지선 앞, 앞바퀴 기준.
    v3 III-아 '뒷바퀴'는 진입·통과·시간 카운트로 해석)
  - 완전정지 확인 시점부터 3초 이상 대기 후 발진, 15초 안에 출발
  - 실패: 정지 미이행, 오차 초과, 3초 미만 대기, 15초 초과

상태
  approach  평소 주행. 표지판(pedestrian) 또는 정지선을 보면 stopping
  stopping  정지선까지 감속·정밀 정지 (StopApproachController). 정지 명령은 래치,
            정지선을 놓치면 그 자리에 정지
  hold      완전정지 확인 시각부터 hold_sec 동안 정지
  go        출발. 표지판·정지선 무시, done_distance_m 이동하면 done
  done      매니저가 다음 미션(M7)으로 넘김

trigger
  'sign_or_line' : 표지판 또는 정지선 (기본. YOLO 모델 없을 때도 테스트 가능)
  'sign'         : 표지판을 봐야만 (다른 가로선에 잘못 서는 것 방지)
  'line'         : 정지선만
"""

from dataclasses import dataclass
from typing import Optional

from jetracer_ros2.stop_approach import FullStopDetector, StopApproachController


@dataclass
class M3Config:
    trigger: str = 'sign_or_line'
    line_arm_distance_m: float = 0.80   # 정지선이 이 거리 안이면 stopping 시작
    hold_sec: float = 3.5               # 규정 3초 + 여유
    done_distance_m: float = 0.50       # 출발 후 이만큼 가면 M3 완료


@dataclass
class M3Decision:
    speed_limit: Optional[float] = None  # None = 평소, 0.0 = 정지
    done: bool = False
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
        self.stop_latched = False
        self.hold_start: Optional[float] = None
        self.go_start_s: Optional[float] = None
        self.full_stop.reset()

    def _armed(self, stop_line_m: Optional[float]) -> bool:
        line_near = stop_line_m is not None and stop_line_m <= self.config.line_arm_distance_m
        trigger = self.config.trigger
        if trigger == 'sign':
            return self.sign_seen and line_near
        if trigger == 'line':
            return line_near
        return line_near            # sign_or_line: 표지판만 보고 선이 아직 멀면 계속 주행

    def update(self, now: float, stop_line_m: Optional[float], speed: Optional[float],
               odom_s: float, pedestrian_seen: bool = False) -> M3Decision:
        if pedestrian_seen and self.state in ('approach', 'stopping'):
            self.sign_seen = True

        if self.state == 'approach' and self._armed(stop_line_m):
            self.state = 'stopping'

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
        return M3Decision(state=self.state)
