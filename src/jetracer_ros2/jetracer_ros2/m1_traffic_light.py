"""M1 신호등 정지·발진 — 판단 로직 (ROS 없음, mission_manager_node 가 매 주기 호출).

규정 (규정집 v3)
  - 3색(적·황·녹) 신호를 인식해 정지·발진. 신호등 옆 광고 모니터가 방해
  - 정지선 ±5cm 완전정지 (정지선 앞, 앞바퀴 기준), 녹색 인지 후 15초 안에 출발
  - 실패: 오차 초과, 정지 미이행, 15초 초과 미출발

입력 light 는 mission_traffic_light_node 가 확정한 'red' / 'yellow' / 'green' / 'unknown'.

상태
  approach  평소 주행. 신호등을 본 뒤(light_valid_distance_m 안) 정지선이 line_arm_distance_m
            안으로 들어오면 → 녹색이면 passing, 아니면 stopping
  stopping  정지선 앞 정밀 정지 (StopApproachController, 정지 래치). 도중에 녹색이 확정되면 go
  hold      정지 상태로 녹색 대기. 녹색 확정 → go. max_wait_sec 지나도 녹색이 없으면 go
            (멈춰서 나머지 미션까지 잃는 것보다 M1만 잃는 쪽을 택함)
  passing   녹색에 도착 → 서지 않고 통과. 정지선까지 stop_switch_min_m 이상 남았는데
            적·황으로 바뀌면 stopping 으로 전환
  go        출발 후 done_distance_m 가면 done
  done      매니저가 다음 미션(M4)으로 넘김
  skipped   신호등을 못 보고 max_approach_distance_m 를 지나침 → M1 포기하고 다음 미션

trigger
  'light'          : 신호등(아무 색)을 본 뒤의 정지선에서만 판단 (기본, 대회용)
  'light_or_line'  : 신호등 없이 정지선만 보여도 판단 (YOLO 없이 테스트할 때)
"""

from dataclasses import dataclass
from typing import Optional

from jetracer_ros2.stop_approach import FullStopDetector, StopApproachController


@dataclass
class M1Config:
    trigger: str = 'light'
    line_arm_distance_m: float = 0.80
    light_valid_distance_m: float = 2.0     # 신호등을 본 뒤 이 거리 안에 나온 정지선만 인정
    stop_on_green_arrival: bool = False     # True 면 녹색에 도착해도 일단 정지 후 출발
    stop_switch_min_m: float = 0.15         # passing 중 적·황 전환 시 이 거리 이상 남았으면 정지
    max_wait_sec: float = 30.0              # 녹색 대기 최대 시간 (0 이하 = 끄기)
    done_distance_m: float = 0.50
    max_approach_distance_m: float = 2.0    # M1 시작 후 판단 못 하고 이만큼 가면 skipped (0 = 끄기)


@dataclass
class M1Decision:
    speed_limit: Optional[float] = None
    done: bool = False
    skipped: bool = False
    state: str = 'approach'


class TrafficLightMission:
    def __init__(self, config: M1Config, approach: StopApproachController,
                 full_stop: FullStopDetector):
        if config.trigger not in ('light', 'light_or_line'):
            raise ValueError("m1 trigger 는 'light' | 'light_or_line'")
        self.config = config
        self.approach = approach
        self.full_stop = full_stop
        self.reset()

    def reset(self) -> None:
        self.state = 'approach'
        self.start_s: Optional[float] = None
        self.stop_latched = False
        self.hold_start: Optional[float] = None
        self.go_start_s: Optional[float] = None
        self.pass_end_s: Optional[float] = None
        self.full_stop.reset()

    def _armed(self, stop_line_m, odom_s, last_light_s) -> bool:
        line_near = stop_line_m is not None and stop_line_m <= self.config.line_arm_distance_m
        if self.config.trigger == 'light_or_line':
            return line_near
        light_recent = (last_light_s is not None
                        and odom_s - last_light_s <= self.config.light_valid_distance_m)
        return light_recent and line_near

    def _start_go(self, odom_s):
        self.state = 'go'
        self.go_start_s = odom_s

    def update(self, now: float, stop_line_m: Optional[float], speed: Optional[float],
               odom_s: float, light: str, last_light_s: Optional[float] = None) -> M1Decision:
        """light: 확정된 신호 색. last_light_s: 신호등(아무 색)을 마지막으로 본 odom 위치."""
        cfg = self.config
        if self.start_s is None:
            self.start_s = odom_s
        green = light == 'green'

        if self.state == 'approach':
            if self._armed(stop_line_m, odom_s, last_light_s):
                if green and not cfg.stop_on_green_arrival:
                    self.state = 'passing'
                    self.pass_end_s = odom_s + stop_line_m + cfg.done_distance_m
                else:
                    self.state = 'stopping'
            elif cfg.max_approach_distance_m > 0 and odom_s - self.start_s >= cfg.max_approach_distance_m:
                self.state = 'skipped'

        if self.state == 'passing':
            if (light in ('red', 'yellow') and stop_line_m is not None
                    and stop_line_m >= cfg.stop_switch_min_m):
                self.state = 'stopping'
            elif odom_s >= self.pass_end_s:
                self.state = 'done'
            else:
                return M1Decision(state='passing')

        if self.state == 'stopping':
            if green and not self.stop_latched and not cfg.stop_on_green_arrival:
                # 정지선에 닿기 전에 녹색 확정 → 서지 않고 출발 (stop_on_green_arrival 이면 일단 정지)
                self._start_go(odom_s)
            else:
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
                    return M1Decision(speed_limit=limit, state='stopping')

        if self.state == 'hold':
            waited = now - self.hold_start
            if green or (cfg.max_wait_sec > 0 and waited >= cfg.max_wait_sec):
                self._start_go(odom_s)
            else:
                return M1Decision(speed_limit=0.0, state='hold')

        if self.state == 'go':
            if odom_s - self.go_start_s >= cfg.done_distance_m:
                self.state = 'done'
            else:
                return M1Decision(state='go')

        if self.state == 'done':
            return M1Decision(done=True, state='done')
        if self.state == 'skipped':
            return M1Decision(done=True, skipped=True, state='skipped')
        return M1Decision(state=self.state)
