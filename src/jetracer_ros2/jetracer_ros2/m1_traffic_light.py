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

import math
from dataclasses import dataclass
from typing import Optional

from jetracer_ros2.section_logic import LinePassDetector
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
    lost_line_revert_m: float = 0.30        # 정지 전에 선을 이 거리보다 멀리서 놓치면
                                            # 잘못 본 선으로 보고 계속 주행 (노면 글씨·무늬 등)
    skip_lines: int = 0                     # M1 구간의 앞쪽 정지선을 이 개수만큼 무시
    slow_speed: float = -1.0                # 서행 속도 [m/s] (음수 = 서행 안 함).
                                            # 경기장 M1 에는 '천천히' 노면 표시가 있지만
                                            # 서행 표지판이 없어 YOLO 로는 못 잡는다
    slow_after_lines: int = 1               # 정지선을 이 개수 지난 뒤부터 서행
                                            # (경기장: 횡단보도 앞에 '천천히' 시작선이 하나 있어 1)


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
        # 지나온 정지선 세기 (앞쪽의 다른 흰 가로선에 서지 않도록)
        self._line_detector = LinePassDetector()
        self.lines_seen = 0
        self._last_line_m: Optional[float] = None
        self.start_s: Optional[float] = None
        self.stop_latched = False
        self.hold_start: Optional[float] = None
        self.go_start_s: Optional[float] = None
        self.pass_end_s: Optional[float] = None
        self.full_stop.reset()

    def _armed(self, stop_line_m, odom_s, last_light_s) -> bool:
        if self.lines_seen <= self.config.skip_lines:
            return False                      # 아직 앞쪽 정지선 구간
        line_near = stop_line_m is not None and stop_line_m <= self.config.line_arm_distance_m
        if self.config.trigger == 'light_or_line':
            return line_near
        light_recent = (last_light_s is not None
                        and odom_s - last_light_s <= self.config.light_valid_distance_m)
        return light_recent and line_near

    def _start_go(self, odom_s):
        self.state = 'go'
        self.go_start_s = odom_s

    def _slow_limit(self) -> Optional[float]:
        """서행 구간이면 그 속도, 아니면 None."""
        cfg = self.config
        if cfg.slow_speed > 0.0 and self.lines_seen >= cfg.slow_after_lines:
            return cfg.slow_speed
        return None

    def _decide(self, state: str, speed_limit: Optional[float] = None, **kwargs) -> M1Decision:
        slow = self._slow_limit()
        if slow is not None and state in ('approach', 'passing', 'stopping', 'go'):
            speed_limit = slow if speed_limit is None else min(speed_limit, slow)
        return M1Decision(speed_limit=speed_limit, state=state, **kwargs)

    def update(self, now: float, stop_line_m: Optional[float], speed: Optional[float],
               odom_s: float, light: str, last_light_s: Optional[float] = None) -> M1Decision:
        """light: 확정된 신호 색. last_light_s: 신호등(아무 색)을 마지막으로 본 odom 위치."""
        cfg = self.config
        if self.start_s is None:
            self.start_s = odom_s
        if self.state in ('approach', 'passing') and \
                self._line_detector.update(odom_s, stop_line_m) is not None:
            self.lines_seen += 1
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
                return self._decide('passing')

        if self.state == 'stopping':
            if green and not self.stop_latched and not cfg.stop_on_green_arrival:
                # 정지선에 닿기 전에 녹색 확정 → 서지 않고 출발 (stop_on_green_arrival 이면 일단 정지)
                self._start_go(odom_s)
            else:
                if stop_line_m is not None:
                    self._last_line_m = stop_line_m
                    limit = 0.0 if self.stop_latched else self.approach.speed_limit(stop_line_m)
                elif self.stop_latched:
                    limit = 0.0                       # 이미 정지 명령 (사각지대)
                elif (self._last_line_m is not None
                      and self._last_line_m > cfg.lost_line_revert_m):
                    # 아직 먼데 선이 사라짐 = 잘못 본 선. 그 자리에 서지 말고 계속 간다
                    self.state = 'approach'
                    self._last_line_m = None
                    return self._decide('approach')
                else:
                    limit = 0.0                       # 사각지대로 들어간 선
                if limit == 0.0:
                    self.stop_latched = True
                    confirmed = self.full_stop.update(now, speed)
                    if confirmed is not None:
                        self.state = 'hold'
                        self.hold_start = confirmed
                else:
                    self.full_stop.reset()
                if self.state == 'stopping':
                    return self._decide('stopping', limit)

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
                return self._decide('go')

        if self.state == 'done':
            return M1Decision(done=True, state='done')
        if self.state == 'skipped':
            return M1Decision(done=True, skipped=True, state='skipped')
        return self._decide(self.state)


class CornerLineGate:
    """M8 -> M1 직후 콘 구간 출구 선을 M1 정지선으로 세지 않게 막는다.

    트랙: 콘 구간 출구 선은 왼쪽 직선(콘과 같은 방향), M1 의 선들(천천히 선, 신호등 정지선,
    횡단보도 끝선)은 그 뒤 오른쪽 90도 코너를 돈 위쪽 직선에 있다.
    M8 강제 탈출(idle·stuck·timeout)이 출구 선 앞에서 일어나면 M1 이 출구 선을 첫 선으로 세어
    천천히 선에서 서 버린다 (19:29 bag 재생: M1 시작을 출구 선 앞으로 당기면 그렇게 됨).

    그래서 M8 동안의 진행 방향(odom yaw 중앙값)을 기준으로 잡고, 차가 그 기준에서
    min_turn_deg 이상 돌기 전까지는 M1 이 정지선을 보지 않게 한다. 한 번 돌면 계속 연다.
    - M1 이 코너 뒤에서 시작해도(늦은 탈출) 이미 돌아 있으므로 바로 연다
    - M8 기준이 없으면(start:=M1 시험, yaw 없음) 막지 않는다 = 예전 동작
    - M8 동안 yaw 가 거의 안 변했으면(min_spread_deg 미만) odom 방향을 못 믿는 것으로 보고
      막지 않는다. 방향값이 고장 나 고정이면 영원히 안 열려 M1 을 통째로 놓치기 때문
      (19:29 bag: M8 동안 yaw 62 ~ -33 도로 변함)
    """

    def __init__(self, min_turn_deg: float = 60.0, min_samples: int = 20,
                 min_spread_deg: float = 2.0):
        self.min_turn_deg = float(min_turn_deg)
        self.min_samples = int(min_samples)
        self.min_spread_deg = float(min_spread_deg)
        self._sin: list = []
        self._cos: list = []
        self.reference_deg: Optional[float] = None
        self.open = True

    def clear_reference(self) -> None:
        self._sin.clear()
        self._cos.clear()

    def add_reference(self, yaw_rad: Optional[float]) -> None:
        if yaw_rad is None or not math.isfinite(yaw_rad):
            return
        self._sin.append(math.sin(yaw_rad))
        self._cos.append(math.cos(yaw_rad))

    def start(self) -> None:
        """M1 시작 때 한 번. 기준이 있으면 닫고 시작한다."""
        self.reference_deg = None
        if self.min_turn_deg > 0 and len(self._sin) >= self.min_samples:
            reference = math.atan2(_median(self._sin), _median(self._cos))
            spread = max(abs(math.degrees(math.atan2(math.sin(math.atan2(sn, cs) - reference),
                                                     math.cos(math.atan2(sn, cs) - reference))))
                         for sn, cs in zip(self._sin, self._cos))
            if spread >= self.min_spread_deg:
                self.reference_deg = math.degrees(reference)
        self.open = self.reference_deg is None
        self.clear_reference()

    def turn_deg(self, yaw_rad: Optional[float]) -> Optional[float]:
        if self.reference_deg is None or yaw_rad is None or not math.isfinite(yaw_rad):
            return None
        return abs((math.degrees(yaw_rad) - self.reference_deg + 180.0) % 360.0 - 180.0)

    def allow(self, yaw_rad: Optional[float]) -> bool:
        """True 면 M1 이 정지선을 봐도 된다. yaw 를 모르면 닫힌 상태 유지."""
        if not self.open:
            turned = self.turn_deg(yaw_rad)
            if turned is not None and turned >= self.min_turn_deg:
                self.open = True
        return self.open


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
