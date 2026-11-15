"""M5 서행 구간(방지턱) — 판단 로직 (ROS 없음).

동작 방식
  1. approach: 평소 속도(base_speed)로 주행하며 정지선 접근 감시
  2. slowing : 정지선에 닿으면(stop_line_m <= trigger_distance_m) 서행 시작.
               서행 속도(min(0.2, base_speed * 0.5))로 slow_duration_sec 동안 유지.
  3. done    : 서행 시간이 끝나면 원래 속도로 복귀.
"""

from dataclasses import dataclass
from typing import Optional


# =====================================================================
# [튜닝용 파라미터]
# =====================================================================
@dataclass
class M5Config:
    base_speed: float = 0.35              # 기본 주행 속도 [m/s]
    slow_duration_sec: float = 7.0        # 서행 유지 시간 [초]
    trigger_distance_m: float = 0.01      # 선에 닿았다고 판정할 남은 거리 [m]
    max_allowed_slow_speed: float = 0.20  # 규정상 서행 최대 속도 [m/s]

    @property
    def computed_slow_speed(self) -> float:
        """기본 속도의 1/2, 단 0.2 m/s 이하."""
        return min(self.max_allowed_slow_speed, self.base_speed * 0.5)


class SlowBumpMission:
    def __init__(self, config: M5Config = M5Config()):
        self.config = config
        self.state = 'approach'
        self.slow_start_time: Optional[float] = None

    def update(self, now: float, stop_line_m: Optional[float]) -> float:
        """현재 출력할 속도 [m/s] 를 반환.

        Args:
            now: 현재 시각 [초]
            stop_line_m: 정지선까지 거리 [m] (선이 없으면 None)
        """
        cfg = self.config

        if self.state == 'approach':
            if stop_line_m is not None and stop_line_m <= cfg.trigger_distance_m:
                self.state = 'slowing'
                self.slow_start_time = now
            else:
                return cfg.base_speed

        if self.state == 'slowing':
            if now - self.slow_start_time < cfg.slow_duration_sec:
                return cfg.computed_slow_speed
            self.state = 'done'

        return cfg.base_speed
