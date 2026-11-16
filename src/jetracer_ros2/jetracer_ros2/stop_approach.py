"""정지선 앞 감속·정밀 정지 + 완전정지 판정 (ROS 없음, M1·M3 공용).

입력 거리는 stop_line_node 가 내는 값 그대로 (기준 바퀴 -> 정지선, 기본 앞바퀴).

속도 프로파일
    남은 거리 d 가 approach_start_m 보다 멀면 제한 없음 (평소 속도)
    그 안에서는 v = clamp(gain * (d - stop_offset_m), creep_speed, max_speed)
    d <= stop_offset_m 이면 0 (정지 명령)
stop_offset_m 은 "명령을 끊은 뒤 차가 더 미끄러지는 거리"(제동거리)다. 실차에서
몇 번 세워 보고 평균적으로 넘어가는 만큼 넣는다.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StopApproachConfig:
    approach_start_m: float = 0.50     # 이 거리 안에서부터 감속
    gain: float = 0.8                  # [1/s] 남은 거리 -> 속도
    creep_speed: float = 0.06          # 마지막 접근 최저 속도 [m/s]
    max_speed: float = 0.25            # 감속 구간 최고 속도 [m/s]
    stop_offset_m: float = 0.01        # 제동거리 보정 [m]


class StopApproachController:
    def __init__(self, config: StopApproachConfig = StopApproachConfig()):
        self.config = config

    def speed_limit(self, distance_m: Optional[float]) -> Optional[float]:
        """None = 제한 없음, 0.0 = 정지, 그 외 = 이 속도 이하로."""
        cfg = self.config
        if distance_m is None or distance_m > cfg.approach_start_m:
            return None
        remaining = distance_m - cfg.stop_offset_m
        if remaining <= 0.0:
            return 0.0
        return min(cfg.max_speed, max(cfg.creep_speed, cfg.gain * remaining))


class FullStopDetector:
    """odom 속도가 threshold 이하로 hold_sec 동안 유지되면 '완전정지'.

    규정 M3: 완전정지 확인 시점부터 3초 이상 대기 -> 대기 시작 시각이 필요하다.
    """

    def __init__(self, speed_threshold: float = 0.01, hold_sec: float = 0.3):
        self.speed_threshold = float(speed_threshold)
        self.hold_sec = float(hold_sec)
        self._below_since: Optional[float] = None

    def reset(self) -> None:
        self._below_since = None

    def update(self, now: float, speed: Optional[float]) -> Optional[float]:
        """완전정지가 확인되면 그 확인 시각(속도가 떨어진 시각 + hold_sec), 아니면 None.

        심사위원이 "멈췄다"고 보는 시점보다 늦게 세는 쪽이 안전하므로 확인 시각을 쓴다.
        """
        if speed is None or abs(speed) > self.speed_threshold:
            self._below_since = None
            return None
        if self._below_since is None:
            self._below_since = now
        if now - self._below_since >= self.hold_sec:
            return self._below_since + self.hold_sec
        return None
