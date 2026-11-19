"""구간 전환 판단 (ROS 없음, section_node 가 매 주기 호출).

하는 일
  1. 정지선 거리(stop_line_node) + odom 누적거리로 "정지선을 지난 위치"를 뽑는다.
  2. 지금 미션이 시작된 뒤 정해진 거리 범위에서 정지선을 지나면 `{미션}_DONE` 을 낸다.
     정지선을 놓쳐도 timeout 거리에서 완료시켜, 한 번의 실수가 뒤 미션으로 번지지 않게 한다.
  3. 서행 구간(미션 시작 기준 거리 범위)에 있으면 slow=True.
     서행 구간은 미션이 넘어간 뒤에도 끝 거리까지 유지된다 (지그재그 끝 ~ 횡단보도 정지선).

거리는 전부 odom 누적 주행거리 [m]. 정지선 거리는 '앞바퀴 -> 정지선' (stop_line_node 기준).
음수 설정값은 '끄기'.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SectionRule:
    line_after_m: float = -1.0   # 미션 시작 후 이 거리 뒤에 나오는 첫 정지선에서 완료 (음수 = 정지선 안 씀)
    timeout_m: float = -1.0      # 정지선과 상관없이 이 거리에서 완료 (음수 = 끄기)
    slow_from_m: float = -1.0    # 서행 시작 (미션 시작 기준, 음수 = 서행 없음)
    slow_to_m: float = -1.0      # 서행 끝
    slow_line_after_m: float = -1.0  # 이 거리 뒤에 처음 보이는 정지선에서 서행 시작 (음수 = 안 씀).
                                     # 선을 놓치면 slow_from_m 에서 시작. 끝은 똑같이 slow_to_m
    slow_speed: float = -1.0     # 이 미션 서행 구간의 속도 [m/s] (음수 = section_node 공통 slow_speed)


class LinePassDetector:
    """정지선 거리 스트림 -> 새 정지선의 위치(누적거리). 같은 선은 한 번만 낸다."""

    def __init__(self, min_frames: int = 2, same_line_m: float = 0.12,
                 refractory_m: float = 0.30, max_distance_m: float = 0.60):
        self.min_frames = int(min_frames)
        self.same_line_m = float(same_line_m)
        self.refractory_m = float(refractory_m)
        self.max_distance_m = float(max_distance_m)
        self._candidate_s: Optional[float] = None
        self._count = 0
        self.last_line_s: Optional[float] = None

    def update(self, s: float, distance: Optional[float]) -> Optional[float]:
        if distance is None or distance < 0.0 or distance > self.max_distance_m:
            self._candidate_s, self._count = None, 0
            return None
        line_s = s + distance
        if self._candidate_s is not None and abs(line_s - self._candidate_s) <= self.same_line_m:
            self._count += 1
            self._candidate_s += (line_s - self._candidate_s) / self._count   # 평균
        else:
            self._candidate_s, self._count = line_s, 1
        if self._count < self.min_frames:
            return None
        if self.last_line_s is not None and self._candidate_s <= self.last_line_s + self.refractory_m:
            return None                      # 이미 낸 선
        self.last_line_s = self._candidate_s
        return self._candidate_s


@dataclass
class SectionOutput:
    event: Optional[str] = None      # 'M5_DONE' 등. 미션이 바뀔 때까지 계속 나온다 (노드가 주기 제한)
    slow: bool = False
    reason: str = ''                 # 완료 이유 ('line' / 'timeout')
    travelled_m: float = 0.0         # 지금 미션에서 간 거리
    line_s: Optional[float] = None   # 이번 주기에 새로 확인된 정지선 위치
    slow_speed: float = -1.0         # 서행 중이면 그 구간의 속도 (음수 = 공통 slow_speed)


class SectionTracker:
    def __init__(self, rules: Dict[str, SectionRule], detector: Optional[LinePassDetector] = None):
        self.rules = dict(rules)
        self.detector = detector or LinePassDetector()
        self.active: Optional[str] = None
        self.start_s = 0.0
        self._done_line_s: Optional[float] = None   # 이 선에 앞바퀴가 닿으면 완료
        self._done_reason = ''
        self._slow_line_used = False
        self._slow_zones: List[Tuple[float, float, float]] = []   # (시작, 끝, 속도)

    def _enter(self, mission: str, s: float) -> None:
        self.active = mission
        self.start_s = s
        self._done_line_s = None
        self._done_reason = ''
        self._slow_line_used = False
        rule = self.rules.get(mission)
        if rule and rule.slow_from_m >= 0.0 and rule.slow_to_m > rule.slow_from_m:
            self._slow_zones.append((s + rule.slow_from_m, s + rule.slow_to_m, rule.slow_speed))

    def update(self, s: float, active: Optional[str],
               stop_line_m: Optional[float]) -> SectionOutput:
        new_line = self.detector.update(s, stop_line_m)
        if active is None:
            return SectionOutput(line_s=new_line)
        if active != self.active:
            self._enter(active, s)
        rule = self.rules.get(active)
        if (rule is not None and new_line is not None and not self._slow_line_used
                and rule.slow_line_after_m >= 0.0 and rule.slow_to_m >= 0.0
                and new_line - self.start_s >= rule.slow_line_after_m
                and s < self.start_s + rule.slow_to_m):
            self._slow_line_used = True               # 서행 표시 선이 보인 순간부터 서행
            self._slow_zones.append((s, self.start_s + rule.slow_to_m, rule.slow_speed))
        self._slow_zones = [z for z in self._slow_zones if s <= z[1]]
        active_zones = [z for z in self._slow_zones if z[0] <= s <= z[1]]
        out = SectionOutput(slow=bool(active_zones),
                            travelled_m=s - self.start_s, line_s=new_line,
                            slow_speed=min((z[2] for z in active_zones if z[2] > 0.0), default=-1.0))
        if rule is None:
            return out
        if self._done_reason:                       # 이미 완료 판정 -> 매니저가 넘길 때까지 유지
            out.event, out.reason = f'{active}_DONE', self._done_reason
            return out
        if (new_line is not None and self._done_line_s is None and rule.line_after_m >= 0.0
                and new_line - self.start_s >= rule.line_after_m):
            self._done_line_s = new_line
        if self._done_line_s is not None and s >= self._done_line_s:
            self._done_reason = 'line'
        elif rule.timeout_m >= 0.0 and s - self.start_s >= rule.timeout_m:
            self._done_reason = 'timeout'
        if self._done_reason:
            out.event, out.reason = f'{active}_DONE', self._done_reason
        return out
