"""M5 서행 구간(방지턱) — 판단 로직 (ROS 없음, mission_manager_node 가 매 주기 호출).

규정 (규정집 v3)
  - 서행 표지판(또는 미션 시작선) 인식 후 속도를 감속하여 방지턱 통과
  - 감속 정도: 기본 주행 속도의 1/2 수준 (최대 0.2 m/s 이하)
  - 실패: 미감속 통과(육안상 차이 없음), 방지턱 통과 불능(멈춤)

동작 방식
  1. approach: 평소 속도(base_speed)로 주행하며 정지선(미션 시작선) 접근 감시
               (trigger='sign_and_line' 일 때는 서행/방지턱 표지판을 본 뒤의 선만 인정)
  2. slowing : 정지선에 앞바퀴가 닿으면(stop_line_m <= trigger_distance_m) 서행 시작.
               자동 계산된 서행 속도(min(0.2, base_speed * 0.5))로 slow_duration_sec 동안 유지.
  3. done    : 서행 유지 시간이 끝나면 원래 속도로 복귀하고 매니저가 다음 미션으로 넘김.
  4. skipped : 선을 못 보고 max_approach_distance_m 를 지나치면 미션 포기 후 다음 미션.
"""

from dataclasses import dataclass
from typing import Optional

from jetracer_ros2.section_logic import LinePassDetector


# =====================================================================
# [튜닝용 파라미터] params.yaml 또는 실행 시 아래 값들만 조정하면 됩니다.
# =====================================================================
@dataclass
class M5Config:
    base_speed: float = 0.35              # 1. 기본 주행 속도 [m/s] (예: 1.0 입력 시에도 아래 규칙으로 자동 제한)
    slow_duration_sec: float = 7.0        # 2. 총 몇 초 동안 서행할지 [초]
    trigger: str = 'line_only'            # 'line_only': 선에 닿으면 바로 서행 (테스트/기본)
                                          # 'sign_and_line': YOLO 서행 표지판을 본 뒤의 선에서만 서행
    trigger_distance_m: float = 0.03      # 선에 닿았다고 판정할 남은 거리 [m] (앞바퀴 기준 3cm 이내)
    sign_valid_distance_m: float = 2.0    # 표지판을 본 뒤 이 거리 안에 나온 선만 인정 [m]
    skip_lines: int = 0                   # M5 구간 진입 전 무시할 앞쪽 가로선 개수
    max_approach_distance_m: float = 2.5  # M5 시작 후 선을 못 찾고 이만큼 가면 skipped (0 = 끄기)
    max_allowed_slow_speed: float = 0.20  # 규정상 서행 최대 속도 상한선 [m/s] (0.2 m/s 이하)

    @property
    def computed_slow_speed(self) -> float:
        """기본 주행 속도의 1/2 수준이면서 동시에 규정 상한(0.2 m/s) 이하로 자동 계산."""
        return min(self.max_allowed_slow_speed, self.base_speed * 0.5)


@dataclass
class M5Decision:
    speed_limit: Optional[float] = None
    done: bool = False
    skipped: bool = False
    state: str = 'approach'


class SlowBumpMission:
    def __init__(self, config: M5Config = M5Config()):
        if config.trigger not in ('line_only', 'sign_and_line'):
            raise ValueError("m5 trigger 는 'line_only' | 'sign_and_line'")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = 'approach'
        self._line_detector = LinePassDetector()
        self.lines_seen = 0
        self.start_s: Optional[float] = None
        self.slow_start_time: Optional[float] = None

    def _line_touched(self, stop_line_m: Optional[float], odom_s: float,
                      last_sign_s: Optional[float]) -> bool:
        """무시할 선을 지나친 뒤, 선에 앞바퀴가 닿았는지 확인."""
        if self.lines_seen < self.config.skip_lines:
            return False

        # stop_line_node 가 사각지대 진입 후에도 odom 으로 0.0m 까지 내려주므로 닿는 순간 포착 가능
        touched = (stop_line_m is not None
                   and stop_line_m <= self.config.trigger_distance_m)
        if not touched:
            return False

        if self.config.trigger == 'line_only':
            return True

        sign_recent = (last_sign_s is not None
                       and (odom_s - last_sign_s) <= self.config.sign_valid_distance_m)
        return sign_recent

    def update(self, now: float, stop_line_m: Optional[float], odom_s: float,
               last_sign_s: Optional[float] = None) -> M5Decision:
        """매 제어 주기마다 mission_manager_node 가 호출.

        Args:
            now: 현재 시각 [초]
            stop_line_m: stop_line_node 가 발행한 기준 바퀴 -> 정지선 거리 [m] (없으면 None)
            odom_s: 누적 주행거리 [m]
            last_sign_s: YOLO가 서행/방지턱 표지판을 마지막으로 본 odom 위치 [m] (없으면 None)
        """
        cfg = self.config
        if self.start_s is None:
            self.start_s = odom_s

        # 1. approach: 평소 속도(base_speed)로 주행하며 정지선 접촉 대기
        if self.state == 'approach':
            if self._line_touched(stop_line_m, odom_s, last_sign_s):
                self.state = 'slowing'
                self.slow_start_time = now
            else:
                # 지나친 선 카운트 (skip_lines 설정 시 앞선 무시용)
                if self._line_detector.update(odom_s, stop_line_m) is not None:
                    self.lines_seen += 1

                if (cfg.max_approach_distance_m > 0
                        and (odom_s - self.start_s) >= cfg.max_approach_distance_m):
                    self.state = 'skipped'
                else:
                    return M5Decision(speed_limit=cfg.base_speed, state='approach')

        # 2. slowing: 선에 닿은 시점부터 slow_duration_sec 동안 서행 유지
        if self.state == 'slowing':
            elapsed = now - self.slow_start_time
            if elapsed < cfg.slow_duration_sec:
                return M5Decision(
                    speed_limit=cfg.computed_slow_speed,
                    state='slowing',
                )
            else:
                self.state = 'done'

        # 3. 종료 상태 반환
        if self.state == 'done':
            return M5Decision(speed_limit=cfg.base_speed, done=True, state='done')
        if self.state == 'skipped':
            return M5Decision(speed_limit=cfg.base_speed, done=True, skipped=True, state='skipped')

        return M5Decision(speed_limit=cfg.base_speed, state=self.state)
