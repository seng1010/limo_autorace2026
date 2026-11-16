"""YOLO 감지 -> mission/yolo_event 이벤트 확정 (ROS 없음, yolo_detection_node 가 사용).

클래스마다 따로 센다. 예전에는 한 프레임에서 **신뢰도 1등 하나만** 보고 세어서,
보행자 표지판과 다른 것이 같이 보이면 점수가 낮은 쪽은 한 번도 발행되지 않았고,
둘이 번갈아 1등이 되면 카운트가 계속 리셋돼 **둘 다** 발행되지 않았다.

이제 이번 프레임에 보인 클래스는 각각 카운트를 올리고, 안 보인 클래스는 0으로 되돌린다.
카운트가 confirm_frames 에 **도달하는 순간 한 번만** 발행한다(매 프레임 반복 발행 방지).
사라졌다가 다시 보이면 다시 센다.
"""

from typing import Dict, Iterable, List


class StableEventTracker:
    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = max(1, int(confirm_frames))
        self._counts: Dict[str, int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def update(self, events: Iterable[str]) -> List[str]:
        """이번 프레임에 보인 이벤트 이름들 -> 이번에 새로 확정된 이벤트 목록."""
        seen = {event for event in events if event}
        confirmed = []
        for event in seen:
            count = self._counts.get(event, 0) + 1
            self._counts[event] = count
            if count == self.confirm_frames:          # 도달하는 순간 1회만
                confirmed.append(event)
        for event in list(self._counts):
            if event not in seen:
                del self._counts[event]
        return sorted(confirmed)
