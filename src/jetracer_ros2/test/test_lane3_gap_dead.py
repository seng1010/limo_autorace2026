"""lane3_gap: 후진을 다 쓴 뒤 '노란 선 무시' 탈출(DEAD_MODE)은 꺼 둔다."""
import numpy as np

from jetracer_ros2 import lane3_gap as G


def _blocked_core(seconds):
    """정면 0.25 m 가 넓게 막힌 채 후진 횟수를 다 쓴 차를 seconds 동안 세워 둔다."""
    n = 250
    amin, inc = -np.radians(110.0), np.radians(220.0) / (n - 1)
    a = amin + np.arange(n) * inc
    r = np.where(np.abs(a) < np.radians(70.0), 0.25 / np.maximum(np.cos(a), 0.2), np.inf).astype(np.float32)
    scan = (r, amin, inc, 0.05)
    core = G.GapCore()
    core.start(0.0, (0.0, 0.0, 0.0))
    core.rev_count = G.REVERSE_MAX
    out = None
    for i in range(int(seconds / 0.1) + 1):
        out = core.step_obs(None, (0.0, 0.0, 0.0), scan, None, i * 0.1)
    return core, out


def test_dead_mode_is_off_and_car_stays_stopped():
    assert G.DEAD_MODE is False
    core, out = _blocked_core(G.DEAD_SEC + 2.0)
    assert 'L3G BLOCKED' in out['note'] and out['v'] == 0.0
    assert core.dead_at is None
    assert '무시' not in core.message


def test_dead_mode_on_behaves_as_before(monkeypatch):
    monkeypatch.setattr(G, 'DEAD_MODE', True)
    core, _ = _blocked_core(G.DEAD_SEC + 2.0)
    assert core.dead_at is not None
