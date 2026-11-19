"""lane3_gap: 3차선 입구에서 제자리면 기다리지 않는다 (ENTRY_STILL_SEC)."""
import numpy as np

from jetracer_ros2 import lane3_gap as G


def _empty_scan():
    n = 250
    return (np.full(n, np.inf, np.float32), -np.radians(110.0), np.radians(220.0) / (n - 1), 0.05)


def _armed_supervisor_stopped_at(distance, wait_sec):
    """M7 시작 -> distance 만큼 간 뒤(흰 점선 못 봄) wait_sec 동안 제자리."""
    sup = G.GapSupervisor()
    sup.set_mission('M7')
    t, x = 0.0, 0.0
    while x < distance:
        x += 0.02
        t += 0.1
        sup.step_obs(None, (x, 0.0, 0.0), _empty_scan(), False, t)
    moving_mode = sup.mode
    for _ in range(int(wait_sec / 0.1)):
        t += 0.1
        sup.step_obs(None, (x, 0.0, 0.0), _empty_scan(), False, t)
    return sup, moving_mode


def test_armed_and_still_at_the_entrance_starts_after_one_second():
    sup, moving_mode = _armed_supervisor_stopped_at(G.ARM_AFTER_M + 0.5, 0.8)
    assert moving_mode == 'armed' and sup.mode == 'armed'            # 움직이는 동안·1초 전에는 그대로 기다린다
    sup, _ = _armed_supervisor_stopped_at(G.ARM_AFTER_M + 0.5, 1.3)
    assert sup.mode == 'on'


def test_still_before_arm_distance_does_not_start():
    sup, _ = _armed_supervisor_stopped_at(G.ARM_AFTER_M - 0.4, 5.0)   # 횡단보도·코너에서 서 있는 경우
    assert sup.mode == 'armed'


def test_other_missions_never_start_by_standing_still():
    sup = G.GapSupervisor()
    for name in ('M5', 'M6', 'M3', 'M8', 'M1', 'M4'):
        sup.set_mission(name)
        for i in range(40):
            sup.step_obs(None, (3.0, 0.0, 0.0), _empty_scan(), False, 100.0 + i * 0.1)
        assert sup.mode == 'off'


def test_switch_off_keeps_waiting(monkeypatch):
    monkeypatch.setattr(G, 'ENTRY_STILL_SEC', 0.0)
    sup, _ = _armed_supervisor_stopped_at(G.ARM_AFTER_M + 0.5, 5.0)
    assert sup.mode == 'armed'


def _yellow_wall_obs():
    """앞 0.40 m 를 가로지르는 노란 선만 있고 장애물은 없다 (입구 사선)."""
    x = np.arange(-1.0, 1.0, 0.02)
    return dict(th=0.0, n=0, lines=[], yellow_xy=np.stack([x, np.full(x.size, 0.40)], 1))


def _core_states(move_per_frame, frames):
    core = G.GapCore()
    core.start(0.0, (0.0, 0.0, 0.0))
    out, x = [], 0.0
    for i in range(frames):
        x += move_per_frame
        o = core.step_obs(_yellow_wall_obs(), (x, 0.0, 0.0), _empty_scan(), None, i * 0.1)
        out.append(('pass' if o.get('pass') else o['note'].split()[1]))
    return core, out


def test_entry_hands_over_to_base_driving_while_the_car_moves():
    _, states = _core_states(0.005, 30)
    assert set(states) == {'pass'}


def test_entry_stops_handing_over_when_the_car_stands_still_and_then_reverses():
    core, states = _core_states(0.0, 40)
    assert states[0] == 'pass' and states[5] == 'pass'               # 1초 전까지는 맡긴다
    assert 'BLOCKED' in states[11:]                                   # 1초 제자리 -> 막힘 처리
    assert core.rev_count == 1                                        # 그 뒤 STUCK_SEC 지나 후진


def test_entry_switch_off_hands_over_forever(monkeypatch):
    monkeypatch.setattr(G, 'ENTRY_STILL_SEC', 0.0)
    _, states = _core_states(0.0, 60)
    assert set(states) == {'pass'}


def test_entry_block_switch_off_keeps_only_the_armed_start(monkeypatch):
    monkeypatch.setattr(G, 'ENTRY_STILL_BLOCK', False)
    _, states = _core_states(0.0, 60)
    assert set(states) == {'pass'}                                    # (b) 꺼짐: 맡긴 채로 둔다
    sup, _ = _armed_supervisor_stopped_at(G.ARM_AFTER_M + 0.5, 1.3)
    assert sup.mode == 'on'                                           # (a) 는 그대로
