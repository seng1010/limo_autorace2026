"""Width learning commits only after pair validation; final single-side paths agree."""
import numpy as np
import pytest

from jetracer_ros2 import simple_lane as S


def observe(monkeypatch, engine, *, side=None, keep='left', separation=.70, slope=0., now=1.):
    f = np.linspace(.45, 1.4, 40)

    def make_line(comp, which):
        x = np.full(len(f), (-1 if which == 'left' else 1) * separation / 2)
        x += slope * (f-f[0])
        out = S.LaneLine(which, x, f, np.arange(len(f)), np.arange(len(f)), x[0])
        out.zigzag = False
        return out

    keep = side or keep
    picked = {s: ((.01 if s == keep else .2), {}) for s in ('left', 'right')}
    if side is not None:
        picked['right' if side == 'left' else 'left'] = None
    monkeypatch.setattr(engine, '_pick_lines', lambda c: picked)
    monkeypatch.setattr(engine, '_make_line', make_line)
    return engine.process(np.zeros((480, 640, 3), np.uint8), S.DEFAULT_TUNE, now)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('separation', [.70, .90])
def test_rejected_pair_uses_same_trusted_width_as_direct_single(monkeypatch, side, separation):
    direct, paired = S.LaneEngine(), S.LaneEngine()
    for engine in (direct, paired):
        engine.lane_width = .33
        engine.single_width = .43
        engine.tracked = {'left': True, 'right': True}
    r1 = observe(monkeypatch, direct, side=side, separation=separation)
    r2 = observe(monkeypatch, paired, keep=side, separation=separation)
    assert r2['source'] == r1['source'] == side + '_only'
    assert r2['lane_width'] == r1['lane_width'] == .43
    np.testing.assert_allclose(r2['center'], r1['center'])
    np.testing.assert_allclose(r2['target'], r1['target'])
    assert r2['raw_angle'] == pytest.approx(r1['raw_angle'])
    assert paired.lane_width == .33 and paired.single_width == .43


def test_accepted_pair_retains_existing_ema_and_center(monkeypatch):
    engine = S.LaneEngine()
    r = observe(monkeypatch, engine, separation=.46)
    assert r['source'] == 'both'
    assert engine.lane_width == pytest.approx(.445)
    assert engine.single_width == pytest.approx(.436)
    assert r['lane_width'] == pytest.approx(.445)
    np.testing.assert_allclose(r['center'][0], 0., atol=1e-12)


def test_accepted_slanted_pair_does_not_update_straight_memory(monkeypatch):
    engine = S.LaneEngine()
    r = observe(monkeypatch, engine, separation=.46, slope=.3)
    assert r['source'] == 'both'
    assert engine.lane_width != .43
    assert engine.single_width == .43


def test_missing_width_measurement_cannot_update_memory(monkeypatch):
    engine = S.LaneEngine()
    monkeypatch.setattr(S, '_perp_width', lambda l, r: None)
    r = observe(monkeypatch, engine, separation=.46)
    assert r['source'] == 'both'
    assert engine.lane_width == engine.single_width == .43


def test_no_observation_preserves_width_memory():
    engine = S.LaneEngine()
    engine.lane_width, engine.single_width = .38, .41
    engine.last_valid_t = 0.
    r = engine.process(np.zeros((480, 640, 3), np.uint8), S.DEFAULT_TUNE, 100.)
    assert not r['valid']
    assert engine.lane_width == .38 and engine.single_width == .41


@pytest.mark.parametrize('gap', [float('inf'), float('nan')])
def test_nonfinite_pair_disagreement_cannot_train_width(monkeypatch, gap):
    engine = S.LaneEngine()
    merge = engine._merge

    def invalid_pair(cands):
        x, f, actual_gap = merge(cands)
        return x, f, gap if len(cands) == 2 else actual_gap

    monkeypatch.setattr(engine, '_merge', invalid_pair)
    r = observe(monkeypatch, engine, separation=.46)
    assert r['source'] == 'left_only'
    assert engine.lane_width == engine.single_width == r['lane_width'] == .43


@pytest.mark.parametrize('side', ['left', 'right'])
def test_alternating_single_and_rejected_pair_has_no_width_or_target_jump(monkeypatch, side):
    engine = S.LaneEngine()
    engine.lane_width, engine.single_width = .33, .43
    engine.tracked = {'left': True, 'right': True}
    targets = []
    for i in range(20):
        r = observe(monkeypatch, engine, side=side if i % 2 == 0 else None,
                    keep=side, now=1.+i/30.)
        assert r['source'] == side + '_only'
        assert engine.lane_width == .33 and engine.single_width == .43
        assert r['lane_width'] == .43
        targets.append(r['target'])
    np.testing.assert_allclose(targets, np.tile(targets[0], (len(targets), 1)))


@pytest.mark.parametrize('side,sign', [('left', -1), ('right', 1)])
def test_boundary_prediction_uses_width_that_created_previous_center(monkeypatch, side, sign):
    e = S.LaneEngine()
    e.lane_width, e.single_width = .36, .43
    r = observe(monkeypatch, e, side=side, separation=.70)
    assert r['source'] == side + '_only'
    assert e._expected(dict(near_x=sign*.35, near_f=.8), sign) == pytest.approx(0., abs=1e-7)
    # A newly measured width must not retroactively change the previous path's geometry.
    e.lane_width = .50
    assert e._expected(dict(near_x=sign*.35, near_f=.8), sign) == pytest.approx(0., abs=1e-7)
