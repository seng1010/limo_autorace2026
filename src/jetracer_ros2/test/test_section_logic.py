from jetracer_ros2.section_logic import LinePassDetector, SectionRule, SectionTracker


def drive(tracker, active_fn, lines, total_m, step=0.01, see_from=0.53, see_to=-0.05):
    """lines: 정지선 위치(누적거리). 앞바퀴 기준 see_from~see_to 범위에서 보인다고 가정."""
    log = []
    s = 0.0
    active = active_fn(None, None)
    while s <= total_m:
        distance = None
        for line in lines:
            if see_to <= line - s <= see_from:
                distance = max(0.0, line - s)
                break
        out = tracker.update(s, active, distance)
        log.append((round(s, 3), active, out))
        active = active_fn(active, out.event)
        s += step
    return log


def manager(order):
    def step(active, event):
        if active is None:
            return order[0]
        if event == f'{active}_DONE' and order.index(active) + 1 < len(order):
            return order[order.index(active) + 1]
        return active
    return step


RULES = {
    'M5': SectionRule(line_after_m=6.0, timeout_m=10.5, slow_from_m=1.2, slow_to_m=5.0),
    'M6': SectionRule(timeout_m=1.0, slow_from_m=0.0, slow_to_m=2.6),
}


def first_s(log, mission):
    return next(s for s, active, _ in log if active == mission)


def test_line_detector_reports_each_line_once():
    det = LinePassDetector()
    hits = []
    s = 0.0
    while s < 3.0:
        d = None
        for line in (1.0, 2.0):
            if -0.05 <= line - s <= 0.5:
                d = max(0.0, line - s)
        hit = det.update(s, d)
        if hit is not None:
            hits.append(hit)
        s += 0.01
    assert len(hits) == 2
    assert abs(hits[0] - 1.0) < 0.03 and abs(hits[1] - 2.0) < 0.03


def test_single_frame_noise_is_ignored():
    det = LinePassDetector(min_frames=2)
    assert det.update(0.0, 0.4) is None
    assert det.update(0.01, None) is None
    assert det.update(0.02, 0.2) is None      # 다른 위치 -> 새 후보 1프레임


def test_m5_ends_at_line_inside_window_and_early_line_is_ignored():
    tracker = SectionTracker(RULES)
    # 1.45 m 의 선(서행 시작 표시)은 무시, 8.8 m 의 선에서 M5 완료
    log = drive(tracker, manager(('M5', 'M6', 'M3')), lines=(1.45, 8.8), total_m=12.0)
    assert abs(first_s(log, 'M6') - 8.8) < 0.05
    assert abs(first_s(log, 'M3') - (8.8 + 1.0)) < 0.05


def test_m5_times_out_when_line_missed():
    tracker = SectionTracker(RULES)
    log = drive(tracker, manager(('M5', 'M6', 'M3')), lines=(), total_m=13.0)
    assert abs(first_s(log, 'M6') - 10.5) < 0.05


def test_slow_zones_and_carry_over_into_next_mission():
    tracker = SectionTracker(RULES)
    log = drive(tracker, manager(('M5', 'M6', 'M3')), lines=(8.8,), total_m=13.0)
    slow = {s: out.slow for s, _, out in log}
    assert not slow[1.0] and slow[1.5] and slow[4.9] and not slow[5.2]
    assert not slow[8.5]
    m6 = first_s(log, 'M6')
    assert slow[round(m6 + 0.5, 2)]
    assert slow[round(m6 + 2.0, 2)]           # M3 로 넘어간 뒤에도 서행 유지
    assert not slow[round(m6 + 2.8, 2)]


def test_event_repeats_until_manager_advances_then_stops():
    tracker = SectionTracker({'M5': SectionRule(timeout_m=1.0)})
    out = tracker.update(0.0, 'M5', None)
    assert out.event is None
    assert tracker.update(1.01, 'M5', None).event == 'M5_DONE'
    assert tracker.update(1.05, 'M5', None).event == 'M5_DONE'
    assert tracker.update(1.10, 'M6', None).event is None     # 규칙 없는 미션


def test_no_active_mission_gives_no_event():
    tracker = SectionTracker(RULES)
    assert tracker.update(20.0, None, None).event is None


def test_mission_start_is_measured_from_where_it_became_active():
    tracker = SectionTracker({'M8': SectionRule(line_after_m=1.0, timeout_m=5.0)})
    tracker.update(30.0, 'M2', None)
    log = []
    s = 31.0
    while s < 37.0:
        d = max(0.0, 33.5 - s) if -0.05 <= 33.5 - s <= 0.5 else None
        out = tracker.update(s, 'M8', d)
        log.append((s, out.event))
        s += 0.01
    done_s = next(s for s, e in log if e)
    assert abs(done_s - 33.5) < 0.05


def test_slow_starts_when_marking_line_is_seen_and_falls_back_to_distance():
    rules = {'M5': SectionRule(line_after_m=8.6, timeout_m=9.6, slow_line_after_m=0.8,
                               slow_from_m=1.7, slow_to_m=5.0)}
    seen = drive(SectionTracker(rules), manager(('M5', 'M6')), lines=(1.52,), total_m=6.0)
    slow = {s: out.slow for s, _, out in seen}
    assert not slow[0.9]
    assert slow[1.1] and slow[1.5] and slow[4.9] and not slow[5.1]     # 선이 보이는 1.0 m 부근부터
    missed = drive(SectionTracker(rules), manager(('M5', 'M6')), lines=(), total_m=6.0)
    slow = {s: out.slow for s, _, out in missed}
    assert not slow[1.6] and slow[1.8] and slow[4.9] and not slow[5.1]


def test_line_before_window_does_not_start_slow():
    rules = {'M5': SectionRule(slow_line_after_m=0.8, slow_from_m=1.7, slow_to_m=5.0)}
    log = drive(SectionTracker(rules), manager(('M5',)), lines=(0.3,), total_m=3.0)
    slow = {s: out.slow for s, _, out in log}
    assert not slow[0.5] and not slow[1.5] and slow[1.8]
