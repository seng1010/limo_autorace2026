from types import SimpleNamespace

import numpy as np
import pytest

from jetracer_ros2.parking_logic import (
    EmptyBayConfirmation, parking_motion_verified, parking_phase_reached,
    scan_sector_percentile, scan_sector_statistics)


def make_scan(side_distance, front_distance=3.0):
    angles = np.radians(np.arange(-180.0, 180.0, 1.0))
    ranges = np.full(angles.shape, 3.0, dtype=np.float32)
    side = (angles >= np.radians(-105.0)) & (angles <= np.radians(-75.0))
    front = (angles >= np.radians(-15.0)) & (angles <= np.radians(15.0))
    ranges[side] = side_distance
    ranges[front] = front_distance
    return SimpleNamespace(
        ranges=ranges,
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        range_min=0.05,
        range_max=12.0,
    )


def test_side_sector_separates_occupied_bay_from_open_bay():
    occupied = make_scan(0.42)
    open_bay = make_scan(1.2)

    assert scan_sector_percentile(occupied, -105.0, -75.0) == pytest.approx(0.42)
    assert scan_sector_percentile(open_bay, -105.0, -75.0) == pytest.approx(1.2)


def test_open_bay_confirmation_requires_consecutive_scans():
    confirmation = EmptyBayConfirmation(0.65, 3)

    assert confirmation.update(1.2) is False
    assert confirmation.update(1.2) is False
    assert confirmation.update(0.42) is False
    assert confirmation.update(1.2) is False
    assert confirmation.update(1.2) is False
    assert confirmation.update(1.2) is True


def test_no_return_ranges_count_as_open_and_nan_ranges_are_ignored():
    scan = make_scan(3.0)
    angles = np.degrees(
        scan.angle_min + np.arange(scan.ranges.size) * scan.angle_increment)
    side = (angles >= -105.0) & (angles <= -75.0)
    scan.ranges[side] = np.inf
    assert scan_sector_percentile(scan, -105.0, -75.0) == pytest.approx(scan.range_max)

    scan.ranges[side] = np.nan
    assert scan_sector_percentile(scan, -105.0, -75.0) is None


def test_sector_statistics_report_far_returns_and_no_return_counts():
    scan = make_scan(1.2)
    angles = np.degrees(
        scan.angle_min + np.arange(scan.ranges.size) * scan.angle_increment)
    side_indices = np.flatnonzero((angles >= -105.0) & (angles <= -75.0))
    scan.ranges[side_indices[:10]] = 0.4
    scan.ranges[side_indices[10:15]] = np.inf

    stats = scan_sector_statistics(scan, -105.0, -75.0)

    assert stats is not None
    assert stats.valid_points == stats.finite_points + stats.no_return_points
    assert stats.no_return_points == 5
    assert stats.finite_points == 26
    assert stats.p10 == pytest.approx(0.4)
    assert stats.p90 == pytest.approx(scan.range_max)


def test_parking_verification_requires_translation_and_right_turn():
    start = (0.0, 0.0, 0.0)

    assert parking_motion_verified(start, (0.2, 0.0, -0.5), 0.08, 20.0)
    assert not parking_motion_verified(start, (0.2, 0.0, 0.5), 0.08, 20.0)
    assert not parking_motion_verified(start, (0.02, 0.0, -0.5), 0.08, 20.0)


def test_parking_phases_use_odometry_distance_and_right_yaw():
    origin = (0.0, 0.0, 0.0)

    assert not parking_phase_reached('forward', origin, (0.14, 0.0, 0.0), 0.15)
    assert parking_phase_reached('forward', origin, (0.15, 0.0, 0.0), 0.15)
    assert not parking_phase_reached('turn', origin, (0.0, 0.0, 1.2), 75.0)
    assert parking_phase_reached('turn', origin, (0.0, 0.0, -1.4), 75.0)
    assert not parking_phase_reached('enter', None, origin, 0.2)


# ---- 정지선 기준 실측 로그 재현 (구간: 시작 m, 끝 m, P30 m) ----
from jetracer_ros2.parking_logic import BaySlotClassifier

OPEN, BLOCKED = 4.0, 0.55

LOGS = {
    # 2번 칸 빈칸 (1차): 0.430 막힘, 0.695 열림, 1.337 막힘
    'slot2_run1': (2, [(0.0, 0.430, OPEN), (0.430, 0.695, BLOCKED),
                       (0.695, 1.337, OPEN), (1.337, 1.378, BLOCKED)]),
    # 2번 칸 빈칸 (2차): 0.472 / 0.735 / 1.364 / 1.687
    'slot2_run2': (2, [(0.0, 0.472, OPEN), (0.472, 0.735, BLOCKED),
                       (0.735, 1.364, OPEN), (1.364, 1.687, BLOCKED),
                       (1.687, 1.925, OPEN)]),
    # 3번 칸 빈칸: 0.412 막힘, 차1-차2 사이 짧은 틈 3개, 1.159 열림
    'slot3': (3, [(0.0, 0.412, OPEN), (0.412, 0.421, BLOCKED), (0.421, 0.431, OPEN),
                  (0.431, 0.628, BLOCKED), (0.628, 0.638, OPEN),
                  (0.638, 0.842, BLOCKED), (0.842, 0.902, OPEN),
                  (0.902, 1.159, BLOCKED), (1.159, 1.919, OPEN)]),
    # 1번 칸 빈칸: 0.925 까지 열림
    'slot1': (1, [(0.0, 0.925, OPEN), (0.925, 1.189, BLOCKED)]),
}


def _replay(segments, step=0.01):
    classifier = BaySlotClassifier()
    for start, end, p30 in segments:
        d = start
        while d < end:
            result = classifier.update(d, p30)
            if result is not None:
                return result, d
            d += step
    return classifier.result, None


@pytest.mark.parametrize('name', sorted(LOGS))
def test_measured_logs_pick_the_empty_slot(name):
    expected, segments = LOGS[name]
    result, decided_at = _replay(segments)
    assert result == expected
    assert decided_at is not None and decided_at < 1.5


def test_no_opening_after_cars_gives_up():
    result, _ = _replay([(0.0, 0.43, OPEN), (0.43, 2.1, BLOCKED)])
    assert result == 0


@pytest.mark.parametrize('name, low, high', [
    ('slot1', 0.74, 0.76), ('slot2_run1', 0.98, 1.01),
    ('slot2_run2', 1.03, 1.05), ('slot3', 1.45, 1.47)])
def test_confirm_distance_matches_measurement(name, low, high):
    _, decided_at = _replay(LOGS[name][1])
    assert low <= decided_at <= high


def test_short_gap_between_cars_is_not_a_bay():
    # 0.25 m 틈 (< 0.30 m) 은 빈칸이 아님 -> 뒤의 진짜 빈칸(3번)으로 판정
    result, _ = _replay([(0.0, 0.45, OPEN), (0.45, 0.80, BLOCKED), (0.80, 1.05, OPEN),
                         (1.05, 1.16, BLOCKED), (1.16, 1.9, OPEN)])
    assert result == 3


@pytest.mark.parametrize('name', sorted(LOGS))
def test_bay_with_limo_parked_behind_still_counts_as_empty(name):
    # 빈칸 뒤에 LIMO 가 서 있어 P30 이 0.85~0.95 m 로만 나오고,
    # 경계에서 차 쪽 잠깐 값(0.85 m, 0.30 m 미만 구간)이 있어도 같은 칸으로 판정
    expected, segments = LOGS[name]
    behind = [(a, b, 0.95 if v == OPEN else v) for a, b, v in segments]
    assert _replay(behind)[0] == expected
    edge = [(a, b, 0.85 if v == OPEN else v) for a, b, v in segments]
    assert _replay(edge)[0] == expected
