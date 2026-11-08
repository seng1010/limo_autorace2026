"""lane_pipeline_ks (곡선 보완 버전) 테스트: 기존 테스트 전부 + 곡선 테스트 2개."""
import cv2
import numpy as np

from jetracer_ros2.lane_pipeline_ks import LanePipeline, LanePipelineConfig


def _synthetic_lane(offset=0, missing_right=False):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    curve = 0.00035 * (y - 240.0) ** 2
    left = 140.0 + offset + curve
    right = left + 360.0
    left_points = np.int32(np.column_stack((left, y))).reshape(-1, 1, 2)
    cv2.polylines(image, [left_points], False, (255, 255, 255), 10)
    if not missing_right:
        right_points = np.int32(np.column_stack((right, y))).reshape(-1, 1, 2)
        cv2.polylines(image, [right_points], False, (255, 255, 255), 10)
    return image


def _synthetic_zigzag():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    normalized_y = (y - 240.0) / 240.0
    center = 320.0 + 180.0 * (normalized_y ** 3 - 0.7 * normalized_y)
    left = center - 180.0
    right = center + 180.0
    left_points = np.int32(np.column_stack((left, y))).reshape(-1, 1, 2)
    right_points = np.int32(np.column_stack((right, y))).reshape(-1, 1, 2)
    cv2.polylines(image, [left_points], False, (255, 255, 255), 10)
    cv2.polylines(image, [right_points], False, (255, 255, 255), 10)
    return image


def _identity_config():
    return LanePipelineConfig(
        src_points=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        dst_points=(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
        white_lightness_min=150,
        gradient_min=20,
        lane_width_px=360.0,
        fit_min_points=80,
    )


def test_detects_two_curved_lanes():
    result = LanePipeline(_identity_config()).process(_synthetic_lane())
    assert result.valid
    assert result.confidence > 0.5
    assert abs(result.lateral_error) < 0.25


def test_recovers_one_missing_lane():
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid
    result = pipeline.process(_synthetic_lane(missing_right=True))
    assert result.valid
    assert result.reason == 'one-side recovery'


def test_rejects_empty_scene():
    result = LanePipeline(_identity_config()).process(
        np.zeros((480, 640, 3), dtype=np.uint8),
    )
    assert not result.valid
    assert result.confidence == 0.0


def test_detects_zigzag_for_slow_driving():
    result = LanePipeline(_identity_config()).process(_synthetic_zigzag())
    assert result.valid
    assert result.zigzag
    assert result.direction_changes >= 1
    assert result.zigzag_score > 0.5


def test_reset_recovers_after_lane_shift_while_lost():
    """차선을 놓친 사이 차가 밀리면 급변 검사에 계속 걸린다. reset()으로 복구돼야 한다."""
    pipeline = LanePipeline(_identity_config())
    assert pipeline.process(_synthetic_lane()).valid
    for _ in range(5):
        pipeline.process(np.zeros((480, 640, 3), dtype=np.uint8))
    shifted = _synthetic_lane(offset=-110)
    assert not pipeline.process(shifted).valid
    pipeline.reset()
    assert pipeline.process(shifted).valid


def test_detects_yellow_lanes_when_enabled():
    """노란 차선은 흰색 검출만으로는 못 찾고, detect_yellow를 켜면 찾아야 한다."""
    image = _synthetic_lane()
    yellow = image.copy()
    yellow[np.all(image == 255, axis=2)] = (0, 220, 255)   # BGR 노란색

    white_only = _identity_config()
    white_only.gradient_min = 255     # 경계선 검출 영향 제거
    assert not LanePipeline(white_only).process(yellow).valid

    yellow_config = _identity_config()
    yellow_config.gradient_min = 255
    yellow_config.detect_white = False
    yellow_config.detect_yellow = True
    result = LanePipeline(yellow_config).process(yellow)
    assert result.valid
    assert abs(result.lateral_error) < 0.25


# ---------------------------------------------------------------------------
# 곡선 구간: 안쪽 선이 화면 밖으로 나가고 바깥쪽 선 하나만 보이는 경우
# (실차에서 오른쪽 커브 중 'lane geometry sanity check failed'로 멈추던 상황)
# ---------------------------------------------------------------------------

def _parallel_right_curve(t, lane=360.0):
    """중심선에서 수직 방향으로 lane/2 씩 떨어진 실제 평행 차선 (t가 클수록 급커브)."""
    y = np.linspace(-200, 680, 2000)
    s = np.clip((479.0 - y) / 479.0, 0, None)
    cx = 320.0 + t * 520.0 * s ** 1.7
    dx = np.gradient(cx, y)
    n = np.sqrt(1 + dx ** 2)
    left = np.column_stack((cx - 0.5 * lane / n, y + 0.5 * lane * dx / n))
    right = np.column_stack((cx + 0.5 * lane / n, y - 0.5 * lane * dx / n))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for points in (left, right):
        cv2.polylines(image, [np.int32(points).reshape(-1, 1, 2)], False, (255, 255, 255), 14)
    return image, lambda yy: float(np.interp(yy, y, cx))


def test_keeps_tracking_through_sharp_right_curve():
    """직선 -> 급커브로 들어가는 동안 한 번도 멈추지 않고, 바깥(왼쪽) 선 기준으로 따라가야 한다."""
    pipeline = LanePipeline(_identity_config())
    y_eval = 480 * pipeline.config.lookahead_y_ratio
    for t in np.round(np.arange(0.0, 1.85, 0.15), 2):
        image, center = _parallel_right_curve(t)
        for _ in range(8):
            result = pipeline.process(image)
        assert result.valid, f'curve t={t}: {result.reason}'
        estimated = 0.5 * (np.polyval(result.left_fit, y_eval) + np.polyval(result.right_fit, y_eval))
        assert abs(estimated - center(y_eval)) < 45.0, f'curve t={t}: center off'
    assert result.reason == 'one-side recovery'
    assert result.recovered_side == 'left'


def test_single_line_grabbed_by_both_windows_is_treated_as_one_line():
    """선 하나가 화면을 대각선으로 가로지르면 왼쪽/오른쪽 탐색이 같은 선을 나눠 잡는다.
    같은 선으로 합쳐서 왼쪽 선으로 인식하고, 중심은 그 선의 오른쪽에 있어야 한다."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    y = np.arange(480, dtype=np.float32)
    x = 90.0 + 470.0 * ((479.0 - y) / 479.0) ** 1.6
    cv2.polylines(image, [np.int32(np.column_stack((x, y))).reshape(-1, 1, 2)],
                  False, (255, 255, 255), 14)
    result = LanePipeline(_identity_config()).process(image)
    assert result.valid, result.reason
    assert result.recovered_side == 'left'
    y_eval = 480 * 0.62
    line_x = float(np.polyval(result.left_fit, y_eval))
    center_x = 0.5 * (line_x + float(np.polyval(result.right_fit, y_eval)))
    assert center_x > line_x + 100.0
