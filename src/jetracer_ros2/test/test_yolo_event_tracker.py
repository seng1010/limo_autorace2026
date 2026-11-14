from jetracer_ros2.yolo_event_tracker import StableEventTracker


def test_confirms_after_n_frames_once():
    t = StableEventTracker(3)
    assert t.update(['pedestrian']) == []
    assert t.update(['pedestrian']) == []
    assert t.update(['pedestrian']) == ['pedestrian']
    assert t.update(['pedestrian']) == []          # 매 프레임 반복 발행하지 않음


def test_two_classes_in_one_frame_both_confirm():
    """예전 버그: 신뢰도 1등만 세어서 한쪽이 묻혔다."""
    t = StableEventTracker(3)
    for _ in range(2):
        assert t.update(['pedestrian', 'traffic_red']) == []
    assert t.update(['pedestrian', 'traffic_red']) == ['pedestrian', 'traffic_red']


def test_alternating_classes_do_not_reset_each_other():
    """예전 버그: 번갈아 1등이 되면 카운트가 리셋돼 둘 다 안 나갔다."""
    t = StableEventTracker(2)
    assert t.update(['sign_a', 'slow_sign']) == []
    assert t.update(['slow_sign', 'sign_a']) == ['sign_a', 'slow_sign']


def test_disappearing_resets_and_can_fire_again():
    t = StableEventTracker(2)
    t.update(['pedestrian'])
    assert t.update([]) == []                       # 사라짐 -> 카운트 초기화
    assert t.update(['pedestrian']) == []
    assert t.update(['pedestrian']) == ['pedestrian']


def test_ignores_empty_labels():
    t = StableEventTracker(1)
    assert t.update(['', None, 'pedestrian']) == ['pedestrian']


def test_confirm_frames_one_fires_immediately():
    assert StableEventTracker(1).update(['slow_sign']) == ['slow_sign']
