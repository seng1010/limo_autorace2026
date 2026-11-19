"""M2 partial-tape cues must never replace independent safe cone approval."""
import cv2
import numpy as np
import pytest

from jetracer_ros2.m8_entry import M8_LINE_CONFIRM_FRAMES, M8PartialEntryGate
from jetracer_ros2.m8_observers import detect_m8_entry_candidates, detect_m8_stop_line


def tape(center=(550., 365.), shape=(280., 20.), angle=12., yellow=True):
    im=np.zeros((480,640,3),np.uint8)
    pts=cv2.boxPoints((center,shape,angle)).astype(np.int32)
    cv2.fillConvexPoly(im,pts,(185,185,185))
    if yellow:
        # A longitudinal yellow lane meets one end of the transverse tape.
        mid=(pts[np.argsort(pts[:,0])[:2]].mean(axis=0)).astype(int)
        cv2.line(im,tuple(mid+[-12,50]),tuple(mid+[12,-50]),(0,200,230),6)
    return im


@pytest.mark.parametrize('mirror',[False,True])
def test_partial_tape_at_either_side_is_not_promoted_to_full_detection(mirror):
    im=tape()
    if mirror: im=cv2.flip(im,1)
    full,partial=detect_m8_entry_candidates(im,.58)
    assert full is None
    assert detect_m8_stop_line(im,.58) is None
    assert partial is not None
    assert .65 <= partial.row_ratio <= .98


def test_bottom_clipped_tape_uses_visible_center_not_clipped_far_edge():
    im=tape(center=(330.,435.),shape=(225.,47.),angle=-23.)
    full,partial=detect_m8_entry_candidates(im,.58)
    assert full is None
    assert partial is not None
    assert .90 < partial.row_ratio <= .98


@pytest.mark.parametrize('kwargs',[
    {'yellow':False}, {'angle':40.}, {'shape':(280.,75.)},
    {'center':(620.,350.),'shape':(120.,15.)},
    {'center':(550.,290.)}, {'center':(550.,460.)},
])
def test_partial_rejects_mat_edge_steep_thick_short_and_out_of_view(kwargs):
    assert detect_m8_entry_candidates(tape(**kwargs),.58)[1] is None


def test_yellow_line_alone_and_blank_never_make_entry_cue():
    im=np.zeros((480,640,3),np.uint8)
    cv2.line(im,(400,330),(639,390),(0,200,230),16)
    assert detect_m8_entry_candidates(im,.58)==(None,None)


def test_full_detection_remains_the_only_source_when_available():
    im=np.zeros((480,640,3),np.uint8)
    cv2.rectangle(im,(60,370),(580,382),(200,200,200),-1)
    full,partial=detect_m8_entry_candidates(im,.58)
    assert full==detect_m8_stop_line(im,.58)
    assert full is not None and partial is None


def confirmed_gate():
    g=M8PartialEntryGate()
    n=M8_LINE_CONFIRM_FRAMES
    for i in range(n):
        assert g.observe(i*.01,.45-i*.01,10+i*.04)==(i==n-1)
    return g


def test_partial_requires_unique_consistent_frames():
    g=M8PartialEntryGate()
    for _ in range(4):g.observe(0.,.4,10.)
    assert g.anchor(0.,10.1) is None
    # Start after the 0.25 s window so the earlier single frame cannot pair up.
    for i,d in enumerate([.4,-1.,.4,.2,.4]):g.observe(0.,d,10.4+i*.04)
    assert g.anchor(0.,10.6) is None


@pytest.mark.parametrize('s,now',[(.03,10.9),(.221,10.1),(.0,10.1),(.03,float('nan'))])
def test_partial_cue_expires_by_time_distance_or_reverse(s,now):
    assert confirmed_gate().anchor(s,now) is None


def test_recent_cue_survives_one_missed_image_but_cannot_renew_itself():
    g=confirmed_gate()
    g.observe(.03,-1.,10.12)
    assert g.anchor(.04,10.2)==pytest.approx(.45)
    assert g.anchor(.04,10.9) is None
