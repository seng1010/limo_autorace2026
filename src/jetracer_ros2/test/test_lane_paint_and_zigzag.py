import math
import cv2
import numpy as np
from jetracer_ros2 import simple_lane as S


def test_broad_paint_is_not_a_lane_or_fragment():
    e=S.LaneEngine();im=np.zeros((230,180),np.uint8)
    cv2.rectangle(im,(72,40),(103,140),255,-1)
    assert e._components(im)==[] and e.frags==[]


def test_thin_straight_and_sharp_arc_survive_thickness_filter():
    e=S.LaneEngine();im=np.zeros((230,180),np.uint8)
    cv2.line(im,(20,220),(20,60),255,5)
    cv2.ellipse(im,(35,130),(100,95),0,270,360,255,5)
    assert len(e._components(im))==2


def test_straight_and_single_bend_are_not_zigzag():
    f=np.linspace(.4,1.4,50)
    assert not S._zigzag_shape(.1*f,f)
    assert not S._zigzag_shape(.3*(f-.7)**2,f)


def test_zigzag_trend_center_has_no_reversed_segments():
    e=S.LaneEngine();f=np.linspace(.4,1.4,800)
    x=-.215+.05*np.sin((f-.4)*2*np.pi/.2)
    # Make a dense narrow ribbon so every radial bin has enough support.
    x=np.tile(x,5)+np.repeat(np.linspace(-.008,.008,5),800);f=np.tile(f,5)
    comp=dict(xm=x,fm=f,px=np.zeros(len(x)),py=np.zeros(len(x)),near_x=-.215)
    line=e._make_line(comp,'left')
    assert line.zigzag
    cx,cf=S._offset(line.xs,line.fs,1)
    assert np.all(np.diff(cf)>-1e-5)
    assert np.max(np.abs(cx))<.035


def test_zigzag_observation_cancels_corner_lock_not_center_tracking():
    e=S.LaneEngine();e.locked=True;e.lock_dir=-1;e.mode='CURVE'
    r=dict(valid=True,raw_angle=.05,ff_angle=-.4,ff_w=1.,zigzag=True,
           source='left_only',tilt=.8,cross=('left',.7),wall=.7)
    v,angle,state,_=e.command(r,S.DEFAULT_TUNE,1.)
    assert not e.locked and v>0 and angle>0 and state=='TRACK'


def test_confirmed_straight_can_release_lock_before_yaw_budget():
    e=S.LaneEngine();e.locked=True;e.lock_dir=-1;e.mode='STRAIGHT';e.lock_t=1.
    e.yaw=e.lock_yaw=0.
    r=dict(valid=True,raw_angle=.02,ff_angle=0.,ff_w=0.,source='both',tilt=.01,cross=None,wall=None)
    for t in (1.1,1.2):
        e.command(r,S.DEFAULT_TUNE,t)
        assert e.locked
    e.command(r,S.DEFAULT_TUNE,1.3)
    assert not e.locked


def test_single_straight_frame_or_visible_outer_wall_cannot_unlock_corner():
    e=S.LaneEngine();e.locked=True;e.lock_dir=-1;e.mode='CURVE';e.lock_t=1.;e.curve_dir=-1
    e.yaw=e.lock_yaw=0.
    r=dict(valid=True,raw_angle=-.15,ff_angle=0.,ff_w=0.,source='left_only',tilt=.5,
           cross=('left',.7),wall=.7)
    for t in np.arange(1.1,1.6,.1):e.command(r,S.DEFAULT_TUNE,t)
    assert e.locked


def test_tracking_history_cannot_authorize_disagreeing_center_paths(monkeypatch):
    e=S.LaneEngine();e.tracked={'left':True,'right':True}
    def line(comp,side):
        f=np.linspace(.45,1.4,40);x=np.full(40,-.45 if side=='left' else .45)
        out=S.LaneLine(side,x,f,np.arange(40),np.arange(40),x[0]);out.zigzag=False
        return out
    monkeypatch.setattr(e,'_pick_lines',lambda c:{'left':(.01,{}),'right':(.2,{})})
    monkeypatch.setattr(e,'_make_line',line)
    r=e.process(np.zeros((480,640,3),np.uint8),S.DEFAULT_TUNE,1.)
    assert r['valid'] and r['source']=='left_only'
    assert r['right'] is None and '불일치' in r['note']
