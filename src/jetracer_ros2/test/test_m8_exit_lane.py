"""Road roles are not screen roles. No ROS or vehicle motion required."""
from types import SimpleNamespace as NS

import cv2
import numpy as np

from jetracer_ros2.m8_exit_lane import ExitLaneResolver, contact_role, tape_contours


class Geometry:
    def ground_to_image(self, x, f):
        return np.column_stack((320+400*np.asarray(x), 480-200*np.asarray(f)))


def scene(right=-.10, role='right', ordinary_correct=False, tape=True):
    geo=Geometry();f=np.linspace(.35,1.3,60);r=np.full_like(f,right)
    line=NS(xs=r,fs=f)
    image=np.zeros((480,640,3),np.uint8)
    if tape:
        x,y=geo.ground_to_image([right],[.75])[0].astype(int)
        dx=-170 if role=='right' else 170
        cv2.line(image,(x,y),(x+dx,y),(255,255,255),10)
    center=r+(-.26 if ordinary_correct and role=='right' else .26)
    return image,dict(left=line,right=None,valid=True,center=(center,f),lane_width=.52),geo


def run(resolver, image, result, geo, pose=(0.,0.,0.), start=1.):
    for i in range(5):
        answer=resolver.update(image,result,geo,pose,start+i*.033)
    return answer


def test_right_boundary_left_of_car_is_offset_into_lane_not_outside():
    image,r,g=scene()
    v=run(ExitLaneResolver(),image,r,g)
    assert v['valid'] and v['roles']==['right'] and not v['aligned']
    assert np.all(v['path'][:,1]>.30) and v['angle']>0  # left towards the road
    assert np.all(r['center'][0]>0)  # untouched ordinary path still wrongly points right


def test_mirrored_left_boundary_not_first_seen_right():
    image,r,g=scene(right=.10,role='left',ordinary_correct=True)
    v=run(ExitLaneResolver(),image,r,g)
    assert v['valid'] and v['roles']==['left'] and v['angle']<0
    assert v['aligned']


def test_odom_preserves_role_when_tape_disappears_and_vehicle_turns():
    resolver=ExitLaneResolver();image,r,g=scene();run(resolver,image,r,g)
    pose=(.02,.01,.04)
    # Same WORLD boundary reprojected into the moving rear-axle frame.
    old=np.column_stack((r['left'].fs+.184,-r['left'].xs))
    c,s=np.cos(pose[2]),np.sin(pose[2]);rot=np.array([[c,-s],[s,c]])
    world=old+np.array([-.10,0.])
    new=(world-(np.array(pose[:2])+rot@[-.10,0.]))@rot
    r['left']=NS(fs=new[:,0]-.184,xs=-new[:,1])
    v=resolver.update(np.zeros_like(image),r,g,pose,1.165)
    assert v['valid'] and v['roles']==['right']
    # History is not a path: no current yellow observation => no authority.
    r['left']=None
    assert not resolver.update(image,r,g,pose,1.198)['valid']


def test_missing_odom_jump_reverse_and_repeated_image_cannot_authorize():
    for pose,stamp in [(None,1.2),((1.,0.,0.),1.2),((-.03,0.,0.),1.2),((0.,0.,0.),1.132)]:
        resolver=ExitLaneResolver();image,r,g=scene();run(resolver,image,r,g)
        v=resolver.update(image,r,g,pose,stamp)
        assert not v['valid']


def test_line_without_tape_or_track_has_no_assumed_role():
    image,r,g=scene(tape=False)
    assert not run(ExitLaneResolver(),image,r,g)['valid']


def test_white_tape_crossing_both_sides_is_not_an_endpoint():
    pixels=Geometry().ground_to_image(np.zeros(50),np.linspace(.35,1.3,50))
    contour=np.array([[150,325],[490,325],[490,335],[150,335]],float)
    assert contact_role(pixels,[contour],640) is None


def test_ordinary_agreement_required_and_lost_immediately():
    resolver=ExitLaneResolver();image,r,g=scene(ordinary_correct=True)
    assert run(resolver,image,r,g)['aligned']
    r['center']=(r['center'][0]+.52,r['center'][1])
    v=resolver.update(image,r,g,(0.,0.,0.),1.165)
    assert v['valid'] and not v['aligned']


def test_blue_clipped_white_tape_and_yellow_glare_are_not_swapped():
    image,r,g=scene()
    # Bag white tape has a clipped blue channel: HLS S can approach 255
    # despite this being low-saturation white in HSV.
    image[image[:,:,0]>0]=(255,220,190)
    cv2.line(image,(280,440),(365,220),(75,200,248),9)
    v=run(ExitLaneResolver(),image,r,g)
    assert v['valid'] and v['roles']==['right']
    image[:]=0
    cv2.line(image,(280,440),(365,220),(235,235,235),10)
    assert not tape_contours(image)  # Longitudinal white glare is not a transverse tape.


def test_tape_less_tracking_expires_without_refreshing_its_own_anchor():
    resolver=ExitLaneResolver();image,r,g=scene();run(resolver,image,r,g)
    image[:]=0
    for stamp in np.arange(1.165,5.4,.05):
        v=resolver.update(image,r,g,(0.,0.,0.),float(stamp))
    assert not v['valid'] and not resolver.tracks


def test_one_direct_contact_does_not_confirm_itself_through_prediction():
    resolver=ExitLaneResolver();image,r,g=scene()
    assert not resolver.update(image,r,g,(0.,0.,0.),1.)['valid']
    image[:]=0
    for stamp in (1.033,1.066,1.099):
        assert not resolver.update(image,r,g,(0.,0.,0.),stamp)['valid']
