"""Run actual node methods with fake ROS messages; no vehicle commands are sent."""
import importlib
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jetracer_ros2.cone_corridor import EntrySearchWindow, Params, Planner, ResumeGate
from jetracer_ros2.m8_exit_memory import ConeExitMemory
from test_m8_control_adapter import Stamped, Twist, Pub


@pytest.fixture
def node_module(monkeypatch):
    for name, attrs in {
        'ament_index_python.packages':dict(get_package_share_directory=lambda _:''),
        'rclpy':{}, 'rclpy.node':dict(Node=object),
        'rclpy.qos':dict(DurabilityPolicy=NS(TRANSIENT_LOCAL=1),
                         ReliabilityPolicy=NS(BEST_EFFORT=1),QoSProfile=lambda **kw:kw),
        'geometry_msgs.msg':dict(PointStamped=Stamped,PolygonStamped=Stamped,Twist=Twist,TwistStamped=Stamped),
        'nav_msgs.msg':dict(Odometry=Stamped), 'sensor_msgs.msg':dict(LaserScan=Stamped),
        'std_msgs.msg':dict(Bool=NS,Float32=NS,Header=NS,String=NS),
    }.items():
        mod=types.ModuleType(name);mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,mod)
    name='jetracer_ros2.mission_cone_avoid_node'
    sys.modules.pop(name,None)
    mod=importlib.import_module(name)
    yield mod
    sys.modules.pop(name,None)


def make_node(mod):
    n=mod.MissionConeAvoidNode.__new__(mod.MissionConeAvoidNode)
    n.config=Params();n.planner=Planner(n.config);n.exit_memory=ConeExitMemory(n.config)
    n.entry_search=EntrySearchWindow();n.entry_search.set_phase('cone')
    n.exit_resume=ResumeGate(2);n.exit_resume.update(True);n.exit_resume.update(True)
    n.motion_enabled=True;n.exit_check=True;n.m8_phase='cone'
    n.recovery_pub=Pub();n.recovery_sent=False;n.exit_cmd_pub=Pub();n.exit_progress_pub=Pub()
    n.approach_risk_pub=Pub()
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=10_000_000_000,to_msg=lambda:NS(sec=10,nanosec=0)))
    lane=Stamped();lane.header.stamp=NS(sec=10,nanosec=0)
    lane.twist.linear.x=.2;lane.twist.angular.z=.1
    n.exit_lane=lane;n.lane_hint=lane
    n.exit_evidence=Stamped()
    n.exit_evidence.header.stamp=NS(sec=10,nanosec=0)
    n.exit_evidence.point.x=1.
    n.exit_evidence.point.y=1.
    n.lane_center=(10_000_000_000,[(.3+i*.01,.1) for i in range(20)])
    return n


def test_exit_preparation_preserves_cone_history_and_lane_geometry(node_module):
    n=make_node(node_module)
    n.planner.memory={'sentinel':1};n.exit_memory.gates=[{'sentinel':2}]
    n.entry_search.started=8.;n.entry_search.travel=.3
    n._change_phase(NS(data='exit_prepare'))
    assert n.planner.memory=={'sentinel':1} and n.exit_memory.gates==[{'sentinel':2}]
    assert n.lane_center and n.lane_hint and n.exit_resume.count==2
    assert n.entry_search.started==8. and n.entry_search.travel==.3
    n._change_phase(NS(data='exit_lane'))
    assert n.planner.memory and n.lane_center and n.exit_resume.count==2
    n._change_phase(NS(data='off'))
    assert n.planner.memory is None and not n.exit_memory.gates and n.lane_center is None


def test_prepare_evaluates_same_checked_lane_and_connector(node_module,monkeypatch):
    n=make_node(node_module);n.m8_phase='exit_prepare'
    calls=[]
    monkeypatch.setattr(node_module,'checked_exit_lane',lambda *a,**kw: calls.append(kw) or (0.,0.))
    monkeypatch.setattr(node_module,'safe_lane_arc',lambda *a: (.1,.05))
    result=dict(points=np.ones((4,2)),scan=((),np.array([])))
    n._publish_exit(result,NS(sec=10,nanosec=0))
    assert calls[0]['approach'] is False  # Keep existing exit horizon, not the shorter M2 check.
    msg=n.exit_cmd_pub.messages[-1]
    assert (msg.twist.linear.x,msg.twist.angular.z)==(.1,.05)
    n.exit_lane.header.stamp.sec=9
    n._publish_exit(result,NS(sec=10,nanosec=0))
    assert n.exit_cmd_pub.messages[-1].twist.linear.x==0.


def test_progress_is_stamped_evidence_not_a_command(node_module):
    n=make_node(node_module)
    n._publish_exit_progress(dict(exit_progress=dict(cleared=True,gates=2,rear_clearance=.04)),
                             NS(sec=9,nanosec=950_000_000))
    msg=n.exit_progress_pub.messages[-1]
    assert msg.header.frame_id=='rear_axle' and msg.header.stamp.sec==9
    assert (msg.point.x,msg.point.y,msg.point.z)==(1.,2.,.04)
    assert not n.exit_cmd_pub.messages
    n._publish_exit_progress()
    assert n.exit_progress_pub.messages[-1].point.x==0.


@pytest.mark.parametrize('stamp,valid',[(9,1.),(10,0.)])
def test_no_exit_command_or_recovery_from_unverified_geometry(node_module,monkeypatch,stamp,valid):
    n=make_node(node_module);n.m8_phase='exit_lane'
    n.exit_evidence.header.stamp.sec=stamp;n.exit_evidence.point.x=valid
    monkeypatch.setattr(node_module,'checked_exit_lane',lambda *a,**kw: pytest.fail('unverified command'))
    monkeypatch.setattr(node_module,'safe_lane_arc',lambda *a: pytest.fail('unverified path'))
    monkeypatch.setattr(node_module,'lane_forward_recovery',lambda *a,**kw: pytest.fail('unverified recovery'))
    offer=n._publish_exit(dict(points=np.ones((4,2)),scan=((),np.array([]))),NS(sec=10,nanosec=0))
    assert offer is None and n.exit_cmd_pub.messages[-1].twist.linear.x==0.


def test_unaligned_ordinary_command_uses_corrected_connector_only(node_module,monkeypatch):
    n=make_node(node_module);n.m8_phase='exit_prepare';n.exit_evidence.point.y=0.
    # In normal transport the observer already zeros the wrong command;
    # independently enforce its alignment evidence at the receiving node.
    n.exit_lane.twist.linear.x=0.
    n.lane_hint=Stamped();n.lane_hint.header.stamp=NS(sec=10,nanosec=0)
    n.lane_hint.twist.linear.x=.1
    monkeypatch.setattr(node_module,'checked_exit_lane',lambda *a,**kw:pytest.fail('unaligned lane'))
    monkeypatch.setattr(node_module,'safe_lane_arc',lambda *a:(.1,-.03))
    n._publish_exit(dict(points=np.ones((4,2)),scan=((),np.array([]))),NS(sec=10,nanosec=0))
    assert n.exit_cmd_pub.messages[-1].twist.angular.z==-.03


def test_aligned_live_command_still_uses_unchanged_margin_and_speed(node_module):
    n=make_node(node_module);n.m8_phase='exit_lane';n.exit_lane.twist.angular.z=0.
    assert n.config.cone_padding==.015 and n.config.clearance==.020
    angles=np.linspace(-1.9,1.9,250)
    scan=((-1.9,1.9,angles[1]-angles[0],0.,.1,.01,10.),np.full(250,3.))
    # .034 m measured side clearance is < .015 padding + .020 clearance.
    points=np.array([[.2,n.config.half_width+.034],[1.,1.],[1.,-1.],[2.,0.]])
    n.lane_center=None
    n._publish_exit(dict(points=points,scan=scan),NS(sec=10,nanosec=0))
    assert n.exit_cmd_pub.messages[-1].twist.linear.x==0.
    points[0,1]=n.config.half_width+.036
    for _ in range(2):
        n._publish_exit(dict(points=points,scan=scan),NS(sec=10,nanosec=0))
    assert n.exit_cmd_pub.messages[-1].twist.linear.x==.1


def test_observer_checker_manager_handoff_completes_only_after_odom_passage(node_module,monkeypatch):
    from test_m8_control_adapter import Owner
    from test_m8_exit_lane import scene
    from jetracer_ros2.m8_control import M8Control
    from jetracer_ros2.m8_observers import M8LaneObserver
    class Polygon(Stamped):
        def __init__(self):super().__init__();self.polygon=NS(points=[])
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'PolygonStamped',Polygon)
    monkeypatch.setattr(sys.modules['geometry_msgs.msg'],'Point32',NS,raising=False)
    owner=Owner();owner.m8=M8Control(owner);owner.m8.enter('M8')
    owner._odom.s=1.;owner.m8._m8_exit.reset(0.)
    ob=M8LaneObserver(owner,.184);n=make_node(node_module)
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=owner.now,to_msg=lambda:NS(
        sec=owner.now//10**9,nanosec=owner.now%10**9)))
    image,r,g=scene(right=.26,ordinary_correct=True);r['raw_angle']=0.
    angles=np.linspace(-1.9,1.9,250)
    scan=((-1.9,1.9,angles[1]-angles[0],0.,.1,.01,10.),np.full(250,3.))
    points=np.array([[.2,.5],[.2,-.5],[1.,1.],[1.,-1.]])
    for i in range(95):
        owner.now=10_000_000_000+i*100_000_000;owner._odom_ns=owner.now
        # Observe >4s while stopped, then integrate physical forward motion.
        travelled=max(0,i-50)*.02;owner._odom.s=1.+travelled
        stamp=NS(sec=owner.now//10**9,nanosec=owner.now%10**9)
        if i<3:
            line=Stamped();line.header.stamp=stamp;line.header.frame_id='front_axle';line.point.x=.3
            owner.m8._m8_exit_line_callback(line)
        ob._phase(NS(data=owner.m8._m8_phase_sent))
        ob.update(stamp,r,(.2,0.,'TRACK'),image=image,pose=(travelled,0.,0.),geometry=g)
        if i==4:image[:]=0
        owner.m8._store_exit_evidence(ob.evidence.messages[-1])
        n.m8_phase=owner.m8._m8_phase_sent;n.exit_lane=ob.command.messages[-1]
        n.lane_hint=ob.hint.messages[-1];n.exit_evidence=ob.evidence.messages[-1]
        n.lane_center=(owner.now,np.array([(p.x,p.y) for p in ob.path.messages[-1].polygon.points]))
        n._publish_exit(dict(points=points,scan=scan),stamp)
        owner.m8._store_exit_lane(n.exit_cmd_pub.messages[-1]);owner.m8.tick(owner.now)
        if owner._active=='M1':
            assert owner._odom.s>=1.60  # exit line 1.30 + unchanged .30 pass margin
            break
        if i>=5:assert owner.output[-1]==(.1,0.)
        if i<=50:assert owner._active=='M8'
    assert owner._active=='M1'
