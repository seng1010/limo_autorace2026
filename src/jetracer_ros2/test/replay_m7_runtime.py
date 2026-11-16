"""Causal ROS-free replay of the SAME M7Worker used by the vehicle.

Requires optional `rosbags`, numpy and opencv-python. Recorded manual poses are
fixed: this audits perception/timing/proposed commands, not autonomous success.
Keep real bags and generated traces outside the repository.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import cv2

from jetracer_ros2 import simple_lane as legacy
from jetracer_ros2.m7_runtime import LatestInputs, M7Worker, load_config, jsonable, SafetyGate


def replay(bag, output, tune_path=None, config_path=None, camera_hz=None):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    nav,options = load_config(config_path)
    cv2.setNumThreads(1)
    if camera_hz is not None:
        options['camera_hz'] = camera_hz
    tune = json.loads(Path(tune_path).read_text()) if tune_path else legacy.load_tune()
    inputs = LatestInputs()
    worker = M7Worker(inputs,nav,options,tune,legacy,True,None)
    gate = SafetyGate(nav,bool(options.get('deskew')))
    store = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    counts = Counter()
    selected_odom = None
    def tick(timestamp, start):
        ros_now = timestamp*1e-9
        now = (timestamp-start)*1e-9
        before = time.perf_counter()
        data = inputs.snapshot()
        try:
            result = worker.tick(data,ros_now,now)
            reason = gate.check(result,data,ros_now,now)
        except (ValueError,RuntimeError) as exc:
            result = dict(state='INPUT_ERROR',reason=str(exc),v=0.,steer=0.,started=now,active=True)
            reason = str(exc)
        row = {k:v for k,v in result.items() if k!='image'}
        row.update(t=now,wall_ms=(time.perf_counter()-before)*1000,gate_reason=reason,
                   proposed_after_gate=0. if reason else result['v'])
        rows.append(row)
    with Reader(Path(bag)) as reader:
        start = reader.start_time
        tick_ns = int(1e9/options['control_hz'])
        next_tick = start
        topics = {legacy.IMAGE_TOPIC,legacy.CAMERA_INFO_TOPIC,'/odom','/limo/odom',legacy.SCAN_TOPIC}
        connections = [c for c in reader.connections if c.topic in topics]
        last = start
        for connection,timestamp,raw in reader.messages(connections=connections):
            while next_tick<timestamp:
                tick(next_tick,start)
                next_tick += tick_ns
            message = store.deserialize_cdr(raw,connection.msgtype)
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            received = (timestamp-start)*1e-9
            topic = connection.topic
            counts[topic] += 1
            if topic in ('/odom','/limo/odom'):
                selected_odom = selected_odom or topic
                if topic!=selected_odom:
                    continue
                p,q = message.pose.pose.position,message.pose.pose.orientation
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                inputs.odom((p.x,p.y,yaw),stamp,received)
            elif topic==legacy.IMAGE_TOPIC:
                inputs.put('image',bytes(message.data),stamp,received,message.header.frame_id)
            elif topic==legacy.CAMERA_INFO_TOPIC:
                inputs.put('camera',dict(k=message.k,d=message.d,width=message.width,height=message.height),
                           stamp,received,message.header.frame_id)
            elif topic==legacy.SCAN_TOPIC:
                inputs.put('scan',dict(ranges=message.ranges,angle_min=message.angle_min,
                    angle_increment=message.angle_increment,range_min=message.range_min,range_max=message.range_max,
                    time_increment=message.time_increment,scan_time=message.scan_time),stamp,received,message.header.frame_id)
            last = timestamp
        while next_tick<=last:
            tick(next_tick,start)
            next_tick += tick_ns
    path = Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    trace = path.with_suffix('.traces.jsonl')
    with trace.open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(jsonable(row),ensure_ascii=False)+'\n')
    times = [row['wall_ms'] for row in rows]
    report = dict(method='fixed manual poses; causal receipt order; acquisition timestamps; no actuation',
                  bag=Path(bag).name,control_hz=options['control_hz'],camera_hz=options['camera_hz'],
                  counts=dict(counts),control_ticks=len(rows),processed_images=worker.frame,
                  states=dict(Counter(row['state'] for row in rows)),plan_count=worker.sup.core.planner.plan_count,
                  wall_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),p99=float(np.percentile(times,99))),
                  trace_file=trace.name,autonomous_success_rate=None,hardware_cpu_percent=None)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('--output',required=True)
    parser.add_argument('--tune')
    parser.add_argument('--config')
    parser.add_argument('--camera-hz',type=float)
    args=parser.parse_args()
    print(json.dumps(replay(args.bag,args.output,args.tune,args.config,args.camera_hz),indent=2))
