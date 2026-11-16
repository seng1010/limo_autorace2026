"""Read-only manager regression replay; no ROS, hardware, or physical simulation.

python test/replay_manager_bag.py BAG --out OUTPUT [--baseline 7d4c027]
Requires the test dependencies (pytest, numpy, OpenCV). Uses original message
reception times. Traffic state is extracted from recorded M1 debug when the
dedicated topic was not recorded; after it ends, it expires to unknown. This
must NOT invent a new green or claim that changed commands produced this odom.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import sqlite3
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
from test_mission_manager_transitions import Node, Twist, modules


class Cdr:
    def __init__(self, raw):
        if raw[:2] != b'\x00\x01':
            raise ValueError('Only little-endian ROS2 CDR is supported')
        self.raw, self.i = raw[4:], 0

    def read(self, fmt, n=1):
        size = struct.calcsize('<'+fmt)
        self.i = (self.i+size-1)//size*size
        values = struct.unpack_from('<'+str(n)+fmt, self.raw, self.i)
        self.i += n*size
        return values

    def string(self):
        length, = self.read('I')
        value = self.raw[self.i:self.i+length-1].decode()
        self.i += length
        return value


def parameters(text):
    """Only the flat scalar manager block; no YAML dependency for this tool."""
    block = text.split('mission_manager_node:\n', 1)[1].split('\n\n', 1)[0]
    result = {}
    for line in block.splitlines():
        if not line.startswith('    ') or ':' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split(':', 1)
        value = value.split('#', 1)[0].strip()
        result[key.strip()] = (dict(true=True, false=False)[value] if value in ('true', 'false')
                               else ast.literal_eval(value))
    return result


def load(path):
    topics = ('/odom', '/inference/cmd_vel', '/cmd_vel', '/mission/stop_line_distance',
              '/mission/traffic_light_state', '/mission/active', '/mission_manager/debug/state')
    rows, origin = [], None
    for file in sorted(path.glob('*.db3')):
        with sqlite3.connect(file.resolve().as_uri()+'?mode=ro', uri=True) as db:
            assert db.execute('pragma quick_check').fetchone()[0] == 'ok'
            first = db.execute('select min(timestamp) from messages').fetchone()[0]
            origin = min(origin, first) if origin is not None else first
            for stamp, topic, raw in db.execute(
                    'select timestamp,name,data from messages join topics on topic_id=topics.id '
                    'order by timestamp'):
                if topic not in topics:
                    continue
                c = Cdr(raw)
                if topic == '/odom':
                    c.read('i', 2); c.string(); c.string()
                    pose = c.read('d', 7); c.read('d', 36); velocity = c.read('d', 6)
                    value = (pose[0], pose[1], velocity[0])
                elif topic.endswith('cmd_vel'):
                    v = c.read('d', 6); value = (v[0], v[5])
                elif topic.endswith('distance'):
                    value = c.read('f')[0]
                else:
                    value = c.string()
                rows.append((stamp, topic, value))
    return sorted(rows), origin


def replay(rows, origin, source, overrides):
    with pytest.MonkeyPatch.context() as patch:
        fixture = modules.__wrapped__(patch)
        next(fixture)
        try:
            module = ModuleType('bag_replayed_manager')
            exec(compile(source, 'mission_manager_node.py', 'exec'), module.__dict__)
            module.MissionManagerNode._start_process = lambda self, name: self.started.append(name)
            module.MissionManagerNode._stop_process = lambda *args, **kwargs: None
            Node.overrides = dict(overrides, start_mission='M1', auto_start_mission_nodes=False)
            n = module.MissionManagerNode()
            start = next(t for t, topic, v in rows if topic == '/mission/active' and v == 'M1')
            next_tick = start
            has_light = any(topic == '/mission/traffic_light_state' for _, topic, _ in rows)
            output = []

            def tick(t):
                n.now = t
                n._control_tick()
                cmd = n.cmd_pub.messages[-1]
                output.append(dict(t=(t-origin)/1e9, mission=n._active, state=n._m1.state,
                    lines=n._m1.lines_seen, s=n._odom.s, odom_v=n._odom_speed,
                    light=n._fresh_traffic(t), v=cmd.linear.x, omega=cmd.angular.z,
                    parking_start=n.parking_start_pub.messages[-1].data
                    if n.parking_start_pub.messages else False))

            for stamp, topic, value in rows:
                while next_tick < stamp:
                    tick(next_tick)
                    next_tick += 50_000_000
                n.now = stamp
                if topic == '/odom':
                    x, y, v = value
                    n._odom_callback(NS(pose=NS(pose=NS(position=NS(x=x, y=y))),
                        twist=NS(twist=NS(linear=NS(x=v)))))
                elif topic == '/inference/cmd_vel':
                    cmd = Twist(); cmd.linear.x, cmd.angular.z = value
                    n._store_cmd('lane', cmd)
                elif topic == '/mission/stop_line_distance':
                    n._stop_line_callback(NS(data=value))
                elif topic == '/mission/traffic_light_state':
                    n._traffic_callback(NS(data=value))
                elif (not has_light and topic == '/mission_manager/debug/state'
                      and value.startswith('M1 ')):
                    match = re.search(r'light=(\w+)', value)
                    if match:
                        n._traffic_callback(NS(data=match[1]))
            return output
        finally:
            fixture.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', default='7d4c027')
    args = parser.parse_args()
    rows, origin = load(args.bag)
    args.out.mkdir(parents=True, exist_ok=True)
    source_path = 'src/jetracer_ros2/jetracer_ros2/mission_manager_node.py'
    param_path = 'src/jetracer_ros2/params/params.yaml'
    summaries = {}
    versions = {}
    for name in ('before', 'after'):
        def read(path):
            return (subprocess.check_output(['git', 'show', args.baseline+':'+path], cwd=REPO).decode()
                    if name == 'before' else (REPO/path).read_text())
        output = replay(rows, origin, read(source_path), parameters(read(param_path)))
        versions[name] = output
        with (args.out/(name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader(); writer.writerows(output)
        transitions, last = [], None
        for row in output:
            key = row['mission'], row['state'], row['lines']
            if key != last:
                transitions.append(row)
                last = key
        summaries[name] = transitions
    # Timing assertions are specific to this regression recording, not course distances.
    checks = 'not_applied: different bag/baseline; inspect CSV and transitions'
    if args.bag.name == 'bag_full_20261116_214231' and args.baseline == '7d4c027':
        before, after = versions['before'], versions['after']
        assert any(r['mission'] == 'M4' and r['t'] < 54 for r in before)
        assert not any(r['mission'] == 'M4' for r in after)
        first = [r for r in after if r['lines'] == 1]
        assert first and all(r['state'] == 'approach' and 0 <= r['v'] <= .15 for r in first)
        second = [r for r in after if r['lines'] == 2]
        assert second and any(r['state'] == 'stopping' and r['v'] == 0 for r in second)
        assert not any(r['parking_start'] for r in after)
        checks = 'passed'
    summary = dict(bag=str(args.bag), baseline=args.baseline, checks=checks,
        limitation='Open-loop: recorded odom does not respond to new commands. No recorded light '
        'state after original M1 ended; never synthesize green. Physical stopping/completion '
        'and process warm-up timing require vehicle confirmation.', transitions=summaries)
    (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
