#!/usr/bin/env python3
"""rosbag 으로 정지선 보정표(lut_rows, lut_distances_m) 만들기 — 자 없이 odom 을 자로 쓴다.

보정표는 **앞바퀴 기준**으로 만든다 (앞바퀴가 선에 닿는 순간이 눈으로 보기 쉬움).
reference_wheel: rear 로 쓰면 stop_line_node 가 wheelbase_m 을 더해서 낸다.

녹화 방법 (차에서)
  1. 정지선이 화면에 보이도록 약 60 cm 앞에 차를 둔다 (stop_line_node 는 꺼도 됨)
  2. ros2 bag record -o stopline_01 /camera/color/image_raw/compressed /odom
  3. 0.05 m/s 정도로 천천히 직진 -> 앞바퀴가 정지선 가까운 쪽 가장자리에 닿으면 정지
  4. 정지한 채로 1~2초 더 녹화하고 Ctrl+C. 3~5번 반복 (stopline_02, ...)
  - 앞바퀴가 선에 못 미쳐 섰으면 --end-offset-m 로 남은 거리를 넣는다 (지나쳤으면 음수)
    자가 없으면 A4 짧은 변(21 cm)을 4번 접은 띠 ≈ 5.25 cm 로 어림

실행
  python3 scripts/calibrate_stop_line.py stopline_01 stopline_02 stopline_03
  python3 scripts/calibrate_stop_line.py stopline_* --csv samples.csv --points 8

bag 읽기: ROS2 가 설치된 차에서는 rosbag2_py, 노트북(mac 등)에서는 `pip install rosbags`.
검출 설정은 params/params.yaml 의 stop_line_node 값을 그대로 쓴다 (정지선 색 등).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from jetracer_ros2.stop_line_detector import (  # noqa: E402
    OdomPathIntegrator, StopLineConfig, detect_stop_line, fit_row_distance_table)


def load_config(params_path):
    with open(params_path) as handle:
        params = yaml.safe_load(handle)['stop_line_node']['ros__parameters']
    fields = StopLineConfig.__dataclass_fields__
    return StopLineConfig(**{k: v for k, v in params.items() if k in fields})


def read_bag(path, image_topic, odom_topic):
    """(t_ns, topic, msg) 를 시간순으로. msg 는 .data / .pose / .twist 로 접근."""
    topics = {image_topic, odom_topic}
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        rosbag2_py = None
    if rosbag2_py is not None:
        with open(os.path.join(path, 'metadata.yaml')) as handle:
            storage = yaml.safe_load(handle)['rosbag2_bagfile_information']['storage_identifier']
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
        out = []
        while reader.has_next():
            topic, data, t_ns = reader.read_next()
            if topic in topics:
                out.append((t_ns, topic, deserialize_message(data, types[topic])))
        return sorted(out, key=lambda item: item[0])
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        sys.exit('bag 을 읽을 수 없습니다: ROS2 환경을 source 하거나 `pip install rosbags`')
    out = []
    with AnyReader([Path(path)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, t_ns, raw in reader.messages(connections=connections):
            out.append((t_ns, connection.topic, reader.deserialize(raw, connection.msgtype)))
    return sorted(out, key=lambda item: item[0])


def samples_from_bag(path, config, image_topic, odom_topic, end_offset_m):
    messages = read_bag(path, image_topic, odom_topic)
    odom = OdomPathIntegrator()
    odom_t, odom_s = [], []
    for t_ns, topic, msg in messages:
        if topic == odom_topic:
            pose = msg.pose.pose.position
            odom_t.append(t_ns)
            odom_s.append(odom.update(pose.x, pose.y, msg.twist.twist.linear.x))
    if len(odom_t) < 2:
        raise RuntimeError(f'{path}: {odom_topic} 메시지가 없습니다')
    s_end = odom_s[-1]                       # 마지막 = 앞바퀴가 정지선에 닿은 위치
    rows = []
    for t_ns, topic, msg in messages:
        if topic != image_topic:
            continue
        image = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        detection = detect_stop_line(image, config)
        if detection is None:
            continue
        s = float(np.interp(t_ns, odom_t, odom_s))
        rows.append((detection.row_ratio, s_end - s + end_offset_m))
    travelled = odom_s[-1] - odom_s[0]
    return rows, travelled


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bags', nargs='+')
    parser.add_argument('--params', default=str(HERE.parent / 'params' / 'params.yaml'))
    parser.add_argument('--image-topic', default='/camera/color/image_raw/compressed')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--end-offset-m', type=float, default=0.0,
                        help='정지한 앞바퀴 -> 정지선 남은 거리 (지나쳤으면 음수)')
    parser.add_argument('--points', type=int, default=8)
    parser.add_argument('--csv', help='모든 샘플(row_ratio, distance_m, bag) 저장')
    args = parser.parse_args()

    config = load_config(args.params)
    all_samples, rows_for_csv, blind = [], [], []
    for bag in args.bags:
        samples, travelled = samples_from_bag(
            bag, config, args.image_topic, args.odom_topic, args.end_offset_m)
        if samples:
            nearest = min(d for _, d in samples)
            blind.append(nearest)
            print(f'{bag}: 이동 {travelled:.3f} m, 검출 {len(samples)}프레임, '
                  f'마지막으로 보인 거리 {nearest:.3f} m')
        else:
            print(f'{bag}: 정지선 검출 0프레임 (색·ROI 설정 확인)')
        all_samples += samples
        rows_for_csv += [(r, d, bag) for r, d in samples]

    if args.csv:
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_ratio', 'distance_m', 'bag'])
            writer.writerows(rows_for_csv)
        print(f'샘플 저장: {args.csv}')

    lut_rows, lut_dists = fit_row_distance_table(all_samples, args.points)
    if blind:
        print(f'\n카메라 사각지대(앞바퀴 기준): 약 {np.mean(blind):.3f} m '
              f'(이 안쪽은 odom 추적. rear 기준이면 이 값 + wheelbase_m 이 '
              f'blind_track_max_m 보다 작아야 함)')
    print('\n# params.yaml stop_line_node 에 붙여넣기')
    print('    lut_rows: [' + ', '.join(f'{r:.4f}' for r in lut_rows) + ']')
    print('    lut_distances_m: [' + ', '.join(f'{d:.4f}' for d in lut_dists) + ']')


if __name__ == '__main__':
    main()
