import math

from jetracer_ros2 import imu_heading as H
from jetracer_ros2 import three_lane_simple as S


def test_bias_learned_when_stationary_and_integrated_when_moving():
    imu = H.ImuYaw()
    t = 0.0
    for _ in range(200):                     # 서 있음: 자이로 영점 0.02 rad/s
        imu.add(t, 0.02, t, stationary=True)
        t += 0.01
    assert abs(imu.bias - 0.02) < 1e-3 and abs(imu.yaw) < 1e-9
    for _ in range(100):                     # 1초 동안 0.5 rad/s 로 돎 (+영점)
        imu.add(t, 0.52, t, stationary=False)
        t += 0.01
    assert abs(imu.yaw - 0.5) < 0.02
    assert imu.fresh(t) and not imu.fresh(t + 1.0)


def test_sign_check():
    imu = H.ImuYaw()
    assert imu.compare(0.1, -0.1) is None
    for _ in range(4):
        r = imu.compare(-0.1, 0.1)
    assert r == -1


def test_lane3_heading_uses_imu_and_distance_from_odom():
    # odom 은 30도 돌았다고 하지만 IMU 는 10도: 차로 방향 th 는 IMU 를 따른다. 거리는 odom 변위 크기 그대로
    k = S.SimpleThreeLane()
    k.start(0.0, 0.0, (0.0, 0.0, 0.0, 0.0), 1.0)
    oyaw = math.radians(30)
    k._advance((0.3 * math.cos(oyaw / 2), 0.3 * math.sin(oyaw / 2), math.radians(10), oyaw))
    assert abs(k.th + math.radians(10)) < 1e-9
    assert abs(k.travel - 0.3) < 1e-6
    # 후진: odom 방향 기준으로 뒤로 간 변위 -> 거리 음수
    k._advance((0.3 * math.cos(oyaw / 2) - 0.1 * math.cos(oyaw), 0.3 * math.sin(oyaw / 2) - 0.1 * math.sin(oyaw),
                math.radians(10), oyaw))
    assert abs(k.travel - 0.2) < 1e-6
