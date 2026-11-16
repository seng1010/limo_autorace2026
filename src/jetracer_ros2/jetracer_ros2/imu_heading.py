"""IMU 자이로로 차 방향(yaw)을 잰다 (ROS 없음, simple_lane 이 사용).

아커만 드라이버의 /odom 방향은 조향 명령으로 계산하는 듯해서 크게 꺾거나 후진할 때 실제보다 많이 돈 것으로 나온다
(11/16 실차 1116_210431: odom 은 49도, 카메라는 도로와 19도). 3차선(M7) 위치 추정은 거리만 odom 으로 재고,
방향은 이 값을 쓴다.

- /imu 의 angular_velocity.z 를 적분한다 (orientation 은 장비마다 비어 있거나 자력계 영향이 있어 쓰지 않는다).
- 자이로 영점(bias): 차가 서 있을 때(stationary=True) 천천히 맞춘다. 처음 BIAS_INIT_N 개는 평균으로 빨리 맞춘다.
- 부호: ROS 규약(위가 +z, 왼쪽으로 돌면 +)이면 odom yaw 와 같은 방향. 반대로 달린 IMU 면 SIGN = -1.
"""
import math

BIAS_INIT_N = 100        # 처음 이만큼(서 있을 때)은 단순 평균으로 영점
BIAS_ALPHA = 0.01        # 그 뒤 서 있을 때 영점 따라가는 비율
MAX_DT = 0.1             # 메시지 사이가 이보다 길면 적분하지 않는다 (끊김)
STALE_SEC = 0.3          # 이 시간 넘게 IMU 가 안 오면 쓰지 않는다


class ImuYaw:
    def __init__(self, sign=1.0):
        self.sign = sign
        self.yaw = 0.0
        self.bias = 0.0
        self.bias_n = 0
        self.last_stamp = None
        self.last_rx = None          # 받은 시각 (monotonic)
        self.check_sum = 0.0         # 부호 확인: sum(imu dyaw * odom dyaw)
        self.check_abs = 0.0         # 그동안 odom 이 돈 양

    def add(self, stamp, wz, rx, stationary):
        """stamp: 메시지 시각 [s], wz: angular_velocity.z [rad/s], rx: 받은 시각, stationary: 지금 차가 서 있나."""
        if stationary:
            if self.bias_n < BIAS_INIT_N:
                self.bias_n += 1
                self.bias += (wz - self.bias) / self.bias_n
            else:
                self.bias += BIAS_ALPHA * (wz - self.bias)
        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            if 0.0 < dt <= MAX_DT and not stationary:
                self.yaw += self.sign * (wz - self.bias) * dt
        self.last_stamp = stamp
        self.last_rx = rx

    def fresh(self, now):
        return self.last_rx is not None and now - self.last_rx <= STALE_SEC and self.bias_n >= 10

    def compare(self, d_imu, d_odom):
        """부호 확인용: 같은 구간에서 IMU 와 odom 이 돈 양. 반환: 확인된 부호(+1/-1) 또는 None."""
        if abs(d_odom) < math.radians(0.5):
            return None
        self.check_sum += d_imu * d_odom
        self.check_abs += abs(d_odom)
        if self.check_abs < math.radians(20.0):          # odom 기준 20도 넘게 돈 뒤에 판단
            return None
        return 1 if self.check_sum > 0 else -1
