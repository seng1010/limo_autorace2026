import math

# limo_base(limo_driver.cpp)의 상수와 반드시 동일해야 한다.
#
# limo_base는 Ackermann 모드에서 /cmd_vel의 angular.z를 조향각으로 그대로 쓰지
# 않는다. 대신 r = linear.x / angular.z 로 회전반경을 구한 뒤, 차량 기하학
# (wheelbase, track)으로 실제 앞바퀴 조향각을 역산하는 방식이다.
# 그래서 우리가 원하는 "조향각"을 데이터/모델에서 얻었다면, 그걸 그대로
# angular.z에 넣으면 안 되고, 이 공식을 반대로 계산해서 원하는 조향각이
# 나오게 만드는 angular.z 값을 만들어서 보내야 한다. 이 파일이 그 역산을 담당한다.
WHEELBASE = 0.2   # 앞뒤 바퀴 거리 [m]
TRACK = 0.172     # 좌우 바퀴 거리 [m]
# limo_base가 하드웨어로 보내기 전에 이 값으로 최종 클램프한다 (약 28도).
# [11/14 실차 측정] limo_base.launch.py 의 /cmd_vel 경로는 이 값으로 자른 뒤 2.47 로 나눠 보내서, 차는 최대 11.3도
# (회전반경 약 1.05m) 밖에 못 꺾었다. race 브랜치는 limo_ackerman_base 드라이버 + 변환 노드(simple_lane bridge)를
# 쓰며, 그때 실제 최대 조향은 안쪽 바퀴 기준 약 25도 (회전반경 약 0.51m) 이다. 이 값보다 큰 조향 명령은 25도로 잘린다.
HARDWARE_MAX_INNER_ANGLE = 0.48869


def inner_angle_to_omega(inner_angle: float, linear_speed: float) -> float:
    """원하는 앞바퀴 조향각(inner_angle, rad)을, limo_base의
    r = linear.x / angular.z 공식을 통과했을 때 그 조향각이 그대로
    나오게 만드는 angular.z(omega) 값으로 변환한다.

    linear_speed가 0이면(정지 상태) 이 공식 자체가 성립하지 않으므로 0을 반환한다
    (이 경우 limo_base는 무조건 최대 조향각으로 튀는 특수 동작을 한다).
    """
    if abs(inner_angle) < 1e-4 or linear_speed == 0.0:
        return 0.0
    r = WHEELBASE / math.tan(abs(inner_angle)) + TRACK / 2.0
    central_angle = math.atan(WHEELBASE / r)
    if inner_angle < 0:
        central_angle = -central_angle
    return linear_speed * math.tan(central_angle) / WHEELBASE
