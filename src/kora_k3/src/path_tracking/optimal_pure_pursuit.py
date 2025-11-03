#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import rospy
from std_msgs.msg import Float64

# ---------------- 사용자 파라미터(ROS param으로 덮어쓰기 가능) ----------------
WAYPOINT_FILE = "waypoints-raceline.npy"                 # raceline 파일(.npy 권장)
LOOK_AHEAD_POINTS = 5                                    # 속도 룩어헤드 포인트 수
MIN_SPEED = 1.3                                          # m/s
MAX_SPEED = 4.0                                          # m/s
WHEELBASE = 0.325                                         # m (차량 휠베이스)
WHEEL_RADIUS = 0.05                                       # m
STEER_LIMIT_DEG = 60.0                                   # 서보 한계
PUBLISH_RATE = 20.0                                      # Hz

MOTOR_TOPIC = "/commands/motor/speed"                    # 모터 토픽 (1000단위, 1000=25rpm)
STEER_TOPIC = "/commands/servo/position"                 # 서보 토픽 (0.0=+60°, 1.0=−60°)
# ---------------------------------------------------------------------------

def circle_indexes(n, i, a1=0, a2=0):
    return i, (i + a1) % n, (i + a2) % n

def circle_radius(coords):
    # coords: [[x1,y1],[x2,y2],[x3,y3]]
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]
    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    b = (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1)
    c = (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2)
    d = (x1**2+y1**2)*(x3*y2-x2*y3) + (x2**2+y2**2)*(x1*y3-x3*y1) + (x3**2+y3**2)*(x2*y1-x1*y2)
    eps = 1e-12
    if abs(a) < eps:
        return float("inf")
    val = (b**2 + c**2 - 4*a*d) / abs(4*a**2)
    if val <= 0:
        return float("inf")
    return math.sqrt(val)

def is_left_curve(coords):
    # coords: [[x1,y1],[x2,y2],[x3,y3]]
    x1, y1, x2, y2, x3, y3 = [i for sub in coords for i in sub]
    return ((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)) > 0

def optimal_velocity(track, min_speed, max_speed, look_ahead_points):
    n = len(track)
    # 반경
    radius = []
    for i in range(n):
        idx = circle_indexes(n, i, a1=-1, a2=1)
        coords = [track[idx[0]], track[idx[1]], track[idx[2]]]
        r = circle_radius(coords)
        radius.append(r)
    # 스케일 보정: v ∝ sqrt(r), r_min에서 v=min_speed
    r_min = min([r for r in radius if np.isfinite(r)] + [1e-6])
    const = min_speed / math.sqrt(r_min)

    if look_ahead_points <= 0:
        v = [min(const * math.sqrt(r) if np.isfinite(r) else max_speed, max_speed) for r in radius]
        return v
    else:
        v = []
        L = look_ahead_points
        for i in range(n):
            cand = []
            for j in range(L+1):
                rj = radius[(i + j) % n]
                cand.append(rj)
            r_look = min([rr for rr in cand if np.isfinite(rr)] + [r_min])
            v_i = min(const * math.sqrt(r_look), max_speed)
            v.append(v_i)
        return v

def signed_radius(track):
    n = len(track)
    R_signed = []
    for i in range(n):
        idx = circle_indexes(n, i, a1=-1, a2=1)
        r = circle_radius([track[idx[0]], track[idx[1]], track[idx[2]]])
        left = is_left_curve([track[idx[1]], track[idx[0]], track[idx[2]]])
        if not np.isfinite(r) or r <= 0:
            R_signed.append(float("inf"))
        else:
            R_signed.append(+r if left else -r)
    return R_signed

def steering_from_radius(R, L, steer_limit_deg=60.0):
    # 자전거 모델: δ = atan(L/R) (좌 + / 우 -)
    steering_deg = []
    for Ri in R:
        if not np.isfinite(Ri) or abs(Ri) < 1e-9:
            delta = math.copysign(math.pi/2, Ri if Ri and Ri != 0 else 1.0)
        else:
            delta = math.atan(L / Ri)
        deg = math.degrees(delta)
        deg = max(-steer_limit_deg, min(steer_limit_deg, deg))
        steering_deg.append(deg)
    return steering_deg

def servo_pos_from_deg(delta_deg):
    # 0.0 → +60°(좌), 1.0 → −60°(우)
    # p = (60 − δ)/120
    p = (60.0 - delta_deg) / 120.0
    return max(0.0, min(1.0, p))

def erpm_cmd_from_speed(v_mps, wheel_radius):
    # RPM = v * 60 / (2πr)
    # 명령: 1000 단위 스텝, 1000당 25 rpm → cmd = round( (RPM/25)*1000 )
    if v_mps <= 0:
        return 0.0
    rpm = v_mps * 60.0 / (2.0 * math.pi * wheel_radius)
    cmd = (rpm / 25.0) * 1000.0
    # 1000 스텝으로 반올림
    cmd = round(cmd / 1000.0) * 1000.0
    return float(cmd)

def main():
    rospy.init_node("raceline_controller")

    # 파라미터 가져오기 (없으면 상수 사용)
    waypoint_file = rospy.get_param("~waypoint_file", WAYPOINT_FILE)
    look_ahead_points = int(rospy.get_param("~look_ahead_points", LOOK_AHEAD_POINTS))
    min_speed = float(rospy.get_param("~min_speed", MIN_SPEED))
    max_speed = float(rospy.get_param("~max_speed", MAX_SPEED))
    wheelbase = float(rospy.get_param("~wheelbase", WHEELBASE))
    wheel_radius = float(rospy.get_param("~wheel_radius", WHEEL_RADIUS))
    steer_limit_deg = float(rospy.get_param("~steer_limit_deg", STEER_LIMIT_DEG))
    publish_rate = float(rospy.get_param("~publish_rate", PUBLISH_RATE))
    motor_topic = rospy.get_param("~motor_topic", MOTOR_TOPIC)
    steer_topic = rospy.get_param("~steer_topic", STEER_TOPIC)

    # 퍼블리셔
    motor_pub = rospy.Publisher(motor_topic, Float64, queue_size=10)
    steer_pub = rospy.Publisher(steer_topic, Float64, queue_size=10)

    # 레이싱 라인 로드(폐곡선 가정)
    raceline = np.load(waypoint_file).astype(float)
    # 마지막이 첫 점과 동일하면 한 번 제거해 순환 인덱싱에 사용
    if len(raceline) >= 2 and np.allclose(raceline[0], raceline[-1]):
        track = raceline[:-1]
    else:
        track = raceline
    n = len(track)
    rospy.loginfo("Loaded raceline: %d points from %s", n, waypoint_file)

    # 속도/조향 프로파일
    velocity = optimal_velocity(track, min_speed, max_speed, look_ahead_points)  # m/s
    R_signed = signed_radius(track)
    steering_deg = steering_from_radius(R_signed, wheelbase, steer_limit_deg)

    rate = rospy.Rate(publish_rate)
    i = 0
    while not rospy.is_shutdown():
        v_mps = velocity[i]
        delta_deg = steering_deg[i]

        motor_cmd = erpm_cmd_from_speed(v_mps, wheel_radius)  # 1000단위 스텝
        servo_pos = servo_pos_from_deg(delta_deg)             # 0.0~1.0

        motor_pub.publish(Float64(motor_cmd))
        steer_pub.publish(Float64(servo_pos))

        i = (i + 1) % n
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
