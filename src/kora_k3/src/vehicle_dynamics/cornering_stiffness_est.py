#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ramp_steer_logger.py
# Usage:
#   rosrun your_pkg ramp_steer_logger.py _csv_path:=/tmp/rs_trial_01.csv _motor_data_sp:=2000
#   (다른 파라미터는 아래 set_param 참고)
import os
import rospy
import numpy as np
import csv
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from time import time as walltime

class RampSteerLogger(object):
    def __init__(self):
        # ---------- 파일 저장 ----------
        self.csv_path = rospy.get_param("~csv_path", "cornering_stiffness/ramp_steer_trial.csv")

        # ---------- 실험 구간(초) ----------
        self.total_time  = rospy.get_param("~total_time", 20.0)
        self.t_warm      = rospy.get_param("~t_warm", 3.0)
        self.t_ramp_up   = rospy.get_param("~t_ramp_up", 4.0)
        self.t_hold_up   = rospy.get_param("~t_hold_up", 2.0)
        self.t_ramp_down = rospy.get_param("~t_ramp_down", 4.0)
        self.t_hold_down = rospy.get_param("~t_hold_down", 2.0)
        self.t_return    = rospy.get_param("~t_return", 3.0)

        # ---------- 조향 맵핑 ----------
        # u in [0,1]  -> delta[rad]  (선형: δ_deg = steer_span_deg*(u - steer_center))
        self.steer_center     = rospy.get_param("~steer_center", 0.5)
        self.steer_span_deg   = rospy.get_param("~steer_span_deg", 60.0)  # 0~1이 ±30°면 60
        self.delta_max_deg    = rospy.get_param("~delta_max_deg", 4.0)    # 램프 최대각(선형 영역)
        self.steer_topic      = rospy.get_param("~steer_topic", "/commands/servo/position")

        # ---------- 속도 변환 ----------
        # rpm = rpm_per_data * motor_cmd,  Vx = rpm*(2π/60)*R_eff*gear_ratio
        self.motor_topic   = rospy.get_param("~motor_topic", "/commands/motor/speed")
        self.motor_data_sp = rospy.get_param("~motor_data_sp", 2000.0)   # 실험 속도 세트포인트(고정)
        self.rpm_per_data  = rospy.get_param("~rpm_per_data", 0.025)     # 너 실험값(1000 -> 25 rpm)
        self.R_eff         = rospy.get_param("~wheel_R_eff", 0.05)      # 유효반경[m]
        self.gear_ratio    = rospy.get_param("~gear_ratio", 1.0)

        # ---------- 차량 상수(이미 확보했다고 했음) ----------
        self.m   = rospy.get_param("~mass", 3.8)
        self.L   = rospy.get_param("~wheelbase", 0.26)
        self.lf  = rospy.get_param("~lf", 0.145)
        self.lr  = rospy.get_param("~lr", self.L - self.lf)
        self.Iz  = rospy.get_param("~Iz", 0.015)

        # ---------- IMU ----------
        # 권장: /imu/data_centered (중력 제거/프레임 정렬된 것)
        self.imu_topic = rospy.get_param("~imu_topic", "/imu/data_centered")
        self.use_centered = True

        # ---------- ROS pub/sub ----------
        self.pub_motor = rospy.Publisher(self.motor_topic, Float64, queue_size=10)
        self.pub_steer = rospy.Publisher(self.steer_topic, Float64, queue_size=10)
        rospy.Subscriber(self.imu_topic, Imu, self.cb_imu)

        # ---------- 버퍼 ----------
        self._started = False
        self.t0 = None
        self.rows = []  # each: [time, u_cmd, delta_rad, motor_cmd, Vx, ay, r]
        self.bias_rz = None  # 시작 직후 정지구간에서 추정
        self.last_imu_stamp = None

        rospy.loginfo("RampSteerLogger ready. Will save to %s", self.csv_path)

    def u_from_delta(self, delta_rad):
        # δ_deg = span*(u - center)
        deg = np.degrees(delta_rad)
        u = deg / self.steer_span_deg + self.steer_center
        return float(np.clip(u, 0.0, 1.0))

    def delta_from_u(self, u):
        deg = self.steer_span_deg*(u - self.steer_center)
        return np.radians(deg)

    def cb_imu(self, msg):
        # for time reference
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else rospy.get_time()
        self.last_imu_stamp = stamp
        if not self._started:
            # trial 시작시간을 첫 IMU 수신 시점으로
            self.t0 = stamp
            self._started = True
            # gyro bias: 첫 0.5초 평균
            self.bias_rz = msg.angular_velocity.z
        # nothing else here; actual logging in run()

    def run(self):
        rate = rospy.Rate(100)
        # 타임라인 총합 검증
        sched_sum = self.t_warm + self.t_ramp_up + self.t_hold_up + self.t_ramp_down + self.t_hold_down + self.t_return
        if sched_sum > self.total_time + 1e-6:
            rospy.logwarn("Sum of segments > total_time. Adjusting total_time to sum.")
            self.total_time = sched_sum

        rospy.loginfo("Experiment starting. total_time=%.1fs", self.total_time)
        while not rospy.is_shutdown():
            if not self._started:
                # IMU가 들어오기 전 대기하면서 모터는 미리 회전 시작해도 OK(원하면 0으로)
                self.pub_motor.publish(Float64(self.motor_data_sp))
                self.pub_steer.publish(Float64(self.steer_center))
                rate.sleep()
                continue

            now = rospy.get_time()
            tau = now - self.t0
            if tau > self.total_time:
                break

            # 1) 속도 명령 고정
            self.pub_motor.publish(Float64(self.motor_data_sp))

            # 2) 램프 스티어 δ 생성
            delta_max = np.radians(self.delta_max_deg)
            if tau < self.t_warm:
                delta = 0.0
            elif tau < self.t_warm + self.t_ramp_up:
                s = (tau - self.t_warm)/self.t_ramp_up
                delta = s*delta_max
            elif tau < self.t_warm + self.t_ramp_up + self.t_hold_up:
                delta = delta_max
            elif tau < self.t_warm + self.t_ramp_up + self.t_hold_up + self.t_ramp_down:
                s = (tau - (self.t_warm + self.t_ramp_up + self.t_hold_up))/self.t_ramp_down
                delta = delta_max + s*(-2.0*delta_max)
            elif tau < self.t_warm + self.t_ramp_up + self.t_hold_up + self.t_ramp_down + self.t_hold_down:
                delta = -delta_max
            else:
                s = (tau - (self.t_warm + self.t_ramp_up + self.t_hold_up + self.t_ramp_down + self.t_hold_down))/self.t_return
                delta = -delta_max + s*(delta_max)

            # 3) δ → u 퍼블리시
            u = self.u_from_delta(delta)
            self.pub_steer.publish(Float64(u))

            # 4) 현재 Vx 계산 (명령 기반 setpoint로부터)
            rpm = self.rpm_per_data * self.motor_data_sp
            omega = rpm * (2*np.pi/60.0)
            Vx = omega * self.R_eff * self.gear_ratio

            # 5) 최신 IMU 값 읽기 (가능한 한 동일 시각)
            ay, r = np.nan, np.nan
            # 여기선 간단히 /imu/data_centered 가속도 y, yaw rate z를 바로 읽는다
            # 실제 구현에서는 콜백에서 최신 값을 멤버에 저장해두고 가져오는 게 일반적이지만,
            # 본 예제는 간결화를 위해 msg를 스냅샷하지 않고 "동일 주기에서 저장"만 한다.
            # -> 더 정확하게 하려면 콜백에서 self.ay, self.r를 저장해와 사용.
            # 임시 처리: bias_rz만 반영
            # (실전에서는 콜백 변수로 바꿔줘)
            # ---------- 개선 포인트 ----------
            # self.ay, self.r 를 콜백에서 멤버로 업데이트하고 여기서 읽어 쓰세요.
            # ---------------------------------

            # 로그 한 줄 (IMU를 콜백에서 멤버로 보관한다고 가정)
            # 안전하게 NaN 방지
            if hasattr(self, "ay_last") and hasattr(self, "r_last"):
                ay = self.ay_last
                r  = self.r_last

            self.rows.append([tau, u, delta, self.motor_data_sp, Vx, ay, r])

            rate.sleep()

        # 종료: 모터/조향 정지
        self.pub_motor.publish(Float64(0.0))
        self.pub_steer.publish(Float64(self.steer_center))

        # CSV 저장
        self._save_csv()
        rospy.loginfo("Saved CSV: %s (%d rows)", self.csv_path, len(self.rows))

    # IMU 콜백에서 최신 측정 보관(위 run()에서 사용)
    def cb_imu(self, msg):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else rospy.get_time()
        if not self._started:
            self.t0 = stamp
            self._started = True
            self.bias_rz = msg.angular_velocity.z
        # 최신값 저장
        r = msg.angular_velocity.z - (self.bias_rz or 0.0)
        ay = msg.linear_acceleration.y  # /imu/data_centered 가정
        self.r_last = r
        self.ay_last = ay


    def _save_csv(self):
        csv_path = os.path.expanduser(self.csv_path)
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)

        directory = os.path.dirname(csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:  # <-- append 모드
            w = csv.writer(f)

            # 새로 만든 파일이라면 헤더 추가
            if not file_exists or os.path.getsize(csv_path) == 0:
                w.writerow(["trial", "time", "u_cmd", "delta_rad", "motor_cmd", "Vx", "ay", "r"])

            # trial ID (자동 증가를 위해 파일 내 기존 행 개수로 추정하거나 내부 카운터 사용)
            trial_id = getattr(self, "trial_id", int(file_exists))  # 단순히 1 이상
            for row in self.rows:
                w.writerow([trial_id] + row)

        self.csv_path = csv_path
        rospy.loginfo("Appended %d rows to %s", len(self.rows), self.csv_path)

if __name__ == "__main__":
    rospy.init_node("ramp_steer_logger")
    node = RampSteerLogger()
    rospy.sleep(0.5)
    node.run()
