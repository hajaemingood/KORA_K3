#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from collections import deque
from sensor_msgs.msg import Imu
import tf
import tf.transformations as tft

def cutoff_to_tau(fc_hz):
    fc_hz = max(1e-6, float(fc_hz))
    return 1.0 / (2.0 * np.pi * fc_hz)

class EMAFilter:
    def __init__(self, dim=3, fc_hz=10.0):
        self.y = None
        self.tau = cutoff_to_tau(fc_hz)
        self.dim = dim

    def set_cutoff(self, fc_hz):
        self.tau = cutoff_to_tau(fc_hz)

    def reset(self):
        self.y = None

    def step(self, x, dt):
        x = np.asarray(x, dtype=float).reshape(self.dim,)
        if self.y is None:
            self.y = x.copy()
            return self.y
        dt = max(1e-6, float(dt))
        alpha = dt / (self.tau + dt)
        self.y = self.y + alpha * (x - self.y)
        return self.y

class MedianFilter:
    def __init__(self, dim=3, window=3):
        self.dim = dim
        self.window = max(1, int(window))
        self.buf = [deque(maxlen=self.window) for _ in range(dim)]

    def step(self, x):
        x = np.asarray(x, dtype=float).reshape(self.dim,)
        for i in range(self.dim):
            self.buf[i].append(x[i])
        if any(len(b) == 0 for b in self.buf):
            return x
        med = np.array([np.median(b) for b in self.buf])
        return med

class ImuPreprocessor(object):
    def __init__(self):
        # topics and frames
        self.input_topic   = rospy.get_param("~input_topic", "/imu/data")
        self.output_topic  = rospy.get_param("~output_topic", "/imu/data_centered")
        self.base_frame    = rospy.get_param("~base_link_frame", "base_link")
        self.imu_frame     = rospy.get_param("~imu_frame", "imu_link")
        self.use_tf        = rospy.get_param("~use_tf", True)
        self.remove_grav   = rospy.get_param("~remove_gravity", True)
        r_param            = rospy.get_param("~r_base", [-0.014, 0.0, -0.05])  # [m] base에서 IMU까지
        self.r_base        = np.array(r_param, dtype=float).reshape(3,)

        # filtering params
        fc_g = float(rospy.get_param("~lowpass_fc_gyro_hz", 15.0))
        fc_a = float(rospy.get_param("~lowpass_fc_accel_hz", 15.0))
        self.use_median    = bool(rospy.get_param("~use_median", True))
        win_g              = int(rospy.get_param("~median_window_gyro", 3))
        win_a              = int(rospy.get_param("~median_window_accel", 3))

        self.med_gyro = MedianFilter(3, win_g) if self.use_median and win_g >= 3 else None
        self.med_acc  = MedianFilter(3, win_a) if self.use_median and win_a >= 3 else None
        self.lp_gyro  = EMAFilter(3, fc_g)
        self.lp_acc   = EMAFilter(3, fc_a)

        # state for alpha computation
        self.prev_w_base = None
        self.prev_t = None

        # TF
        self.tf_listener = tf.TransformListener()

        # pub sub
        self.pub = rospy.Publisher(self.output_topic, Imu, queue_size=50)
        self.sub = rospy.Subscriber(self.input_topic, Imu, self.cb, queue_size=100)

        rospy.loginfo("imu_preprocessor: in=%s out=%s base=%s imu=%s use_tf=%s remove_g=%s r=%s fc_g=%.2f fc_a=%.2f median=%s",
                      self.input_topic, self.output_topic, self.base_frame, self.imu_frame,
                      str(self.use_tf), str(self.remove_grav), str(self.r_base.tolist()),
                      fc_g, fc_a, str(self.use_median))

    @staticmethod
    def quat_to_mat(qx, qy, qz, qw):
        return tft.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

    def get_R_imu_to_base(self, stamp):
        # imu_frame → base_link
        try:
            # try exact time then latest
            self.tf_listener.waitForTransform(self.base_frame, self.imu_frame, stamp, rospy.Duration(0.03))
            trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.imu_frame, stamp)
        except Exception:
            try:
                self.tf_listener.waitForTransform(self.base_frame, self.imu_frame, rospy.Time(0), rospy.Duration(0.03))
                trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.imu_frame, rospy.Time(0))
            except Exception as e2:
                rospy.logwarn_throttle(2.0, "TF lookup failed (%s→%s): %s. Using identity.", self.imu_frame, self.base_frame, str(e2))
                return np.eye(3), np.zeros(3)
        R = tft.quaternion_matrix(rot)[:3, :3]
        return R, np.array(trans).reshape(3,)

    def cb(self, msg):
        # time and dt
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        dt = None if self.prev_t is None else max(1e-6, t - self.prev_t)

        # raw imu signals
        w_imu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        a_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=float)

        # pre-filter spikes with median
        if self.med_gyro is not None:
            w_imu = self.med_gyro.step(w_imu)
        if self.med_acc is not None:
            a_imu = self.med_acc.step(a_imu)

        # low-pass filtering (causal EMA)
        if dt is None:
            # first sample: just initialize filters
            self.lp_gyro.reset()
            self.lp_acc.reset()
            w_imu_f = self.lp_gyro.step(w_imu, 1e-3)
            a_imu_f = self.lp_acc.step(a_imu, 1e-3)
        else:
            w_imu_f = self.lp_gyro.step(w_imu, dt)
            a_imu_f = self.lp_acc.step(a_imu, dt)

        # frame transform imu→base
        if self.use_tf:
            R_ib, t_ib = self.get_R_imu_to_base(msg.header.stamp)
            w_base = R_ib.dot(w_imu_f)
            a_base = R_ib.dot(a_imu_f)
        else:
            R_ib = np.eye(3)
            w_base = w_imu_f
            a_base = a_imu_f

        # angular acceleration alpha from filtered omega
        if self.prev_w_base is None or dt is None or dt < 1e-6:
            alpha_base = np.zeros(3)
        else:
            alpha_base = (w_base - self.prev_w_base) / dt

        # rigid-body correction from IMU point to CoM
        r = self.r_base
        a_C_base = a_base - np.cross(alpha_base, r) - np.cross(w_base, np.cross(w_base, r))

        # gravity removal using Rb2w = Ri2w * Rib^T
        if self.remove_grav:
            q = msg.orientation
            R_i2w = self.quat_to_mat(q.x, q.y, q.z, q.w)
            R_b2w = R_i2w.dot(R_ib.T) if self.use_tf else R_i2w
            g_world = np.array([0.0, 0.0, 9.81])
            a_C_world = R_b2w.dot(a_C_base)
            a_C_world_no_g = a_C_world - g_world
            a_C_base = R_b2w.T.dot(a_C_world_no_g)

        # build output message
        out = Imu()
        out.header = msg.header
        out.header.frame_id = self.base_frame

        # orientation: 그대로 내보내되 frame 의미는 주의 필요
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance

        out.angular_velocity.x = w_base[0]
        out.angular_velocity.y = w_base[1]
        out.angular_velocity.z = w_base[2]
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        out.linear_acceleration.x = a_C_base[0]
        out.linear_acceleration.y = a_C_base[1]
        out.linear_acceleration.z = a_C_base[2]
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)

        # update state
        self.prev_w_base = w_base
        self.prev_t = t

def main():
    rospy.init_node("imu_preprocessor")
    ImuPreprocessor()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
