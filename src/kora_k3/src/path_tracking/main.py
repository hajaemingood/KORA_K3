#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
import sys

# Ensure local packages resolve when launched via the catkin relay script.
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import rospy 
from math import *
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float64
from obstacle.follow_the_gap import Follow_the_gap
from path_tracking.lookahead import Pure_pursuit
import numpy as np

class Controller:
    def __init__(self):
        rospy.init_node("main_controller")
        self.motor_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        self.follow_gap = Follow_the_gap()
        self.pure_pursuit = Pure_pursuit()
        self.gap_threshold = 0.4 # lidar detection range (m)

        self.scan_msg = None
        self.pose_msg = None

        rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber("/base_link_pose", Pose2D, self.pose_callback, queue_size=1)

        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

    def scan_callback(self, scan_msg):
        self.scan_msg = scan_msg

    def pose_callback(self, pose_msg):
        self.pose_msg = pose_msg

    def pub_motor(self, speed, steer):
        speed_msg = Float64()
        steer_msg = Float64()
        speed_msg.data = speed
        steer_msg.data = steer
        self.motor_pub.publish(speed_msg)
        self.servo_pub.publish(steer_msg)

    def compute_min_distance(self, scan_msg):
        # if scan_msg is None:
        #     return float('inf')
        # valid_ranges = [r for r in scan_msg.ranges if not isinf(r) and not isnan(r)]
        # if not valid_ranges:
        #     return float('inf')
        # return min(valid_ranges)
        angle_ranges, dist_ranges = self.follow_gap.preprocess_lidar(scan_msg)
        valid_ranges = dist_ranges[np.isfinite(dist_ranges)]
        if valid_ranges.size:
            return float(valid_ranges.min()), angle_ranges, dist_ranges
        else:
            return float('inf'), angle_ranges, dist_ranges
        

    def control_loop(self, _event):
        min_distance,angle_ranges, dist_ranges = self.compute_min_distance(self.scan_msg)
        print(f"min distance: {min_distance:.2f} m")
        if min_distance <= self.gap_threshold:
            theta = self.follow_gap.lidar_CB(angle_ranges, dist_ranges, self.gap_threshold)
            steer, speed = self.follow_gap.gap_control(theta)
            print("follow the gap")
        else:
            steer, speed = self.pure_pursuit.compute_control(self.pose_msg)
            print("pure pursuit")

        self.pub_motor(speed, steer)

def main():
    try:
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass    
  
if __name__ == '__main__':
    main()
