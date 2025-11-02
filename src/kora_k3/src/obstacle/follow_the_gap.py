#!/usr/bin/env python3  
#-*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from math import *
import numpy as np

class Follow_the_gap:
    def __init__(self):
        rospy.init_node("follow_the_gap")
        rospy.Subscriber("/scan",LaserScan, self.lidar_CB) 
        self.motor_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1) 

        self.speed_msg = Float64()
        self.steer_msg = Float64()
     
    def lidar_CB(self,msg):
        self.scan_msg = msg
    
        angle_ranges , proc_ranges = self.preprocess_lidar(self.scan_msg)
        start_idx, end_idx = self.find_max_gap(proc_ranges)
        # print(start_idx)
        # print(end_idx)
        print(angle_ranges[start_idx]*180/pi)
        print(angle_ranges[end_idx]*180/pi)
        print("-------------")

        theta = self.calculate_angle(angle_ranges, proc_ranges, start_idx, end_idx)
        self.control(theta)

    def preprocess_lidar(self, scan_msg):
        ranges_raw = np.array(scan_msg.ranges)

        half_window = 3
        mvg_window = 2 * half_window + 1 
        ranges = np.append(np.append(np.array([np.nan]*half_window), ranges_raw), np.array([np.nan]*half_window))
        ranges = np.convolve(ranges, np.ones(mvg_window), 'valid') / (mvg_window)
        
        ranges[np.isinf(ranges) |
               (ranges > scan_msg.range_max)] = 10

        angle_ranges = np.arange(len(ranges_raw))*scan_msg.angle_increment -3*pi/4
        proc_ranges = ranges[(angle_ranges >= -75/180*pi) & (angle_ranges <= 75/180*pi)]
        angle_ranges = angle_ranges[(angle_ranges >= -75/180*pi) & (angle_ranges <= 75/180*pi)]
        # print(proc_ranges[len(proc_ranges)//4])
        # print('--------------')

        return angle_ranges, proc_ranges
    
    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges

        The max gap should not include nan 
        """
        
        start_idx = 0
        max_length = 0
        curr_length = 0
        curr_idx = 0
        threshold = 0.6 # 1.5m보다 멀리 있다면 빈공간
        for k in range(len(free_space_ranges)):
            if free_space_ranges[k] > threshold:
                curr_length +=1
    
                # New sequence, store beginning index
                if curr_length == 1:
                    curr_idx = k
            else:
                if curr_length > max_length:
                    max_length = curr_length
                    start_idx = curr_idx
                curr_length = 0
        
        if curr_length > max_length:
            max_length = curr_length
            start_idx = curr_idx

        return start_idx, start_idx + max_length - 1
    
    def calculate_angle(self, angle_ranges, proc_ranges, start_idx, end_idx):
        safety_idx = 10
        start_idx = start_idx - safety_idx
        end_idx = end_idx + safety_idx

        if start_idx < 0:
            start_idx = 0
        if end_idx > len(angle_ranges)-1:
            end_idx = len(angle_ranges)-1

        d1 = proc_ranges[start_idx]
        d2 = proc_ranges[end_idx]
        print(d1)
        print(d2)
        phi1 = abs(angle_ranges[start_idx])
        phi2 = abs(angle_ranges[end_idx])

        theta = acos((d1+d2*cos(phi1+phi2)) / sqrt(d1**2+d2**2+2*d1*d2*cos(phi1+phi2))) - phi1
        # h = sqrt(d1**2+d2**2+2*d1*d2*cos(phi1+phi2))/2

        return theta 

    def control(self, theta):
        
        if theta < -0.5:
            theta = -0.5
        elif theta > 0.5:
            theta = 0.5

        if abs(theta) > 0.35:
            motor_speed = 5000
        elif abs(theta) > 0.175:
            motor_speed = 7000
        else:
            motor_speed = 9000

        steering_angle = -(theta-0.5)
        self.speed_msg.data = motor_speed
        self.steer_msg.data = steering_angle
        # print(steering_angle)

        self.motor_pub.publish(self.speed_msg)
        self.servo_pub.publish(self.steer_msg)


def main():
    try:
        follow_the_gap = Follow_the_gap()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__=="__main__":
    main()