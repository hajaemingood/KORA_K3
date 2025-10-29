#!/usr/bin/env python3
import rospy
import numpy as np
import csv
import sys
from math import atan, cos, sin, sqrt
from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped

class Pure_pursuit:
    def __init__(self):
        rospy.init_node("pure_pursuit_node", anonymous=True)
        pose_topic = rospy.get_param("~pose_topic", "/gt_pose")
        rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)
        self.motor_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)
        self.path_drive_pub = rospy.Publisher('/pure_pursuit/drive', AckermannDriveStamped, queue_size=1)
        self.path_steer_pub = rospy.Publisher('/pure_pursuit/steering', Float64, queue_size=1)
        self.path_speed_pub = rospy.Publisher('/pure_pursuit/speed', Float64, queue_size=1)
        
        self.speed_msg = Float64()
        self.steer_msg = Float64()
        

        self.csv_file = '/root/KORA_K3/src/kora_k3/src/path_planning/outputs/waypoints.csv'
        self.waypoints = self.load_waypoints()
        # Parameters
        self.lookahead_distance = 0.8  # Lookahead distance for Pure Pursuit
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.35)  # radians
        self.wheelbase = rospy.get_param("~wheelbase", 0.325)  # meters
        self.target_speed = rospy.get_param("~target_speed", 1.0)  # m/s
        

    def pose_callback(self, pose_msg: PoseStamped):
        yaw = self.quaternion_to_yaw(pose_msg.pose.orientation)
        pose2d = Pose2D(x=pose_msg.pose.position.x, y=pose_msg.pose.position.y, theta=yaw)
        self.base_callback(pose2d)

    def base_callback(self, odom_msg: Pose2D):
        # 1. Find the current waypoint to track
        # 2. Transform the goal point to the vehicle frame
        goal_point = self.find_goal_point(odom_msg)
        print(goal_point)
        # 3. Calculate curvature (steering angle)
        steering_angle = self.calculate_steering_angle(goal_point)
        # print(steering_angle)
        # 4. Publish the drive message
        self.publish_drive_message(steering_angle)

    def imu_callback(self, imu_msg):
        self.imu_msg = imu_msg

    def find_goal_point(self, odom_msg):
        # 현재 차량 위치
        car_x = odom_msg.x                    # car x
        car_y = odom_msg.y                       # car y
        yaw = odom_msg.theta                  # car yaw

        # 목표 경로점 리스트 초기화
        selected_point = None
        closest_distance = float("inf")
        fallback_point = None
        fallback_distance = -1

        for x, y in self.waypoints:
            # 차량과 경로점 간의 거리 계산
            dx = x - car_x
            dy = y - car_y
            distance = sqrt(dx ** 2 + dy ** 2)

            rotated_x = cos(-yaw) * dx - sin(-yaw) * dy
            rotated_y = sin(-yaw) * dx + cos(-yaw) * dy

            if rotated_x > 0:
                if distance >= self.lookahead_distance and distance < closest_distance:
                    closest_distance = distance
                    selected_point = (x, y, rotated_x, rotated_y, distance)
                if distance > fallback_distance:
                    fallback_distance = distance
                    fallback_point = (x, y, rotated_x, rotated_y, distance)

        if selected_point is not None:
            return selected_point
        if fallback_point is not None:
            return fallback_point
        # fallback: stay aligned forward with nominal lookahead
        return (car_x + self.lookahead_distance * cos(yaw), car_y + self.lookahead_distance * sin(yaw),
                self.lookahead_distance, 0.0, self.lookahead_distance)

    def calculate_steering_angle(self, goal_point):
        # Calculate the curvature using Pure Pursuit formula
        L = max(goal_point[4], 1e-3)
        y = goal_point[3]
        curvature = 2 * y / (L ** 2)  # Curvature formula

        steering_angle = atan(self.wheelbase * curvature)
        steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)

        return steering_angle

    def publish_drive_message(self, steering_angle):
        # Create and publish the Ackermann drive message

        # Lookup table for hardware command (if needed)
        abs_angle = abs(steering_angle)
        if abs_angle > 0.45:
            velocity = 7000
        elif abs_angle > 0.25:
            velocity = 9000
        else:
            velocity = 15000

        # Map steering angle (rad) to servo command (0.0 ~ 1.0)
        if self.max_steering_angle > 1e-6:
            servo_command = -steering_angle / (2 * self.max_steering_angle) + 0.5
        else:
            servo_command = 0.5
        servo_command = max(min(servo_command, 1.0), 0.0)

        self.speed_msg.data = velocity
        self.steer_msg.data = servo_command

        self.motor_pub.publish(self.speed_msg)
        self.servo_pub.publish(self.steer_msg)

        # Publish path tracking command for other modules (e.g., FTG blending)
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.target_speed
        self.path_drive_pub.publish(drive_msg)
        self.path_steer_pub.publish(self.steer_msg)
        self.path_speed_pub.publish(Float64(data=drive_msg.drive.speed))

    @staticmethod
    def quaternion_to_yaw(orientation_q):
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y ** 2 + orientation_q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def load_waypoints(self):
        """CSV 파일에서 웨이포인트를 읽어 리스트로 반환"""
        waypoints = []
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # 헤더 건너뛰기 (필요 시)
            for row in reader:
                # x, y 좌표 추출
                x = float(row[0])
                y = float(row[1])
                waypoints.append((x, y))
        return waypoints

def main():
    try:
        pure_pursuit = Pure_pursuit()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
