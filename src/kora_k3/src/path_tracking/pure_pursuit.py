#!/usr/bin/env python3
import csv
from math import atan, cos, sin, sqrt

import numpy as np
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import Float64


class PurePursuit:
    def __init__(self) -> None:
        rospy.init_node("pure_pursuit_node", anonymous=True)


        self.motor_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.servo_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)
        self.path_drive_pub = rospy.Publisher("/pure_pursuit/drive", AckermannDriveStamped, queue_size=1)
        self.path_steer_pub = rospy.Publisher("/pure_pursuit/steering", Float64, queue_size=1)
        self.path_speed_pub = rospy.Publisher("/pure_pursuit/speed", Float64, queue_size=1)

        self.speed_msg = Float64()
        self.steer_msg = Float64()

        self.csv_file = rospy.get_param(
            "~waypoint_file", "/root/KORA_K3/src/kora_k3/src/path_planning/outputs/waypoints.csv"
        )
        self.waypoints = self.load_waypoints()

        # Parameters
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.6)
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.35)
        self.wheelbase = rospy.get_param("~wheelbase", 0.325)
        self.target_speed = rospy.get_param("~target_speed", 1.0)

        pose_topic = rospy.get_param("~pose_topic", "/gt_pose")
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)

    def pose_callback(self, pose_msg: PoseStamped) -> None:
        yaw = self.quaternion_to_yaw(pose_msg.pose.orientation)
        pose2d = Pose2D(x=pose_msg.pose.position.x, y=pose_msg.pose.position.y, theta=yaw)
        self.base_callback(pose2d)

    def base_callback(self, odom_msg: Pose2D) -> None:
        goal_point = self.find_goal_point(odom_msg)
        steering_angle = self.calculate_steering_angle(goal_point)
        self.publish_drive_message(steering_angle)

    def find_goal_point(self, odom_msg: Pose2D):
        car_x = odom_msg.x
        car_y = odom_msg.y
        yaw = odom_msg.theta

        selected_point = None
        closest_distance = float("inf")
        fallback_point = None
        fallback_distance = -1

        for x, y in self.waypoints:
            dx = x - car_x
            dy = y - car_y
            distance = sqrt(dx**2 + dy**2)

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

        return (
            car_x + self.lookahead_distance * cos(yaw),
            car_y + self.lookahead_distance * sin(yaw),
            self.lookahead_distance,
            0.0,
            self.lookahead_distance,
        )

    def calculate_steering_angle(self, goal_point) -> float:
        lookahead = max(goal_point[4], 1e-3)
        y = goal_point[3]
        curvature = 2 * y / (lookahead**2)
        steering_angle = atan(self.wheelbase * curvature)
        return max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)

    def publish_drive_message(self, steering_angle: float) -> None:
        abs_angle = abs(steering_angle)
        if abs_angle > 0.45:
            velocity = 2000
        elif abs_angle > 0.25:
            velocity = 3000
        else:
            velocity = 3500

        if self.max_steering_angle > 1e-6:
            servo_command = -steering_angle / (2 * self.max_steering_angle) + 0.5
        else:
            servo_command = 0.5
        servo_command = max(min(servo_command, 1.0), 0.0)

        self.speed_msg.data = velocity
        self.steer_msg.data = servo_command

        self.motor_pub.publish(self.speed_msg)
        self.servo_pub.publish(self.steer_msg)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.target_speed
        self.path_drive_pub.publish(drive_msg)
        self.path_steer_pub.publish(self.steer_msg)
        self.path_speed_pub.publish(Float64(data=drive_msg.drive.speed))

    @staticmethod
    def quaternion_to_yaw(orientation_q) -> float:
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y**2 + orientation_q.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def load_waypoints(self):
        waypoints = []
        with open(self.csv_file, "r") as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                waypoints.append((float(row[0]), float(row[1])))
        return waypoints


def main() -> None:
    try:
        PurePursuit()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
