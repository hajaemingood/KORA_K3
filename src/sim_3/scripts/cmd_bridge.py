#!/usr/bin/env python3
"""
/commands/motor/speed (Float64)와 /commands/servo/position (Float64)를
/drive (AckermannDriveStamped)로 변환해 주는 브리지 노드.
실차 제어 노드를 그대로 사용할 수 있도록 하기 위한 최소 구현.
"""

import math

import rospy
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped


class CommandBridge:
    def __init__(self) -> None:
        rospy.init_node("sim3_cmd_bridge")
        self.wheelbase = rospy.get_param("~wheelbase", 0.33)
        self.max_speed_mps = rospy.get_param("~max_speed_mps", 5.0)
        self.max_steer_rad = rospy.get_param("~max_steer_rad", 0.5)
        self.servo_center = rospy.get_param("~servo_center", 0.5)
        self.motor_scale = rospy.get_param("~motor_scale", 0.0001)  # ERPM → m/s scale (예시)

        self.motor_cmd = 0.0
        self.servo_cmd = self.servo_center

        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=10)
        rospy.Subscriber("/commands/motor/speed", Float64, self._motor_cb, queue_size=10)
        rospy.Subscriber("/commands/servo/position", Float64, self._servo_cb, queue_size=10)

        rospy.loginfo("[sim_3] command bridge ready")

    def spin(self) -> None:
        rospy.spin()

    def _motor_cb(self, msg: Float64) -> None:
        self.motor_cmd = float(msg.data)
        self._publish_drive()

    def _servo_cb(self, msg: Float64) -> None:
        self.servo_cmd = float(msg.data)
        self._publish_drive()

    def _publish_drive(self) -> None:
        drive = AckermannDriveStamped()
        drive.header.stamp = rospy.Time.now()
        drive.header.frame_id = "base_link"

        speed = self.motor_cmd * self.motor_scale
        speed = max(-self.max_speed_mps, min(self.max_speed_mps, speed))

        steer = (self.servo_cmd - self.servo_center) * (2.0 * self.max_steer_rad)
        steer = max(-self.max_steer_rad, min(self.max_steer_rad, steer))

        drive.drive.speed = speed
        drive.drive.steering_angle = steer
        self.drive_pub.publish(drive)


def main() -> None:
    try:
        CommandBridge().spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
