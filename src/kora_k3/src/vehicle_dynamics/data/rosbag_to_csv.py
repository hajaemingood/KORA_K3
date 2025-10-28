#!/usr/bin/env python3
import rospy
import csv
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from datetime import datetime

class TopicLogger:
    def __init__(self):
        rospy.init_node("topic_logger", anonymous=True)

        # 파라미터 설정 (필요시 launch 파일에서 바꿀 수 있음)
        self.csv_filename = rospy.get_param("~csv_filename", "ros_data_log.csv")

        # CSV 파일 열기
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time",
            "servo_position",
            "motor_speed",
            "imu_angular_x", "imu_angular_y", "imu_angular_z",
            "imu_linear_x", "imu_linear_y", "imu_linear_z"
        ])

        # 초기값
        self.servo = 0.0
        self.motor = 0.0
        self.imu_ang = [0.0, 0.0, 0.0]
        self.imu_lin = [0.0, 0.0, 0.0]

        # 토픽 구독
        rospy.Subscriber("/commands/servo/position", Float64, self.servo_callback)
        rospy.Subscriber("/commands/motor/speed", Float64, self.motor_callback)
        rospy.Subscriber("/imu/data_centered", Imu, self.imu_callback)

        # 주기적으로 저장
        rate = rospy.Rate(20)  # 20 Hz
        while not rospy.is_shutdown():
            self.log_data()
            rate.sleep()

    def servo_callback(self, msg):
        self.servo = msg.data

    def motor_callback(self, msg):
        self.motor = msg.data

    def imu_callback(self, msg):
        self.imu_ang = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.imu_lin = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

    def log_data(self):
        t = rospy.get_time()
        row = [
            t,
            self.servo,
            self.motor,
            self.imu_ang[0], self.imu_ang[1], self.imu_ang[2],
            self.imu_lin[0], self.imu_lin[1], self.imu_lin[2]
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 실시간 저장

    def __del__(self):
        self.csv_file.close()

if __name__ == "__main__":
    try:
        TopicLogger()
    except rospy.ROSInterruptException:
        pass
