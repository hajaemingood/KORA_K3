#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd

df = pd.read_csv(
    "src/kora_k3/src/vehicle_dynamics/data/speed_10000_steer_0.0_test_1.csv",
    usecols=["time","servo_position","motor_speed","imu_angular_z","imu_linear_y"]
)

class cornering_stiffness_estimation:
    def __init__(self):
        rospy.init_node("c_alpha_estimation", anonymous=True)
        self.time = df["time"].to_numpy()
        self.steering_angle = df["servo_position"].to_numpy() * (60.0 / 1.0) * (np.pi / 180.0)  # rad
        self.Vx = df["motor_speed"].to_numpy()*0.025*(2.0*np.pi/60.0)*0.05 # m/s
        self.yaw_rate = df["imu_angular_z"].to_numpy()  # rad/s
        self.ay = df["imu_linear_y"].to_numpy()  # m/s^2
    
    def preprocessing(self):
        print(self.Vx)

def main():
    c_alpha_estimator = cornering_stiffness_estimation()
    c_alpha_estimator.preprocessing()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    

