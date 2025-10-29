#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import gradient, cos
from scipy.signal import savgol_filter

df = pd.read_csv(
    "src/kora_k3/src/vehicle_dynamics/data/speed_10000_steer_0.0_test_1.csv",
    usecols=["time","servo_position","motor_speed","imu_angular_z","imu_linear_y"]
)

class cornering_stiffness_estimation:
    def __init__(self):
        rospy.init_node("c_alpha_estimation", anonymous=True)
        self.time = df["time"].to_numpy()
        self.steering_angle = df["servo_position"].to_numpy() * (60.0 / 1.0) * (np.pi / 180.0)  # rad
        self.wheel_radius = 0.05  # m
        self.rpm_per_data = 0.025  # rpm/data
        self.Vx = df["motor_speed"].to_numpy()*self.rpm_per_data*(2.0*np.pi/60.0)*self.wheel_radius # m/s
        self.yaw_rate = df["imu_angular_z"].to_numpy()  # rad/s
        self.ay = df["imu_linear_y"].to_numpy()  # m/s^2
        self.m = 3.5      # kg
        self.mf = 1.63    # kg
        self.mr = self.m - self.mf  # kg
        self.L = 0.325
        self.lf = 0.1736    # m
        self.lr = self.L - self.lf
        self.Iz = self.mf*self.lf**2 + self.mr*self.lr**2    # kg*m^2
        
    def Vy_estimation(self):
        t = self.time
        Vydot = self.ay - self.yaw_rate*self.Vx
        Vy = np.zeros_like(t)
        if len(t) > 1:
            dt = np.diff(t, prepend=t[0])
            Vy[1:] = np.cumsum(0.5*(Vydot[1:]+Vydot[:-1])*dt) # 사다리꼴 적분
        return Vy

    def beta_estimation(self, Vy):
        beta = np.zeros_like(self.time)
        for i in range(len(self.time)):
            beta[i] = np.arctan2(Vy[i],self.Vx[i])
        return beta
    
    def slip_angle_estimation(self,Vy):
        alpha_f = np.zeros_like(self.time)
        alpha_r = np.zeros_like(self.time)
        for i in range(len(self.time)):
            if self.Vx[i] != 0:
                alpha_f[i] = self.steering_angle[i] - np.arctan2(Vy[i] + self.lf*self.yaw_rate[i], self.Vx[i])
                alpha_r[i] = -np.arctan2((Vy[i] - self.lr*self.yaw_rate[i]), self.Vx[i])
            else:
                alpha_f[i] = 0.0
                alpha_r[i] = 0.0
        return alpha_f, alpha_r
    
    def lateral_force_estimation(self):
        # 연립: [cosδ  1][Fy_f] = [m ay]
        #       [ lf  -lr][Fy_r]   [Iz rdot]
        r_s = savgol_filter(self.r, 51 if len(self.r)>60 else len(self.r)//2*2-1, 3, mode="interp")
        rdot = gradient(r_s, self.time)

        A11 = np.cos(self.steering_angle)
        A12 = np.ones_like(self.time)
        B1  = self.m*self.ay

        A21 = np.full_like(self.time, self.lf)
        A22 = np.full_like(self.time, -self.lr)
        B2  = self.Iz*rdot

        det = A11*A22 - A12*A21
        det = np.where(np.abs(det) < 1e-9, np.sign(det)*1e-9 + 1e-9, det)

        Fy_f = (B1*A22 - B2*A12) / det
        Fy_r = (A11*B2 - A21*B1) / det
        return Fy_f, Fy_r

    
    def data_scatter(self, x, y, xlabel, ylabel, title):
        plt.figure()
        plt.scatter(x, y, s=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()
    
    def preprocessing(self):
        print(self.Vx.shape)
        print(self.yaw_rate.shape)
        print(self.ay.shape)
        Vy = self.Vy_estimation()
        print(Vy.shape)
        beta = self.beta_estimation(Vy)
        alpha_f, alpha_r = self.slip_angle_estimation(Vy)
        F_yf, F_yr = self.lateral_force_estimation()
        self.data_scatter(alpha_f, F_yf, "Front Slip Angle (rad)", "Front Lateral Force (N)", "Front Cornering Stiffness")
        self.data_scatter(alpha_r, F_yr, "Rear Slip Angle (rad)", "Rear Lateral Force (N)", "Rear Cornering Stiffness")
        
def main():
    c_alpha_estimator = cornering_stiffness_estimation()
    c_alpha_estimator.preprocessing()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


