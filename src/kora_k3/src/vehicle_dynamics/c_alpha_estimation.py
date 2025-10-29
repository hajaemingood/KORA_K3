#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import gradient
from scipy.signal import savgol_filter

df = pd.read_csv(
    "/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/data/speed_6000_steer_1.0_test_2.csv",
    usecols=["time","servo_position","motor_speed","imu_angular_z","imu_linear_y"]
)

class cornering_stiffness_estimation:
    def __init__(self):
        rospy.init_node("c_alpha_estimation", anonymous=True)
        self.time = df["time"].to_numpy()
        self.steering_angle = df["servo_position"].to_numpy() # rad
        self.wheel_radius = 0.05  # m
        self.rpm_per_data = 0.025  # rpm/data
        self.Vx = df["motor_speed"].to_numpy()*self.rpm_per_data*(2.0*np.pi/60.0)*self.wheel_radius # m/s
        self.yaw_rate = df["imu_angular_z"].to_numpy()  # rad/s
        self.m = 3.5      # kg
        self.mf = 1.63    # kg
        self.mr = self.m - self.mf  # kg
        self.L = 0.325
        self.lf = 0.1736    # m
        self.lr = self.L - self.lf
        self.Iz = self.mf*self.lf**2 + self.mr*self.lr**2    # kg*m^2
        self.ay= df["imu_linear_y"].to_numpy()  # m/s^2
    
    def steering_angle_conversion(self, steering_angle):
        steer_deg_range = 60.0 # 0~1이 ±60°
        scaling = self.steering_angle*steer_deg_range*2
        scaling = -(scaling - steer_deg_range)
        steering_angle= scaling * (np.pi/180.0)
        return steering_angle

    def Vy_estimation(self):
        t = self.time
        Vydot = self.ay - self.yaw_rate*self.Vx
        Vy = np.zeros_like(t)
        if len(t) > 1:
            dt = np.diff(t)
            Vy[1:] = np.cumsum(0.5*(Vydot[1:]+Vydot[:-1])*dt) # 사다리꼴 적분
        return Vy

    def beta_estimation(self, Vy):
        beta = np.zeros_like(self.time)
        for i in range(len(self.time)):
            beta[i] = np.arctan2(Vy[i],self.Vx[i])
        return beta
    
    def slip_angle_estimation(self, Vy, steering_angle):
        alpha_f = np.zeros_like(self.time)
        alpha_r = np.zeros_like(self.time)
        for i in range(len(self.time)):
            if self.Vx[i] != 0:
                alpha_f[i] = steering_angle[i] - np.arctan2(Vy[i] + self.lf*self.yaw_rate[i], self.Vx[i])
                alpha_r[i] = -np.arctan2((Vy[i] - self.lr*self.yaw_rate[i]), self.Vx[i])
            else:
                alpha_f[i] = 0.0
                alpha_r[i] = 0.0
        return alpha_f, alpha_r
    
    def lateral_force_estimation(self,steering_angle):
        # 연립: [cosδ  1][Fy_f] = [m ay]
        #       [ lf  -lr][Fy_r]   [Iz rdot]
        r_raw = self.yaw_rate
        if len(r_raw) >= 3:
            window = 51 if len(r_raw) > 60 else len(r_raw)
            if window % 2 == 0:
                window -= 1
            window = max(3, min(window, len(r_raw) if len(r_raw) % 2 == 1 else len(r_raw) - 1))
            r_s = savgol_filter(r_raw, window, 3, mode="interp")
        else:
            r_s = r_raw       
        
        rdot = gradient(r_s, self.time)

        A11 = np.cos(steering_angle)
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

    def data_anaysis(self, value):
        value = value[~np.isnan(value)]
        value_mean = np.mean(value)
        value_std = np.std(value)
        value_midean = np.median(value)
        return value_mean, value_std, value_midean


    def data_scatter(self, x, y, xlabel, ylabel, title):
        plt.figure()
        plt.scatter(x, y, s=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()

    
    def preprocessing(self):
        steering_angle = self.steering_angle_conversion(self.steering_angle)
        #print(self.Vx.shape)
        #print(steering_angle)
        #print(self.yaw_rate)
        #print(self.ay)
        Vy = self.Vy_estimation()
        # print(Vy.shape)
        # beta = self.beta_estimation(Vy)
        alpha_f, alpha_r = self.slip_angle_estimation(Vy, steering_angle)
        alpha_f_mean,alpha_f_std, alpha_f_mid = self.data_anaysis(alpha_f)
        alpha_r_mean,alpha_r_std, alpha_r_mid = self.data_anaysis(alpha_r)
        
        F_yf, F_yr = self.lateral_force_estimation(steering_angle)
        F_yf_mean,F_yf_std,F_yf_mid = self.data_anaysis(F_yf)
        F_yr_mean,F_yr_std,F_yr_mid = self.data_anaysis(F_yr)
        
        print(f"alpha_f :: mean : {alpha_f_mean}, std : {alpha_f_std}, median : {alpha_f_mid}")
        print(f"alpha_r :: mean : {alpha_r_mean}, std : {alpha_r_std}, median : {alpha_r_mid}")
        print(f"F_yf :: mean : {F_yf_mean}, std : {F_yf_std}, median : {F_yf_mid}")
        print(f"F_yr :: mean : {F_yr_mean}, std : {F_yr_std}, median : {F_yr_mid}")
        #print(f"F_yf: {F_yf}, F_yr: {F_yr}")
        #print(f"alpha_f: {alpha_f}, alpha_r: {alpha_r}")
        self.data_scatter(alpha_f, F_yf, "Front Slip Angle (rad)", "Front Lateral Force (N)", "Front Cornering Stiffness")
        self.data_scatter(alpha_r, F_yr, "Rear Slip Angle (rad)", "Rear Lateral Force (N)", "Rear Cornering Stiffness")
        # c_alpha_f, intercept_f = self.least_squares_fit(alpha_f, F_yf)
        # c_alpha_r, intercept_r = self.least_squares_fit(alpha_r, F_yr)
        # print(f"Front Cornering Stiffness: {c_alpha_f:.2f} N/rad, Intercept: {intercept_f:.2f} N")
        # print(f"Rear Cornering Stiffness: {c_alpha_r:.2f} N/rad, Intercept: {intercept_r:.2f} N")   

def main():
    c_alpha_estimator = cornering_stiffness_estimation()
    c_alpha_estimator.preprocessing()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


