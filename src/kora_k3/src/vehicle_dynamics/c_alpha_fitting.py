#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_1 = pd.read_csv(
    "/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/data/F_yf_C_alpha_f_pair.csv",
    usecols=["alpha_f","F_yf"]
)

df_2 = pd.read_csv(
    "/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/data/F_yr_C_alpha_r_pair.csv",
    usecols=["alpha_r","F_yr"]
)

class cornering_stiffness_fitting:
    def __init__(self):
        rospy.init_node("c_alpha_fitting", anonymous=True)

        # 문자열 → float 변환 + NaN 제거
        self.alpha_f = pd.to_numeric(df_1["alpha_f"], errors="coerce").dropna().to_numpy()
        self.F_yf    = pd.to_numeric(df_1["F_yf"],    errors="coerce").dropna().to_numpy()
        self.alpha_r = pd.to_numeric(df_2["alpha_r"], errors="coerce").dropna().to_numpy()
        self.F_yr    = pd.to_numeric(df_2["F_yr"],    errors="coerce").dropna().to_numpy()

        # fitting 범위 설정 (작은 slip angle만 사용)
        self.alpha_window = float(rospy.get_param("~alpha_window", 0.05))  # rad

    def fit_linear_origin(self, x, y):
        """최소제곱법(원점 통과): F_y = k * alpha"""
        mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= self.alpha_window)
        x_sel, y_sel = x[mask], y[mask]

        if len(x_sel) < 2 or np.allclose(np.dot(x_sel, x_sel), 0):
            rospy.logwarn("유효한 데이터가 부족하거나 분모가 0입니다.")
            return np.nan

        k = np.dot(x_sel, y_sel) / np.dot(x_sel, x_sel)
        return k

    def data_fitting_plot(self, x, y, xlabel, ylabel, title):
        k = self.fit_linear_origin(x, y)
        if np.isnan(k):
            rospy.logwarn(f"{title}: 유효하지 않은 fitting (NaN)")
            return

        x_fit = np.linspace(np.min(x), np.max(x), 100)
        y_fit = k * x_fit

        plt.figure()
        plt.scatter(x, y, s=2, label="Measured Data")
        plt.plot(x_fit, y_fit, color='red', linewidth=2,
                 label=f"Least Squares Fit (k = {k:.3f} N/rad)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

        rospy.loginfo(f"{title} cornering stiffness k = {k:.3f} [N/rad]")

    def data_fitting(self):
        self.data_fitting_plot(
            self.alpha_f, self.F_yf,
            "Front Slip Angle (rad)", "Front Lateral Force (N)",
            "Front Cornering Stiffness"
        )
        self.data_fitting_plot(
            self.alpha_r, self.F_yr,
            "Rear Slip Angle (rad)", "Rear Lateral Force (N)",
            "Rear Cornering Stiffness"
        )

def main():
    c_alpha_fitter = cornering_stiffness_fitting()
    c_alpha_fitter.data_fitting()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
