#!/usr/bin/env python3

import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 기본 CSV 경로 (cornering_stiffness_est.py 기본값과 동일하게 유지)
DEFAULT_CSV_PATH = os.path.expanduser("/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/cornering_stiffness/ramp_steer_trial.csv")

m  = 0.3568      # kg
Iz = 0.0094    # kg*m^2
lf = 0.1736    # m
L  = 0.325
lr = L - lf

need_cols = ["time","delta_rad","Vx","ay","r"]  # 최소 필요 컬럼

def load_csv(path):
    df = pd.read_csv(path)
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError("missing column %s in %s" % (c, path))
    # 타입 정리
    for c in need_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # trial 컬럼이 있으면 정수형으로 캐스팅 시도
    if "trial" in df.columns:
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")
    return df

def iter_trial_groups_from_paths(paths):
    files = []
    for p in paths:
        files += glob.glob(p)
    files = sorted(list(set(files)))
    if not files:
        return [], []

    groups = []
    tags   = []  # 디버그/출력용 태그: (filename, trial_id)

    for f in files:
        df = load_csv(f)

        # 파일 내에 trial 컬럼이 있으면 trial 단위로 분해
        if "trial" in df.columns and df["trial"].notna().any():
            for tid, g in df.groupby("trial"):
                g = g.sort_values("time").reset_index(drop=True)
                # time이 trial마다 0부터 시작한다는 가정. 아니라도 dt 계산은 그룹 내에서 수행하므로 OK.
                # 극단적으로 샘플이 너무 적은 그룹은 스킵
                if len(g) < 21:
                    continue
                groups.append(g)
                tags.append((os.path.basename(f), int(tid) if pd.notna(tid) else -1))
        else:
            # trial 없으면 파일 단위 하나의 그룹으로 처리
            g = df.sort_values("time").reset_index(drop=True)
            if len(g) < 21:
                continue
            groups.append(g)
            tags.append((os.path.basename(f), None))

    return groups, tags

def build_regression_blocks(df):
    # df: 단일 trial 그룹
    t = df["time"].values.astype(float)
    # dt 추정(잡음 방지용 중앙값)
    if len(t) >= 3:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.01
    else:
        dt = 0.01

    delta = df["delta_rad"].values.astype(float)
    Vx    = df["Vx"].values.astype(float)
    ay    = df["ay"].values.astype(float)
    r     = df["r"].values.astype(float)

    # Savitzky-Golay 필터 파라미터
    # 창 길이: 약 0.2s, 홀수 보장
    win = max(21, int(max(0.2, 5*dt)/dt) | 1)
    poly = 3
    if win >= len(r):  # 창이 시계열보다 길면 축소
        win = (len(r) // 2) * 2 + 1
        win = max(win, 5)
        poly = min(poly, win-2)

    r_f   = savgol_filter(r,   win, poly, mode="interp")
    ay_f  = savgol_filter(ay,  win, poly, mode="interp")
    r_dot = savgol_filter(r,   win, poly, deriv=1, delta=dt, mode="interp")

    eps = 1e-6
    V = np.maximum(Vx, eps)

    # 회귀 행렬 (v_y 제거형; 합의한 φ 구성)
    phi11 = 2.0*lf*( delta - (lf/V)*r_f )
    phi12 = 2.0*lr*( (lr/V)*r_f )
    phi21 = 2.0*( delta - (lf/V)*r_f )
    phi22 = 2.0*( (lr/V)*r_f )

    y1 = Iz * r_dot
    y2 = m  * ay_f

    # 유효 구간 마스크
    valid = (V > 0.1) & np.isfinite(delta) & np.isfinite(r_f) & np.isfinite(ay_f) & np.isfinite(r_dot)

    Phi1 = np.stack([phi11, -phi12], axis=1)[valid]
    Phi2 = np.stack([phi21,  phi22], axis=1)[valid]
    Y1   = y1[valid]
    Y2   = y2[valid]

    # 2개의 식을 세로로 붙이기
    Phi = np.vstack([Phi1, Phi2])
    Y   = np.hstack([Y1, Y2])

    # 속도 샘플도 반환(속도별 피팅용)
    return Phi, Y, V[valid]

def fit_global_constant(Phi_all, Y_all):
    theta, *_ = np.linalg.lstsq(Phi_all, Y_all, rcond=None)
    Caf, Car = theta[0], theta[1]
    return Caf, Car

def fit_quadratic_vs_speed(dfs):
    # C_af(V)=a0+a1V+a2V^2, C_ar도 동일형태
    blocks = []
    Ys = []
    for df in dfs:
        t = df["time"].values.astype(float)
        if len(t) >= 3:
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dt = 0.01
        else:
            dt = 0.01

        delta = df["delta_rad"].values.astype(float)
        Vx = df["Vx"].values.astype(float)
        ay = df["ay"].values.astype(float)
        r  = df["r"].values.astype(float)

        win = max(21, int(max(0.2, 5*dt)/dt) | 1)
        poly=3
        if win >= len(r):
            win = (len(r) // 2) * 2 + 1
            win = max(win, 5)
            poly = min(poly, win-2)

        r_f   = savgol_filter(r,   win, poly, mode="interp")
        ay_f  = savgol_filter(ay,  win, poly, mode="interp")
        r_dot = savgol_filter(r,   win, poly, deriv=1, delta=dt, mode="interp")

        V = np.maximum(Vx, 1e-6)

        phi11 = 2.0*lf*( delta - (lf/V)*r_f )
        phi12 = 2.0*lr*( (lr/V)*r_f )
        phi21 = 2.0*( delta - (lf/V)*r_f )
        phi22 = 2.0*( (lr/V)*r_f )

        y1 = Iz * r_dot
        y2 = m  * ay_f

        valid = (V > 0.1) & np.isfinite(delta) & np.isfinite(r_f) & np.isfinite(ay_f) & np.isfinite(r_dot)

        v   = V[valid]
        p11 = phi11[valid]; p12 = phi12[valid]; p21 = phi21[valid]; p22 = phi22[valid]
        y1v = y1[valid];    y2v = y2[valid]

        Y = np.hstack([y1v, y2v])

        # 설계행렬 (6개의 계수: Caf a0,a1,a2 / Car b0,b1,b2)
        X1 = np.column_stack([p11*np.ones_like(v), p11*v, p11*v*v,
                              -p12*np.ones_like(v), -p12*v, -p12*v*v])
        X2 = np.column_stack([p21*np.ones_like(v), p21*v, p21*v*v,
                               p22*np.ones_like(v),  p22*v,  p22*v*v])
        X  = np.vstack([X1, X2])

        blocks.append(X)
        Ys.append(Y)

    if not blocks:
        raise RuntimeError("no valid samples for polynomial fit")
    Xall = np.vstack(blocks)
    Yall = np.hstack(Ys)
    coeff, *_ = np.linalg.lstsq(Xall, Yall, rcond=None)
    return coeff  # [a0,a1,a2, b0,b1,b2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_poly", action="store_true", help="속도 의존 2차 다항(C(V)) 피팅")
    ap.add_argument(
        "paths",
        nargs="*",
        help="CSV 파일 경로(여러 개 또는 와일드카드). 미지정 시 ramp_steer_trial.csv 기본 경로를 사용."
    )
    args = ap.parse_args()

    paths = args.paths or [DEFAULT_CSV_PATH]
    resolved = []
    for p in paths:
        expanded = os.path.expanduser(p)
        resolved.append(expanded)

    if not args.paths and not os.path.exists(resolved[0]):
        ap.error(
            f"기본 CSV 경로({resolved[0]})를 찾을 수 없습니다. "
            "cornering_stiffness_est 노드로 파일을 생성했는지 확인하거나, 직접 경로를 인자로 넘겨주세요."
        )

    groups, tags = iter_trial_groups_from_paths(resolved)
    if not groups:
        print("No valid groups (trials/files) found.")
        return

    # 1) 글로벌 상수 피팅
    if not args.fit_poly:
        Phi_all = []
        Y_all = []
        n_rows = 0
        for g in groups:
            Phi, Y, _V = build_regression_blocks(g)
            if len(Y) == 0:
                continue
            Phi_all.append(Phi); Y_all.append(Y)
            n_rows += len(Y)
        if not Phi_all:
            print("No valid samples for global constant fit.")
            return
        Phi_all = np.vstack(Phi_all)
        Y_all   = np.hstack(Y_all)
        Caf, Car = fit_global_constant(Phi_all, Y_all)
        print("===== Global constant fit over {} groups ({} samples) =====".format(len(groups), n_rows))
        print("C_af = {:.3f}  N/rad".format(Caf))
        print("C_ar = {:.3f}  N/rad".format(Car))

    # 2) 속도 의존 2차 다항 피팅
    else:
        coeff = fit_quadratic_vs_speed(groups)
        a0,a1,a2,b0,b1,b2 = coeff.tolist()
        print("===== Quadratic speed-fit (C(V)=a0+a1 V + a2 V^2) over {} groups =====".format(len(groups)))
        print("C_af(V) = {:.3f} + {:.3f} V + {:.3f} V^2   [N/rad]".format(a0,a1,a2))
        print("C_ar(V) = {:.3f} + {:.3f} V + {:.3f} V^2   [N/rad]".format(b0,b1,b2))

if __name__ == "__main__":
    main()
