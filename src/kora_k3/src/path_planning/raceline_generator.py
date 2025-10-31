#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np

# shapely는 경계 내부 판정에 사용 (6컬럼 입력일 때)
from shapely.geometry import Polygon, LinearRing, Point
from shapely.geometry import LineString

# ---------------- 사용자 파라미터 ----------------
WAYPOINT_FILE = os.environ.get("outputs", "waypoints.npy")  # .npy 또는 .py
XI_ITERATIONS = int(os.environ.get("XI_ITERATIONS", "8"))         # 점별 이분 탐색 반복
LINE_ITERATIONS = int(os.environ.get("LINE_ITERATIONS", "500"))   # 전체 스캔 반복
# -------------------------------------------------

def load_waypoints(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        wp = np.load(path)
    elif ext == ".py":
        with open(path, "r") as f:
            code = f.read()
        # 신뢰 가능한 파일만 사용 (eval 주의)
        from numpy import array  # array(...) 인식용
        wp = eval(code)
    else:
        raise ValueError("지원하지 않는 파일 확장자: {}".format(ext))
    wp = np.asarray(wp, dtype=float)

    # 폐곡선 중복점 제거
    if len(wp) >= 2 and np.allclose(wp[0], wp[-1]):
        wp = wp[:-1]
    return wp

# 멘거 곡률(안정화 버전)
def menger_curvature_safe(p1, p2, p3, atol=1e-9):
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v21 = p1 - p2
    v23 = p3 - p2
    n21 = np.linalg.norm(v21)
    n23 = np.linalg.norm(v23)
    if n21 < atol or n23 < atol:
        return 0.0
    cosv = np.dot(v21, v23) / (n21 * n23)
    cosv = np.clip(cosv, -1.0, 1.0)
    theta = math.acos(cosv)
    # 일직선(π) 보정
    if math.isclose(theta, math.pi, rel_tol=0.0, abs_tol=1e-9):
        theta = 0.0
    d13 = np.linalg.norm(p1 - p3)
    if d13 < atol:
        return 0.0
    return 2.0 * math.sin(theta) / d13

def improve_race_line(old_line, inner_border, outer_border, xi_iters=8):
    """
    K1999에서 영감: 각 점의 곡률을 이웃 평균 곡률로 맞추며 트랙(outer-홀=inner) 영역 내에서만 이동.
    이동은 prev-nexxt 중점 방향의 선분에서 이분 탐색으로 제한.
    """
    new_line = np.array(old_line, dtype=float, copy=True)

    # 도로 폴리곤 (outer가 외곽, inner가 구멍)
    inner_ring = LinearRing(inner_border)
    outer_ring = LinearRing(outer_border)
    road_poly = Polygon(outer_ring, holes=[inner_ring])

    n = len(new_line)
    for i in range(n):
        prevprev = (i - 2) % n
        prev = (i - 1) % n
        nexxt = (i + 1) % n
        nexxtnexxt = (i + 2) % n

        xi = tuple(new_line[i])
        c_i = menger_curvature_safe(new_line[prev], xi, new_line[nexxt])
        c1 = menger_curvature_safe(new_line[prevprev], new_line[prev], xi)
        c2 = menger_curvature_safe(xi, new_line[nexxt], new_line[nexxtnexxt])
        target_c = 0.5 * (c1 + c2)

        # 이분 탐색 경계: 현위치와 이웃의 중점
        b1 = tuple(xi)
        b2 = ((new_line[nexxt][0] + new_line[prev][0]) * 0.5,
              (new_line[nexxt][1] + new_line[prev][1]) * 0.5)
        p = tuple(xi)

        for _ in range(xi_iters):
            p_c = menger_curvature_safe(new_line[prev], p, new_line[nexxt])
            if math.isclose(p_c, target_c, rel_tol=1e-3, abs_tol=1e-4):
                break

            if p_c < target_c:
                # 곡률이 모자람 → 더 굽히도록 b2 쪽을 당김
                b2 = p
                new_p = ((b1[0] + p[0]) * 0.5, (b1[1] + p[1]) * 0.5)
                if not Point(new_p).within(road_poly):
                    b1 = new_p
                else:
                    p = new_p
            else:
                # 곡률이 과함 → 펴주도록 b1 쪽을 당김
                b1 = p
                new_p = ((b2[0] + p[0]) * 0.5, (b2[1] + p[1]) * 0.5)
                if not Point(new_p).within(road_poly):
                    b2 = new_p
                else:
                    p = new_p

        new_line[i] = p

    return new_line

def main():
    wp = load_waypoints(WAYPOINT_FILE)
    track_name = os.path.splitext(os.path.basename(WAYPOINT_FILE))[0]

    if wp.shape[1] >= 6:
        # [center(xy), inner(xy), outer(xy)]
        center = wp[:, 0:2]
        inner = wp[:, 2:4]
        outer = wp[:, 4:6]

        # 최적화 시작(중복점 없는 열린 라인으로)
        raceline = center.copy()
        if np.allclose(raceline[0], raceline[-1]):
            raceline = raceline[:-1]

        for it in range(LINE_ITERATIONS):
            raceline = improve_race_line(raceline, inner, outer, xi_iters=XI_ITERATIONS)

        # 폐곡선으로 닫기
        loop_raceline = np.vstack([raceline, raceline[0]])

    elif wp.shape[1] == 2:
        # 중심선만 존재: 최적화 생략, 폐곡선만 확정
        raceline = wp.copy()
        if not np.allclose(raceline[0], raceline[-1]):
            loop_raceline = np.vstack([raceline, raceline[0]])
        else:
            loop_raceline = raceline
        print("[경고] 2컬럼 입력입니다. inner/outer 경계가 없어 최적화는 생략했습니다.")
    else:
        raise ValueError("입력 배열의 열 수가 예상과 다릅니다. (2 또는 6 컬럼 필요)")

    out_npy = f"{track_name}-raceline.npy"
    np.save(out_npy, loop_raceline)
    print(f"[완료] 저장: {out_npy}  (points={loop_raceline.shape[0]})")

    # 길이 정보(참고)
    try:
        L_center = LineString(wp[:, 0:2]).length if wp.shape[1] >= 6 else LineString(loop_raceline).length
        L_race = LineString(loop_raceline).length
        print(f"Center length: {L_center:.2f}, Raceline length: {L_race:.2f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
