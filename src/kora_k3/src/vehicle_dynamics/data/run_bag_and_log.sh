#!/usr/bin/env bash
set -e

# =========================================
# 사용자 설정
# =========================================
BAG_PATH=${1:-"/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/bagfiles/speed_10000_steer_0.0_test_1.bag"}    # 첫 번째 인자: bag 경로
CSV_PATH=${2:-"/root/KORA_K3/src/kora_k3/src/vehicle_dynamics/data/speed_10000_steer_0.0_test_1.csv"}           # 두 번째 인자: CSV 저장 경로
RATE=${3:-1.0}                                # 세 번째 인자: 재생 배속

# =========================================
# 시뮬레이션 시간 사용 설정
# =========================================
rosparam set use_sim_time true

# =========================================
# topic_logger 실행
# =========================================
echo "[INFO] topic_logger running..."
rosrun kora_k3 rosbag_to_csv.py _csv_filename:=${CSV_PATH} &
LOGGER_PID=$!
sleep 1

# =========================================
# rosbag 재생
# =========================================
echo "[INFO] rosbag play start (${BAG_PATH})"
rosbag play --clock -r ${RATE} "${BAG_PATH}"

# =========================================
# 종료 처리
# =========================================
echo "[INFO] rosbag exit. logger stopping..."
kill ${LOGGER_PID} >/dev/null 2>&1 || true

echo "[INFO] CSV 저장 완료: ${CSV_PATH}"
