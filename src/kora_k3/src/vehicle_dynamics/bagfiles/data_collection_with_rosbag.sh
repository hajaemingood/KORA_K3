#!/bin/bash

# 1. rosbag record를 백그라운드로 실행
rosbag record -O speed_10000_steer_0.0_test_2.bag /imu/data_centered /imu/data /commands/motor/speed /commands/servo/position &
BAG_PID=$!   # rosbag 프로세스 ID 저장

# 2. Python 노드 실행
rosrun kora_k3 data_collection.py

# 3. Python 노드 종료 후 rosbag도 종료
kill -2 $BAG_PID
