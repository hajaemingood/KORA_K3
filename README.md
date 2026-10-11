# KORA K3 Autorace

2025 ICCAS에서 주관한 Autorace 참가를 위해 개발한 ROS1 기반 자율주행 레이싱 프로젝트입니다.  
KORA K3 차량 플랫폼에서 센서 입력, wheel odometry, EKF, mapping, localization, pure pursuit 기반 경로 추종, LiDAR 기반 obstacle avoidance를 통합해 주행하는 것을 목표로 합니다.

## 프로젝트 개요

이 저장소는 Autorace 주행을 위해 다음 기능을 포함합니다.

- IMU 및 wheel odometry 기반 차량 상태 추정
- `robot_localization` EKF를 이용한 odometry fusion
- `slam_toolbox` 기반 mapping
- `map_server`와 AMCL 기반 localization
- waypoint 시각화
- pure pursuit 기반 레이싱 라인 추종
- follow-the-gap 기반 장애물 회피
- VESC motor/servo command publish

## 주요 디렉터리

```text
src/kora_k3/
├── launch/                 # 실행 launch 파일
├── config/                 # EKF, wheel odom, RViz 설정
├── maps/                   # 저장된 map yaml/pgm
├── urdf/                   # K3 차량 모델
└── src/
    ├── sensor_hub/         # IMU publisher/filtering
    ├── odometry/           # wheel odometry node
    ├── path_tracking/      # pure pursuit, main controller, PID
    ├── obstacle/           # follow-the-gap obstacle avoidance
    ├── path_planning/      # waypoint/raceline generation
    └── vehicle_dynamics/   # vehicle dynamics data/estimation scripts
```

## 실행 전 준비

ROS workspace 루트에서 빌드 후 환경을 source 합니다.

```bash
cd /root/KORA_K3
catkin_make
source devel/setup.bash
```

Docker 환경을 사용하는 경우:

```bash
cd /root/KORA_K3/docker_setting
docker compose up -d --build
docker exec -it kora_k3 bash
cd /root/KORA_K3
source devel/setup.bash
```

GUI/RViz를 사용할 경우 호스트에서 X11 권한이 필요할 수 있습니다.

```bash
xhost +local:docker
```

## Launch 파일 정리

### 1. Robot State

```bash
roslaunch kora_k3 k3_state.launch
```

실행 내용:

- `robot_state_publisher`
- `joint_state_publisher`
- `urdf/k3.xacro` 기반 `robot_description` publish

차량의 TF/URDF 상태를 ROS에 올리는 기본 launch입니다.

### 2. Wheel Odometry + IMU + EKF

```bash
roslaunch kora_k3 wheel_odom.launch
```

실행 내용:

- `imu_pub.py`
- `wheel_odom_node.py`
- `robot_localization/ekf_localization_node`
- `config/wheel_odom.yaml`
- `config/ekf_wheel_imu.yaml`

센서와 odometry를 먼저 켜는 단계입니다. IMU와 wheel odometry를 사용해 `odom` 기준 차량 상태를 추정합니다.

### 3. Mapping

```bash
roslaunch kora_k3 mapping.launch
```

실행 내용:

- `k3_state.launch`
- `wheel_odom.launch`
- `slam_toolbox/online_async.launch`
- RViz: `config/mapping.rviz`

맵을 새로 만들 때 사용하는 launch입니다. 차량 상태, odometry, SLAM Toolbox를 함께 실행합니다.

### 4. Localization

```bash
roslaunch kora_k3 localization.launch
```

실행 내용:

- `k3_state.launch`
- `wheel_odom.launch`
- `amcl.launch`
- `waypoint_visualization.py`

저장된 맵 위에서 차량 위치를 추정하는 단계입니다. `amcl.launch` 내부에서는 `maps/real.yaml`을 `map_server`로 publish하고 AMCL을 실행합니다.

### 5. Pure Pursuit / Main Driving

```bash
roslaunch kora_k3 drive_main.launch
```

실행 내용:

- `drive_main.py`
- `base_link_pub.py`

`drive_main.py`는 실제 주행 제어의 중심 노드입니다. `/scan`과 `/base_link_pose`를 받아 pure pursuit 또는 follow-the-gap 모드를 선택하고, motor/servo command를 publish합니다.

### 6. 통합 Autorace 실행

```bash
roslaunch kora_k3 main.launch
```

실행 내용:

- `localization.launch`
- `drive_main.launch`

대회 주행에서 가장 가까운 통합 실행 파일입니다. Localization을 먼저 구성한 뒤 주행 controller를 함께 실행합니다.


### 맵 생성 시

```bash
roslaunch kora_k3 mapping.launch
```

순서상 포함되는 흐름:

```text
robot state
→ IMU / wheel odometry
→ EKF odometry fusion
→ SLAM Toolbox mapping
→ RViz mapping visualization
```

### 저장된 맵으로 위치 추정 시

```bash
roslaunch kora_k3 localization.launch
```

순서상 포함되는 흐름:

```text
robot state
→ IMU / wheel odometry
→ EKF odometry fusion
→ map_server
→ AMCL localization
→ waypoint visualization
```

### Autorace 주행 시

```bash
roslaunch kora_k3 main.launch
```

순서상 포함되는 흐름:

```text
robot state
→ IMU / wheel odometry
→ EKF odometry fusion
→ map_server + AMCL
→ base_link pose publish
→ pure pursuit path tracking
→ follow-the-gap obstacle avoidance
→ motor / servo command publish
```

## 주행 제어 구조

주행 제어의 핵심은 `src/kora_k3/src/path_tracking/drive_main.py`입니다.

입력 topic:

- `/scan`: LiDAR scan
- `/base_link_pose`: 차량 pose

출력 topic:

- `/commands/motor/speed`
- `/commands/servo/position`

제어 모드:

- `pure_pursuit`: 평상시 waypoint/raceline 추종
- `gap_follow`: 전방 장애물이 가까울 때 LiDAR gap 기반 회피

간단한 제어 흐름:

```text
1. LiDAR와 pose를 읽는다.
2. 전방 최소 장애물 거리를 계산한다.
3. 장애물이 threshold 이내면 follow-the-gap을 사용한다.
4. 장애물이 없으면 pure pursuit를 사용한다.
5. steering PID와 speed PID를 적용한다.
6. motor/servo command를 publish한다.
```

## Pure Pursuit

`src/kora_k3/src/path_tracking/pure_pursuit.py`에서 구현되어 있습니다.

주요 역할:

- waypoint CSV 로드
- 차량 현재 pose 기준 lookahead point 선택
- 차량 좌표계에서 lateral error 계산
- curvature 기반 steering command 계산
- curvature가 클수록 목표 속도 감소

여기서 `L_d`는 lookahead distance, `y`는 차량 좌표계에서 목표점의 lateral error, `kappa`는 곡률입니다.

## Follow-the-Gap

`src/kora_k3/src/obstacle/follow_the_gap.py`에서 구현되어 있습니다.

주요 역할:

- LiDAR scan smoothing
- 전방 `-45° ~ +45°` 구간 선택
- threshold보다 먼 연속 free-space 구간 탐색
- 가장 큰 gap 방향으로 steering angle 계산
- steering angle이 클수록 낮은 safe speed 선택

기본 흐름:

```text
1. LiDAR range를 moving average로 smoothing한다.
2. invalid/inf range를 큰 거리 값으로 처리한다.
3. 전방 시야각만 사용한다.
4. distance > threshold인 구간을 free space로 본다.
5. 가장 긴 free-space gap을 찾는다.
6. gap edge geometry로 steering angle을 계산한다.
7. steering angle 크기에 따라 safe speed를 정한다.
```

## 주요 파일

- `src/kora_k3/src/path_tracking/drive_main.py`: 통합 주행 controller
- `src/kora_k3/src/path_tracking/pure_pursuit.py`: pure pursuit 경로 추종
- `src/kora_k3/src/path_tracking/PID_control.py`: PID controller
- `src/kora_k3/src/obstacle/follow_the_gap.py`: obstacle avoidance
- `src/kora_k3/src/path_tracking/base_link_pub.py`: base_link pose publish
- `src/kora_k3/src/path_tracking/waypoint_visualization.py`: waypoint visualization
- `src/kora_k3/src/odometry/wheel_odom_node.py`: wheel odometry
- `src/kora_k3/src/sensor_hub/imu_pub.py`: IMU publish/filtering

## 참고

- 기본 localization map: `src/kora_k3/maps/real.yaml`
- 기본 waypoint/raceline 출력: `src/kora_k3/src/path_planning/outputs/optimal_waypoints.csv`
- 통합 실행 launch: `src/kora_k3/launch/main.launch`
